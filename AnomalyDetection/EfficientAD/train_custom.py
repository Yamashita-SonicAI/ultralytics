#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_custom.py
anomalib v2.2.0 の EfficientAD をカスタム画像フォルダで学習するスクリプト。

MVTec 形式ではなく、正常画像のみのフォルダを指定して学習する。
正常画像の一部を自動で validation/test に分割する。

実行例:
  python train_custom.py --name yellow_bowl \
      --normal_dir captured/yellow_bowl

  python train_custom.py --name yellow_bowl \
      --normal_dir captured/yellow_bowl \
      --max_epochs 300 --image_size 256
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import EfficientAd


DEFAULT_BASE_OUT = Path("/home/sonicai/workspace/yamada-develop/yolo-dev/AnomalyDetection/EfficientAD")
DEFAULT_DATASET_DIR = DEFAULT_BASE_OUT / "datasets"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--name", type=str, required=True,
                    help="データセット名（出力ディレクトリ名にもなる）")
    ap.add_argument("--normal_dir", type=str, required=True,
                    help="正常画像のフォルダパス")
    ap.add_argument("--abnormal_dir", type=str, default=None,
                    help="異常画像のフォルダパス（テスト評価用、省略可）")

    ap.add_argument("--imagenette_dir", type=str, default=str(DEFAULT_DATASET_DIR / "imagenette"),
                    help="ImageNette dir（なければ自動ダウンロード）")

    ap.add_argument("--image_size", type=int, default=256, help="train/eval resize (square)")
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--num_workers", type=int, default=8)

    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--precision", type=str, default="16-mixed",
                    choices=["16-mixed", "bf16-mixed", "32"])

    ap.add_argument("--model_size", type=str, default="small", choices=["small", "medium"])
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--normal_split_ratio", type=float, default=0.2,
                    help="正常画像のうち test/val に回す割合（デフォルト0.2=20%%）")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--out_base", type=str, default=str(DEFAULT_BASE_OUT))

    return ap.parse_args()


def build_augmentations(image_size: int) -> Compose:
    """
    torchvision.transforms.v2 でリサイズ + テンソル変換を行う。
    """
    return Compose([
        ToImage(),
        Resize((image_size, image_size), antialias=True),
        ToDtype(torch.float32, scale=True),
    ])


def main():
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Falling back to CPU.")
        device = "cpu"

    normal_dir = Path(args.normal_dir).expanduser().resolve()
    if not normal_dir.exists():
        raise FileNotFoundError(f"normal_dir が見つかりません: {normal_dir}")

    imagenette_dir = Path(args.imagenette_dir).expanduser().resolve()

    out_base = Path(args.out_base).expanduser().resolve()
    out_cat = out_base / args.name
    train_root = out_cat / "train_logs"
    export_dir = out_cat / "export"
    ensure_dir(train_root)
    ensure_dir(export_dir)

    # 1) DataModule
    #    Folder: カスタムフォルダ用の DataModule。
    #    normal_dir のみ指定すると、normal_split_ratio の割合で test/val に自動分割。
    #    test_split_mode="from_dir" だと abnormal_dir が必要になるため、
    #    正常画像のみの場合は "synthetic" で擬似異常を生成するか、
    #    "none" でテストをスキップする。
    augmentations = build_augmentations(args.image_size)

    datamodule = Folder(
        name=args.name,
        normal_dir=str(normal_dir),
        abnormal_dir=str(Path(args.abnormal_dir).expanduser().resolve()) if args.abnormal_dir else None,
        normal_split_ratio=args.normal_split_ratio,
        train_batch_size=1,       # EfficientAD は batch_size=1 固定
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        augmentations=augmentations,
        test_split_mode="from_dir" if args.abnormal_dir else "synthetic",
        val_split_mode="from_test",
        val_split_ratio=0.5,
        seed=args.seed,
    )

    # 2) Model
    model = EfficientAd(
        imagenet_dir=str(imagenette_dir),
        model_size=args.model_size,
    )

    # 3) Engine
    engine = Engine(
        default_root_dir=str(train_root),
        max_epochs=args.max_epochs,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        precision=args.precision,
    )

    print("========== TRAIN CONFIG ==========")
    print(f"name          : {args.name}")
    print(f"normal_dir    : {normal_dir}")
    print(f"abnormal_dir  : {args.abnormal_dir or '(なし)'}")
    print(f"imagenette_dir: {imagenette_dir}")
    print(f"image_size    : {args.image_size}")
    print(f"model_size    : {args.model_size}")
    print(f"epochs        : {args.max_epochs}")
    print(f"device        : {device} (precision={args.precision})")
    print(f"split_ratio   : {args.normal_split_ratio}")
    print(f"save_root     : {train_root}")
    print("==================================")

    # 4) Train
    engine.fit(model=model, datamodule=datamodule)

    # 5) ckpt paths
    best_ckpt = None
    last_ckpt = None
    try:
        best_ckpt = engine.trainer.checkpoint_callback.best_model_path
        last_ckpt = engine.trainer.checkpoint_callback.last_model_path
    except Exception:
        pass

    print(f"[INFO] best_ckpt: {best_ckpt}")
    print(f"[INFO] last_ckpt: {last_ckpt}")

    if best_ckpt:
        (export_dir / "best_ckpt_path.txt").write_text(str(best_ckpt) + "\n", encoding="utf-8")
    if last_ckpt:
        (export_dir / "last_ckpt_path.txt").write_text(str(last_ckpt) + "\n", encoding="utf-8")

    # 6) Test
    test_results = engine.test(model=model, datamodule=datamodule, ckpt_path=best_ckpt if best_ckpt else None)
    print("[INFO] test_results:", test_results)

    print(f"[DONE] outputs: {out_cat}")
    print(f"[DONE] ckpt files: {export_dir}")


if __name__ == "__main__":
    main()
