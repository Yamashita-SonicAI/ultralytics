#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_MVTec.py
anomalib v2.2.0 の EfficientAD を MVTec AD で学習 → test評価 → ckpt保存

データセット:
  - MVTec AD : --mvtec_root で指定。存在しなければ anomalib が自動ダウンロードする
  - ImageNette: --imagenette_dir で指定。存在しなければ EfficientAd が自動ダウンロードする

保存先（固定）:
  /home/sonicai/workspace/yamada-develop/yolo-dev/AnomalyDetection/EfficientAD
  └── <category>/
      ├── train_logs/   (lightning_logs, checkpoints 等)
      └── export/
          ├── best_ckpt_path.txt
          └── last_ckpt_path.txt

実行例:
  python train_MVTec.py --category bottle
  python train_MVTec.py --category bottle --max_epochs 300 --model_size m
  python train_MVTec.py --category bottle --precision 32 --device cpu

注意:
  - EfficientAD は train_batch_size=1 固定（論文準拠）
  - anomalib v2.2.0 の MVTecAD は image_size 引数を取らない（augmentations で制御）
  - EfficientAd は学習時に ImageNette も使う（teacher-student 学習用）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd


DEFAULT_BASE_OUT = Path("/home/sonicai/workspace/yamada-develop/yolo-dev/AnomalyDetection/EfficientAD")
# データセットのデフォルト保存先（EfficientAD ディレクトリ配下にまとめる）
DEFAULT_DATASET_DIR = DEFAULT_BASE_OUT / "datasets"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--category", type=str, required=True, help="bottle, hazelnut, transistor, ...")

    # データセットのパス（未ダウンロードなら自動取得される）
    ap.add_argument("--mvtec_root", type=str, default=str(DEFAULT_DATASET_DIR / "MVTecAD"),
                    help="MVTec AD root dir（なければ自動ダウンロード）")
    ap.add_argument("--imagenette_dir", type=str, default=str(DEFAULT_DATASET_DIR / "imagenette"),
                    help="ImageNette dir（なければ自動ダウンロード）")

    ap.add_argument("--image_size", type=int, default=256, help="train/eval resize (square)")
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--num_workers", type=int, default=8)

    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--precision", type=str, default="16-mixed",
                    choices=["16-mixed", "bf16-mixed", "32"],
                    help="Trainer precision（GPU推奨: 16-mixed）")

    ap.add_argument("--model_size", type=str, default="small", choices=["small", "medium"],
                    help="EfficientAD モデルサイズ (small or medium)")
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)

    # 出力先（デフォルトは固定パス）
    ap.add_argument("--out_base", type=str, default=str(DEFAULT_BASE_OUT))

    return ap.parse_args()


def build_augmentations(image_size: int) -> Compose:
    """
    anomalib v2.2.0 では MVTecAD に image_size 引数がないため、
    torchvision.transforms.v2 の Compose でリサイズを指定する。
    - Resize: 画像を (H, W) にリサイズ
    - ToImage + ToDtype: テンソル変換と正規化（0-1 float）
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

    mvtec_root = Path(args.mvtec_root).expanduser().resolve()
    imagenette_dir = Path(args.imagenette_dir).expanduser().resolve()

    out_base = Path(args.out_base).expanduser().resolve()
    out_cat = out_base / args.category
    train_root = out_cat / "train_logs"
    export_dir = out_cat / "export"
    ensure_dir(train_root)
    ensure_dir(export_dir)

    # 1) DataModule
    #    anomalib v2.2.0 の MVTecAD は image_size パラメータを持たない。
    #    代わりに augmentations で前処理パイプラインを渡す。
    #    データが存在しなければ prepare_data() で自動ダウンロードされる。
    augmentations = build_augmentations(args.image_size)
    datamodule = MVTecAD(
        root=str(mvtec_root),
        category=args.category,
        train_batch_size=1,       # EfficientAD は batch_size=1 固定（論文準拠）
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        augmentations=augmentations,
        seed=args.seed,
    )

    # 2) Model
    #    imagenet_dir: teacher-student 学習で使う ImageNette のパス。
    #                  存在しなければ自動ダウンロードされる。
    #    model_size: "s" (small) or "m" (medium)
    model = EfficientAd(
        imagenet_dir=str(imagenette_dir),
        model_size=args.model_size,
    )

    # 3) Engine (Lightning Trainer のラッパー)
    #    **kwargs は内部で Lightning Trainer にそのまま渡される。
    #    precision: "16-mixed" = FP16混合精度（GPU推奨）、"32" = FP32
    engine = Engine(
        default_root_dir=str(train_root),
        max_epochs=args.max_epochs,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        precision=args.precision,
    )

    print("========== TRAIN CONFIG ==========")
    print(f"mvtec_root    : {mvtec_root}")
    print(f"imagenette_dir: {imagenette_dir}")
    print(f"category      : {args.category}")
    print(f"image_size    : {args.image_size}")
    print(f"model_size    : {args.model_size}")
    print(f"epochs        : {args.max_epochs}")
    print(f"device        : {device} (precision={args.precision})")
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

    # 保存（推論側がコピペで使えるように）
    if best_ckpt:
        (export_dir / "best_ckpt_path.txt").write_text(str(best_ckpt) + "\n", encoding="utf-8")
    if last_ckpt:
        (export_dir / "last_ckpt_path.txt").write_text(str(last_ckpt) + "\n", encoding="utf-8")

    # 6) Test（MVTec test splitで評価）
    test_results = engine.test(model=model, datamodule=datamodule, ckpt_path=best_ckpt if best_ckpt else None)
    print("[INFO] test_results:", test_results)

    print(f"[DONE] category outputs: {out_cat}")
    print(f"[DONE] ckpt path files  : {export_dir}")


if __name__ == "__main__":
    main()
