#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocess_tiles.py
学習用画像をタイル（パッチ）に分割して保存する前処理スクリプト。

推論時のタイルサイズと学習時の入力スケールを揃えることで、
異常検出の精度向上を狙う。

実行例:
  python preprocess_tiles.py \
      --input_dir captured/20260303_182148 \
      --output_dir captured/20260303_182148_patches \
      --tile_size 256 --overlap 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True,
                    help="元画像のフォルダパス")
    ap.add_argument("--output_dir", type=str, required=True,
                    help="パッチ保存先フォルダパス")
    ap.add_argument("--tile_size", type=int, default=256,
                    help="切り出すタイルの1辺サイズ（デフォルト: 256）")
    ap.add_argument("--overlap", type=int, default=64,
                    help="タイル間のオーバーラップ（デフォルト: 64）")
    ap.add_argument("--min_size", type=int, default=128,
                    help="端の切れ端がこのサイズ未満なら破棄（デフォルト: 128）")
    return ap.parse_args()


def generate_tiles(H: int, W: int, tile: int, overlap: int):
    """タイル座標を生成する（infer_MVTec.py と同じロジック）"""
    stride = tile - overlap
    xs = list(range(0, max(1, W - tile + 1), stride))
    ys = list(range(0, max(1, H - tile + 1), stride))

    if not xs:
        xs = [0]
    if not ys:
        ys = [0]

    # 画像端をカバー
    if xs[-1] != max(0, W - tile):
        xs.append(max(0, W - tile))
    if ys[-1] != max(0, H - tile):
        ys.append(max(0, H - tile))

    tiles = []
    for y in ys:
        for x in xs:
            x2 = min(x + tile, W)
            y2 = min(y + tile, H)
            tiles.append((x, y, x2, y2))
    return tiles


def main():
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_size = args.tile_size
    overlap = args.overlap
    min_size = args.min_size

    # 画像ファイル一覧
    images = sorted(
        p for p in input_dir.rglob("*") if p.suffix.lower() in IMG_EXTS
    )
    if not images:
        raise FileNotFoundError(f"画像が見つかりません: {input_dir}")

    print(f"[INFO] input_dir  : {input_dir}")
    print(f"[INFO] output_dir : {output_dir}")
    print(f"[INFO] tile_size  : {tile_size}")
    print(f"[INFO] overlap    : {overlap}")
    print(f"[INFO] min_size   : {min_size}")
    print(f"[INFO] 画像数     : {len(images)}")

    total_patches = 0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 読み込み失敗: {img_path}")
            continue

        H, W = img.shape[:2]
        tiles = generate_tiles(H, W, tile=tile_size, overlap=overlap)
        stem = img_path.stem

        patch_count = 0
        for (x1, y1, x2, y2) in tiles:
            patch = img[y1:y2, x1:x2]
            ph, pw = patch.shape[:2]

            # 端の小さすぎるパッチはスキップ
            if ph < min_size or pw < min_size:
                continue

            out_name = f"{stem}_y{y1:04d}_x{x1:04d}.png"
            cv2.imwrite(str(output_dir / out_name), patch)
            patch_count += 1

        total_patches += patch_count
        print(f"[OK] {img_path.name}: {W}x{H} -> {patch_count} patches")

    print(f"[DONE] 合計 {total_patches} パッチを {output_dir} に保存しました")


if __name__ == "__main__":
    main()
