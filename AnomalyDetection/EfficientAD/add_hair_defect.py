#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_hair_defect.py
MVTec bottle の good 画像に髪の毛風の黒い曲線を合成し、
test/hair_contamination/ に保存するスクリプト。

使い方:
  python add_hair_defect.py

パラメータはスクリプト上部の定数で調整可能。
"""

import random
from pathlib import Path

import cv2
import numpy as np

# ========== 設定 ==========
MVTEC_BOTTLE_DIR = Path(
    "/home/sonicai/workspace/yamada-develop/yolo-dev/AnomalyDetection/EfficientAD/datasets/MVTecAD/bottle"
)
SRC_DIR = MVTEC_BOTTLE_DIR / "test" / "good"
DST_DIR = MVTEC_BOTTLE_DIR / "test" / "hair_contamination"

SEED = 42

# 髪の毛パラメータ
NUM_HAIRS_RANGE = (1, 3)       # 1画像あたりの本数（min, max）
NUM_POINTS_RANGE = (5, 10)     # 制御点の数（多いほどくねくね）
HAIR_LENGTH_RANGE = (80, 250)  # 髪の毛の長さ（ピクセル）
THICKNESS_RANGE = (1, 3)       # 線の太さ（3px固定）
COLOR_RANGE = (0, 40)          # BGR各チャンネルの値範囲（0=真っ黒、40=やや暗い灰色）


def generate_hair_points(cx, cy, length, num_points):
    """
    始点 (cx, cy) から、ランダムに曲がりながら伸びる制御点列を生成する。
    - 全体の進行方向をランダムに決定
    - 各ステップで進行角度に小さなランダム偏差を加え、くねくねさせる
    """
    angle = random.uniform(0, 2 * np.pi)  # 全体の進行方向
    step = length / num_points

    points = [(cx, cy)]
    for _ in range(num_points):
        # 角度にランダムな揺れを加える（大きいほどくねくね）
        angle += random.gauss(0, 0.5)
        dx = step * np.cos(angle)
        dy = step * np.sin(angle)
        cx += dx
        cy += dy
        points.append((int(cx), int(cy)))

    return np.array(points, dtype=np.int32)


def draw_hair(img, hair_points, thickness, color):
    """
    制御点列を滑らかな曲線として描画する。
    cv2.polylines で折れ線を描画し、isClosed=False で開いた線にする。
    """
    # polylines は点列のリストを受け取る
    cv2.polylines(img, [hair_points], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def add_hairs_to_image(img):
    """
    画像に髪の毛風の曲線を追加する。
    髪の毛の始点は画像中央付近（瓶の領域内）にランダム配置。
    """
    h, w = img.shape[:2]
    out = img.copy()

    num_hairs = random.randint(*NUM_HAIRS_RANGE)

    for _ in range(num_hairs):
        # 始点を画像中央付近に配置（端に出すぎないように）
        margin = int(min(h, w) * 0.2)
        cx = random.randint(margin, w - margin)
        cy = random.randint(margin, h - margin)

        length = random.randint(*HAIR_LENGTH_RANGE)
        num_points = random.randint(*NUM_POINTS_RANGE)
        thickness = random.randint(*THICKNESS_RANGE)

        # 暗い色（黒〜暗い灰色）
        c = random.randint(*COLOR_RANGE)
        color = (c, c, c)

        hair_points = generate_hair_points(cx, cy, length, num_points)
        draw_hair(out, hair_points, thickness, color)

    return out


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    DST_DIR.mkdir(parents=True, exist_ok=True)

    src_images = sorted(SRC_DIR.glob("*.png"))
    if not src_images:
        print(f"[ERROR] good 画像が見つかりません: {SRC_DIR}")
        return

    print(f"[INFO] 入力: {SRC_DIR} ({len(src_images)} 枚)")
    print(f"[INFO] 出力: {DST_DIR}")
    print(f"[INFO] 髪の毛: {NUM_HAIRS_RANGE[0]}~{NUM_HAIRS_RANGE[1]}本/枚, "
          f"長さ{HAIR_LENGTH_RANGE[0]}~{HAIR_LENGTH_RANGE[1]}px, "
          f"太さ{THICKNESS_RANGE[0]}~{THICKNESS_RANGE[1]}px")

    for src_path in src_images:
        img = cv2.imread(str(src_path))
        if img is None:
            print(f"[WARN] 読み込み失敗: {src_path}")
            continue

        out = add_hairs_to_image(img)

        dst_path = DST_DIR / src_path.name
        cv2.imwrite(str(dst_path), out)
        print(f"[OK] {src_path.name} -> {dst_path}")

    print(f"[DONE] {len(src_images)} 枚を {DST_DIR} に保存しました")


if __name__ == "__main__":
    main()
