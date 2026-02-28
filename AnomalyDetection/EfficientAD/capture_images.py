#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capture_images.py
カメラから画像を連続キャプチャして保存するスクリプト。

操作:
  's' : 現在のフレームを保存（30枚まで）
  'q' or ESC : 終了

保存先:
  EfficientAD/captured/<タイムスタンプ>/000.png, 001.png, ...
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# work/ 配下の MvImport を import できるようにパスを追加
WORK_DIR = Path(__file__).resolve().parent.parent.parent / "work"
sys.path.insert(0, str(WORK_DIR))

from MvImport.camera_grab_image import cs_camera

# ========== 設定 ==========
SAVE_BASE = Path(__file__).resolve().parent / "captured"
MAX_CAPTURES = 30
EXPOSURE_TIME = 4000  # マイクロ秒
FPS = 30


def main():
    # 保存先ディレクトリ（タイムスタンプ付き）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = SAVE_BASE / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    # カメラ初期化
    cam = cs_camera()
    cam.open_cam()
    payload = cam.set_param(cam.WIDTH, cam.HEIGHT, exposure_time=EXPOSURE_TIME, fps=FPS)
    cam.start_capture()

    win = "Capture (press 's' to save, 'q' to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    count = 0
    print(f"[INFO] 保存先: {save_dir}")
    print(f"[INFO] 's' で撮影・保存（最大 {MAX_CAPTURES} 枚）, 'q' で終了")

    try:
        while True:
            frame = cam.grab_frame(payload)
            if frame is None:
                continue

            # grab_frameはBGRで返す（rotate_image.pyと同様）
            frame_bgr = frame

            # 情報をオーバーレイ表示
            disp = frame_bgr.copy()
            cv2.putText(disp, f"Saved: {count}/{MAX_CAPTURES}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(disp, "'s': capture  'q': quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(win, disp)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                if count >= MAX_CAPTURES:
                    print(f"[INFO] {MAX_CAPTURES} 枚に達しました。終了します。")
                    break

                filename = f"{count:03d}.png"
                filepath = save_dir / filename
                cv2.imwrite(str(filepath), frame_bgr)
                count += 1
                print(f"[OK] {filepath} ({count}/{MAX_CAPTURES})")

                if count >= MAX_CAPTURES:
                    print(f"[INFO] {MAX_CAPTURES} 枚保存完了。終了します。")
                    break

            elif key == ord("q") or key == 27:
                print(f"[INFO] 終了。{count} 枚保存しました。")
                break

    finally:
        cam.stop_capture()
        cam.close_cam()
        cam.destroy_handle()
        cv2.destroyAllWindows()

    print(f"[DONE] {save_dir} に {count} 枚保存")


if __name__ == "__main__":
    main()
