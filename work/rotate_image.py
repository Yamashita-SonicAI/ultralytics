# -- coding: utf-8 --
"""
YOLOE Visual Prompt (VP) セグメンテーション
- 参照ROIを保存（キャプチャ全体 + ROI切り抜き）
- 参照ROIから 0/90/180/270 回転した4枚を作り、2x2モザイク参照画像にまとめる
- そのモザイク参照の複数bboxを visual_prompts として使い、VPEを「初回だけ」セット（高速）
- 以降の推論は refer_image を渡さずに model.predict(source=frame) のみ（高速）

操作:
- 's' : 現在フレームを保存 → ROI選択 → rot4モザイク参照を作ってVPEセット
- 'r' : VPをリセット（次に 's' で再セット）
- 'q' or ESC : 終了

注意:
- YOLOEは "yoloe-26x-seg.pt" が重いので、GPU + half=True 推奨
"""

import os
import time
import cv2
import numpy as np

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

from MvImport.camera_grab_image import cs_camera


# =============================
# 保存先（任意に変更）
# =============================
SAVE_DIR = "/home/sonicai/workspace/yamada-develop/yolo-dev/work/data"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def ts_str():
    t = time.time()
    lt = time.localtime(t)
    ms = int((t - int(t)) * 1000)
    return time.strftime("%Y%m%d_%H%M%S", lt) + f"_{ms:03d}"


def draw_overlay(img_bgr, text_lines, org=(10, 25), line_h=26):
    x, y = org
    for i, t in enumerate(text_lines):
        yy = y + i * line_h
        cv2.putText(img_bgr, t, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img_bgr, t, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def overlay_mask_whiten_fast(bgr, mask01, alpha=0.35):
    """
    マスク領域を白方向にブレンド（高速版）
    - findContours は重いのでやらない
    """
    if mask01.dtype != np.float32:
        mask01 = mask01.astype(np.float32)
    mask01 = np.clip(mask01, 0.0, 1.0)

    white = np.full_like(bgr, 255)
    a = (mask01 * alpha)[..., None]
    return (bgr * (1.0 - a) + white * a).astype(np.uint8)


def save_reference_images(frame_bgr, bbox_xyxy, save_dir):
    """
    キャプチャ画像全体と ROI切り抜きを保存する
    戻り値: (capture_path, roi_path)
    """
    ensure_dir(save_dir)
    tag = ts_str()

    capture_path = os.path.join(save_dir, f"capture_{tag}.png")
    cv2.imwrite(capture_path, frame_bgr)

    x1, y1, x2, y2 = map(int, bbox_xyxy)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = max(0, x2)
    y2 = max(0, y2)

    roi_bgr = frame_bgr[y1:y2, x1:x2].copy()

    roi_path = os.path.join(save_dir, f"roi_{tag}_x1{x1}_y1{y1}_x2{x2}_y2{y2}.png")
    if roi_bgr.size > 0:
        cv2.imwrite(roi_path, roi_bgr)
    else:
        roi_path = None

    return capture_path, roi_path


def select_roi_from_current_frame(win, frame_bgr):
    """
    現在フレームからROI選択して xyxy を返す
    """
    draw_overlay(frame_bgr, ["Select ROI then Enter/Space  (ESC/c to cancel)"], org=(10, 25))
    roi = cv2.selectROI(win, frame_bgr, fromCenter=False, showCrosshair=True)
    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        return None
    return np.array([x, y, x + w, y + h], dtype=np.int32)


def build_rot4_mosaic_refer_from_roi(frame_bgr, bbox_xyxy, pad=0, tile_size=256, gap=8):
    """
    ROIを切り抜き、0/90/180/270回転の4枚を作って2x2モザイクにする。
    返り値:
      refer_rgb_mosaic: (H,W,3) RGB
      bboxes: (4,4) float32 それぞれのタイルbbox（xyxy）
    """
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, bbox_xyxy)

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad)
    y2 = min(H, y2 + pad)

    roi = frame_bgr[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return None, None

    # 回転
    r0 = roi
    r90 = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
    r180 = cv2.rotate(roi, cv2.ROTATE_180)
    r270 = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # サイズ統一（重要）
    def fit(im):
        return cv2.resize(im, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)

    tiles = [fit(r0), fit(r90), fit(r180), fit(r270)]

    # モザイク作成
    canvas_h = tile_size * 2 + gap * 3
    canvas_w = tile_size * 2 + gap * 3
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    xA = gap
    xB = gap * 2 + tile_size
    yA = gap
    yB = gap * 2 + tile_size

    canvas[yA:yA + tile_size, xA:xA + tile_size] = tiles[0]
    canvas[yA:yA + tile_size, xB:xB + tile_size] = tiles[1]
    canvas[yB:yB + tile_size, xA:xA + tile_size] = tiles[2]
    canvas[yB:yB + tile_size, xB:xB + tile_size] = tiles[3]

    bboxes = np.array([
        [xA, yA, xA + tile_size, yA + tile_size],
        [xB, yA, xB + tile_size, yA + tile_size],
        [xA, yB, xA + tile_size, yB + tile_size],
        [xB, yB, xB + tile_size, yB + tile_size],
    ], dtype=np.float32)

    refer_rgb_mosaic = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return refer_rgb_mosaic, bboxes


def main():
    # -----------------------------
    # YOLOE 設定
    # -----------------------------
    device = 0
    half = False  # GPUなら True 推奨（遅い場合はまずここ）
    conf_th = 0.05
    mask_th = 0.15
    imgsz = 640

    # rot4 参照の作り方パラメータ
    rot_pad = 0         # まず 0（tight）。必要なら 6〜12へ
    rot_tile = 256      # 256推奨（参照が大きいほど背景に引っ張られやすい）
    rot_gap = 8

    model = YOLOE("yoloe-26x-seg.pt")

    # -----------------------------
    # カメラ初期化
    # -----------------------------
    cam = cs_camera()
    cam.open_cam()
    payload = cam.set_param(cam.WIDTH, cam.HEIGHT, exposure_time=4000, fps=30)
    cam.start_capture()

    win = "YOLOE VP Seg (rot4 mosaic VPE set-once, then fast)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    ensure_dir(SAVE_DIR)
    print(f"[INFO] SAVE_DIR = {SAVE_DIR}")

    vp_ready = False
    prev_t = time.perf_counter()
    ema_fps = None

    # VPEセット時に作った参照（必要なら保存/デバッグに使える）
    last_refer_mosaic = None

    try:
        while True:
            frame = cam.grab_frame(payload)
            if frame is None:
                continue

            key = cv2.waitKey(1) & 0xFF

            # -----------------------------
            # VP未セット： 's' でセット
            # -----------------------------
            if not vp_ready:
                disp = frame.copy()
                draw_overlay(disp, [
                    "'s' : capture -> save -> select ROI -> build rot4 mosaic -> set VPE (one-time)",
                    "'q' : quit",
                    f"SAVE_DIR: {SAVE_DIR}",
                    f"rot_pad={rot_pad}  tile={rot_tile}  gap={rot_gap}",
                ])
                cv2.imshow(win, disp)

                if key == ord("q") or key == 27:
                    break

                if key == ord("s"):
                    bbox_xyxy = select_roi_from_current_frame(win, frame.copy())
                    if bbox_xyxy is None:
                        print("[WARN] ROI selection canceled")
                        continue

                    cap_path, roi_path = save_reference_images(frame, bbox_xyxy, SAVE_DIR)
                    print(f"[INFO] saved capture: {cap_path}")
                    if roi_path:
                        print(f"[INFO] saved roi    : {roi_path}")

                    refer_rgb_mosaic, vp_bboxes = build_rot4_mosaic_refer_from_roi(
                        frame_bgr=frame,
                        bbox_xyxy=bbox_xyxy,
                        pad=rot_pad,
                        tile_size=rot_tile,
                        gap=rot_gap,
                    )
                    if refer_rgb_mosaic is None:
                        print("[WARN] failed to build rot4 mosaic refer_image")
                        continue

                    # モザイク参照画像を保存（RGB→BGR変換して保存）
                    mosaic_path = os.path.join(SAVE_DIR, f"refer_mosaic_{ts_str()}.png")
                    cv2.imwrite(mosaic_path, cv2.cvtColor(refer_rgb_mosaic, cv2.COLOR_RGB2BGR))
                    print(f"[INFO] saved mosaic : {mosaic_path}")

                    visual_prompts = dict(
                        bboxes=vp_bboxes,
                        cls=np.zeros((len(vp_bboxes),), dtype=int),  # 同一クラスにまとめる
                    )

                    print("[INFO] Setting VPE with rot4 mosaic (one-time)...")
                    _ = model.predict(
                        source=refer_rgb_mosaic,
                        refer_image=refer_rgb_mosaic,
                        visual_prompts=visual_prompts,
                        predictor=YOLOEVPSegPredictor,
                        device=device,
                        half=half,
                        conf=conf_th,
                        imgsz=imgsz,
                        verbose=False,
                    )
                    vp_ready = True
                    last_refer_mosaic = refer_rgb_mosaic
                    print("[INFO] VPE set done. Now fast inference WITHOUT refer_image per frame.")

                continue

            # -----------------------------
            # 高速推論（重要：refer_image/visual_prompts を渡さない）
            # -----------------------------
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            t0 = time.perf_counter()
            r = model.predict(
                source=frame_rgb,
                device=device,
                half=half,
                conf=conf_th,
                imgsz=imgsz,
                verbose=False,
            )[0]
            infer_ms = (time.perf_counter() - t0) * 1000.0

            disp = frame.copy()
            H, W = disp.shape[:2]

            has_masks = (r.masks is not None and r.masks.data is not None)
            n_inst = 0
            best_conf = 0.0

            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else None
                masks = r.masks.data.cpu().numpy() if has_masks else None

                # best conf だけ描画（速度優先）
                bi = int(np.argmax(confs)) if confs is not None and len(confs) > 0 else 0
                best_conf = float(confs[bi]) if confs is not None else 0.0

                if best_conf >= conf_th:
                    n_inst = 1

                    if masks is not None and bi < len(masks):
                        m = masks[bi]
                        m_bin = (m > mask_th).astype(np.uint8) * 255
                        m_bin = cv2.resize(m_bin, (W, H), interpolation=cv2.INTER_NEAREST)
                        m01 = (m_bin > 0).astype(np.float32)
                        if m01.sum() > 0:
                            disp = overlay_mask_whiten_fast(disp, m01, alpha=0.35)

                    x1, y1, x2, y2 = boxes[bi]
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        disp, f"{best_conf:.2f}",
                        (x1, max(0, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
                    )

            # FPS
            now = time.perf_counter()
            inst_fps = 1.0 / max(1e-6, (now - prev_t))
            prev_t = now
            ema_fps = inst_fps if ema_fps is None else (0.9 * ema_fps + 0.1 * inst_fps)

            draw_overlay(disp, [
                f"infer: {infer_ms:.1f} ms  imgsz={imgsz}",
                f"FPS  : {ema_fps:.1f}",
                f"inst : {n_inst} (best={best_conf:.2f})",
                f"masks: {'OK' if has_masks else 'None'}  mask_th={mask_th}",
                f"conf : {conf_th}  fp16={half}  device={device}",
                f"ref  : rot4 mosaic (pad={rot_pad}, tile={rot_tile})",
                "'r' : reset VP   'q' : quit",
            ])
            cv2.imshow(win, disp)

            if key == ord("q") or key == 27:
                break
            elif key == ord("r"):
                vp_ready = False
                last_refer_mosaic = None
                print("[INFO] VP reset. Press 's' to set VPE again.")

    finally:
        cam.stop_capture()
        cam.close_cam()
        cam.destroy_handle()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()