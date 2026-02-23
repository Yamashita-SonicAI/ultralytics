# -- coding: utf-8 --
"""
YOLOE Visual Prompt (VP) で「参照ROIと同じ物体」を推論（追跡なし）
+ 参照に使った"キャプチャ画像"と"ROI切り抜き画像"を保存

推論方式:
  VPEを1回だけセットし、以降は refer_image なしで高速推論

操作:
- 's' : 現在フレームをキャプチャ → 参照画像保存 → ROI選択 → ROI切り抜き保存 → VPEセット
- 'r' : 参照ROIを再選択（VPEリセット）
- 'm' : マスクオーバーレイの輪郭描画あり/なしを切り替え
- 'q' or ESC : 終了

保存先:
- SAVE_DIR を任意のパスに変更してください
"""

import os
import shutil
import time
import cv2
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

from MvImport.camera_grab_image import cs_camera


# =============================
# 保存先（ここを任意に変更）
# =============================
SAVE_DIR = "/home/sonicai/workspace/yamada-develop/yolo-dev/work/data"

# =============================
# 起動時に保存ディレクトリをクリアするか
#   True  : 既存ディレクトリを削除して作り直す
#   False : 既存ディレクトリはそのまま保持
# =============================
CLEAR_SAVE_DIR = False


def ensure_dir(path: str, clear: bool = False):
    """ディレクトリを作成する。clear=Trueなら既存を削除してから作り直す。"""
    if clear and os.path.exists(path):
        shutil.rmtree(path)
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


def overlay_mask_whiten(bgr, mask01, alpha=0.35, draw_contour=True):
    """マスクオーバーレイ。draw_contour=Trueで輪郭線も描画する。"""
    if mask01.dtype != np.float32:
        mask01 = mask01.astype(np.float32)
    mask01 = np.clip(mask01, 0.0, 1.0)
    white = np.full_like(bgr, 255)
    a = (mask01 * alpha)[..., None]
    out = (bgr * (1.0 - a) + white * a).astype(np.uint8)

    if draw_contour:
        m = (mask01 > 0.5).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(out, cnts, -1, (0, 255, 0), 2)
    return out


def save_reference_images(frame_bgr, bbox_xyxy, save_dir):
    ensure_dir(save_dir)
    tag = ts_str()

    capture_path = os.path.join(save_dir, f"capture_{tag}.png")
    cv2.imwrite(capture_path, frame_bgr)

    x1, y1, x2, y2 = map(int, bbox_xyxy)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = max(0, x2); y2 = max(0, y2)
    roi_bgr = frame_bgr[y1:y2, x1:x2].copy()

    roi_path = os.path.join(save_dir, f"roi_{tag}_x1{x1}_y1{y1}_x2{x2}_y2{y2}.png")
    if roi_bgr.size > 0:
        cv2.imwrite(roi_path, roi_bgr)
    else:
        roi_path = None

    return capture_path, roi_path


def select_roi_from_current_frame(win, frame_bgr):
    draw_overlay(frame_bgr, ["Select ROI then Enter/Space  (ESC/c to cancel)"], org=(10, 25))
    roi = cv2.selectROI(win, frame_bgr, fromCenter=False, showCrosshair=True)
    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        return None
    return np.array([x, y, x + w, y + h], dtype=np.int32)


def main():
    device = 0
    half = False            # GPUならTrue推奨
    conf_th = 0.05
    mask_th = 0.15
    imgsz = 640

    # マスクオーバーレイの輪郭描画（'m'キーで切り替え可能）
    draw_contour = False

    model = YOLOE("yoloe-26x-seg.pt")

    cam = cs_camera()
    cam.open_cam()
    payload = cam.set_param(cam.WIDTH, cam.HEIGHT, exposure_time=4000, fps=30)
    cam.start_capture()

    win = "YOLOE VP Seg (VPE set once, then fast) - no tracking"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    ensure_dir(SAVE_DIR, clear=CLEAR_SAVE_DIR)
    print(f"[INFO] SAVE_DIR = {SAVE_DIR}")

    vp_ready = False
    refer_rgb = None
    visual_prompts = None

    prev_t = time.perf_counter()
    ema_fps = None

    try:
        while True:
            frame = cam.grab_frame(payload)
            if frame is None:
                continue

            key = cv2.waitKey(1) & 0xFF

            # -----------------------------
            # VP未セット時：sで参照ROIを取り、VPEを1回だけセット
            # -----------------------------
            if not vp_ready:
                disp = frame.copy()
                draw_overlay(disp, [
                    "'s' : capture -> save -> select ROI -> set VPE (one-time)",
                    "'q' : quit",
                    f"SAVE_DIR: {SAVE_DIR}",
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

                    # 参照画像（RGB）とVP情報
                    refer_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    bbox = bbox_xyxy.astype(np.float32)[None, :]  # (1,4)
                    visual_prompts = dict(bboxes=bbox, cls=np.array([0], dtype=int))

                    # VPEを「一回だけ」セットする初期化predict
                    print("[INFO] Setting VPE (one-time)...")
                    try:
                        _ = model.predict(
                            source=refer_rgb,
                            refer_image=refer_rgb,
                            visual_prompts=visual_prompts,
                            predictor=YOLOEVPSegPredictor,
                            device=device,
                            half=half,
                            conf=conf_th,
                            imgsz=imgsz,
                            verbose=False,
                        )
                        vp_ready = True
                        print("[INFO] VPE set done. Now running fast inference WITHOUT refer_image per frame.")
                    except Exception as e:
                        print("[ERROR] VPE setup failed:", repr(e))
                        import traceback; traceback.print_exc()
                        vp_ready = False

                continue

            # -----------------------------
            # 高速推論（refer_image/visual_prompts を渡さない）
            # -----------------------------
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            t0 = time.perf_counter()
            try:
                r = model.predict(
                    source=frame_rgb,
                    device=device,
                    half=half,
                    conf=conf_th,
                    imgsz=imgsz,
                    verbose=False,
                )[0]
            except Exception as e:
                print("[ERROR] predict failed:", repr(e))
                import traceback; traceback.print_exc()
                vp_ready = False
                refer_rgb = None
                visual_prompts = None
                continue
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

                for i in range(len(boxes)):
                    c = float(confs[i]) if confs is not None else 0.0
                    if c < conf_th:
                        continue
                    n_inst += 1
                    best_conf = max(best_conf, c)

                    if masks is not None and i < len(masks):
                        m = masks[i]
                        m_bin = (m > mask_th).astype(np.uint8) * 255
                        m_bin = cv2.resize(m_bin, (W, H), interpolation=cv2.INTER_NEAREST)
                        m01 = (m_bin > 0).astype(np.float32)
                        if m01.sum() > 0:
                            disp = overlay_mask_whiten(disp, m01, alpha=0.35, draw_contour=draw_contour)

                    x1, y1, x2, y2 = boxes[i]
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        disp, f"{c:.2f}",
                        (x1, max(0, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
                    )

            now = time.perf_counter()
            inst_fps = 1.0 / max(1e-6, (now - prev_t))
            prev_t = now
            ema_fps = inst_fps if ema_fps is None else (0.9 * ema_fps + 0.1 * inst_fps)

            contour_label = "ON" if draw_contour else "OFF"
            draw_overlay(disp, [
                f"infer: {infer_ms:.1f} ms  imgsz={imgsz}",
                f"FPS  : {ema_fps:.1f}",
                f"inst : {n_inst} (best={best_conf:.2f})",
                f"masks: {'OK' if has_masks else 'None'}  mask_th={mask_th}",
                f"conf : {conf_th}  fp16={half}  device={device}",
                f"contour: {contour_label}",
                "'r' : re-select ROI   'm' : toggle contour   'q' : quit",
            ])
            cv2.imshow(win, disp)

            if key == ord("q") or key == 27:
                break
            elif key == ord("r"):
                vp_ready = False
                refer_rgb = None
                visual_prompts = None
                print("[INFO] VP reset. Press 's' to set VPE again.")
            elif key == ord("m"):
                draw_contour = not draw_contour
                print(f"[INFO] contour drawing: {contour_label}")

    finally:
        cam.stop_capture()
        cam.close_cam()
        cam.destroy_handle()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
