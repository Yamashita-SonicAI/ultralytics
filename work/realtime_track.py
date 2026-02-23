# -- coding: utf-8 --
import time
import cv2
import numpy as np
from ultralytics import YOLOE

# ---- ここにあなたの cs_camera クラス定義を置く（同ファイル内でOK） ----
from MvImport.camera_grab_image import cs_camera

def draw_overlay(img_bgr, text_lines, org=(10, 25), line_h=26):
    x, y = org
    for i, t in enumerate(text_lines):
        yy = y + i * line_h
        cv2.putText(img_bgr, t, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img_bgr, t, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

def overlay_mask(bgr, mask01, alpha=0.45):
    """
    mask01: (H,W) float/bool, 1=mask
    色指定は避けたいので、マスク領域を「明るく」して視認性を上げる方式
    """
    if mask01.dtype != np.float32:
        mask01 = mask01.astype(np.float32)
    mask01 = np.clip(mask01, 0.0, 1.0)

    # 明度を上げる（白方向にブレンド）
    white = np.full_like(bgr, 255)
    a = (mask01 * alpha)[..., None]
    out = (bgr * (1.0 - a) + white * a).astype(np.uint8)

    # 輪郭線（境界を少し強調）
    m = (mask01 > 0.5).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(out, cnts, -1, (0, 255, 0), 2)
    return out

def main():
    # -----------------------------
    # YOLOE 設定（GPU利用を明示）
    # -----------------------------
    device = 0
    half = False
    conf_th = 0.01          # デバッグ用：極力低く
    mask_th = 0.2           # マスク二値化用
    imgsz = 640             # 必要なら固定（安定＆高速化しやすい）

    model = YOLOE("yoloe-26x-seg.pt")

    # --- プロンプトパターン: 1つずつ試して検出されるか確認 ---
    # パターン1: 視覚的特徴を強調
    model.set_classes(["shiny object", "metallic object", "silver object", "reflective object"])
    # パターン2: 素材に着目
    # model.set_classes(["tin foil", "silver foil", "aluminum foil", "metal wrapper"])
    # パターン3: 包装・食品寄り
    # model.set_classes(["food wrapper", "snack", "candy bar", "chocolate bar"])
    # パターン4: 超汎用
    # model.set_classes(["thing", "stuff", "item", "object"])

    # model.set_classes(["hand"])

    # -----------------------------
    # ByteTrack 追跡設定
    # -----------------------------
    tracker = "bytetrack.yaml"
    # tracker="botsort.yaml"
    # persist=True でIDが維持される（追跡継続）
    persist = True

    # -----------------------------
    # カメラ初期化
    # -----------------------------
    cam = cs_camera()
    cam.open_cam()
    payload = cam.set_param(cam.WIDTH, cam.HEIGHT, exposure_time=4000, fps=30)
    cam.start_capture()

    win = "YOLOE26 Hand Instance-Seg + TrackID (q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    prev_t = time.perf_counter()
    ema_fps = None

    try:
        while True:
            frame = cam.grab_frame(payload)
            if frame is None:
                continue

            # ultralyticsはRGB入力が無難
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # -----------------------------
            # 推論 + 追跡（ByteTrack）
            # -----------------------------
            t0 = time.perf_counter()
            try:
                # track は内部で検出→追跡を実行し、boxes.id に track_id が付く
                r = model.track(
                    source=frame_rgb,
                    device=device,
                    half=half,
                    conf=conf_th,
                    iou=0.5,
                    imgsz=imgsz,
                    tracker=tracker,
                    persist=persist,
                    verbose=False
                )[0]
            except Exception as e:
                print("[ERROR] track failed:", repr(e))
                import traceback; traceback.print_exc()
                continue
            infer_ms = (time.perf_counter() - t0) * 1000.0

            disp = frame.copy()

            # -----------------------------
            # デバッグ：推論結果の詳細を出力
            # -----------------------------
            n_boxes = len(r.boxes) if r.boxes is not None else 0
            n_masks = len(r.masks) if r.masks is not None else 0
            cls_ids = r.boxes.cls.cpu().numpy().tolist() if n_boxes > 0 else []
            confs_dbg = r.boxes.conf.cpu().numpy().tolist() if n_boxes > 0 else []
            names = r.names if hasattr(r, 'names') else {}
            cls_names = [names.get(int(c), str(int(c))) for c in cls_ids]
            print(f"[DEBUG] boxes={n_boxes} masks={n_masks} cls={cls_names} conf={[f'{c:.3f}' for c in confs_dbg]}")

            # -----------------------------
            # インスタンスセグ：各インスタンスのマスクを重ねる
            # + TrackID表示（boxes.id）
            # -----------------------------
            n_inst = 0
            best_conf = 0.0

            if r.boxes is not None and len(r.boxes) > 0:
                confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else None
                boxes = r.boxes.xyxy.cpu().numpy().astype(int) if r.boxes.xyxy is not None else None
                tids  = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else None

                # masks: [N, Hm, Wm]（model入力サイズ基準）
                masks = None
                if r.masks is not None and r.masks.data is not None:
                    masks = r.masks.data.cpu().numpy()

                H, W = disp.shape[:2]

                # N は boxes を基準（masks が無い場合も bbox+IDだけ出す）
                N = len(boxes) if boxes is not None else 0
                for i in range(N):
                    c = float(confs[i]) if confs is not None else 0.0
                    if c < conf_th:
                        continue

                    n_inst += 1
                    best_conf = max(best_conf, c)

                    # ---- mask overlay（あれば）----
                    if masks is not None and i < len(masks):
                        m = masks[i]
                        m = (m > mask_th).astype(np.uint8) * 255
                        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                        m01 = (m > 0).astype(np.float32)
                        disp = overlay_mask(disp, m01, alpha=0.35)
                        print(masks[i].min(), masks[i].max())

                    # ---- bbox + conf + track_id ----
                    x1, y1, x2, y2 = boxes[i]
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    tid = int(tids[i]) if tids is not None and i < len(tids) else -1
                    label = f"hand id={tid} {c:.2f}" if tid >= 0 else f"hand {c:.2f}"
                    cv2.putText(
                        disp, label,
                        (x1, max(0, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
                    )
            else:
                print("mask is None")

            # -----------------------------
            # 推論時間 / FPS overlay
            # -----------------------------
            now = time.perf_counter()
            inst_fps = 1.0 / max(1e-6, (now - prev_t))
            prev_t = now
            ema_fps = inst_fps if ema_fps is None else (0.9 * ema_fps + 0.1 * inst_fps)

            lines = [
                f"infer: {infer_ms:.1f} ms  imgsz={imgsz}",
                f"FPS  : {ema_fps:.1f}",
                f"inst : {n_inst} (best={best_conf:.2f})",
                f"GPU  : {device}  fp16={half}  conf>={conf_th}",
                f"TRK  : {tracker}  persist={persist}",
            ]
            draw_overlay(disp, lines)

            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        cam.stop_capture()
        cam.close_cam()
        cam.destroy_handle()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()