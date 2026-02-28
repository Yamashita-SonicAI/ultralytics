import os
import time
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLOE

# ----------------------------
# settings
# ----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
INPUT_DIR = "/home/sonicai/workspace/yamada-develop/yolo-dev/AnomalyDetection/TestSet"          # ここを画像ディレクトリに
OUT_DIR = "/home/sonicai/workspace/yamada-develop/yolo-dev/AnomalyDetection/Result"          # 出力先
CONF_TH = 0.01                    # 必要なら
MASK_TH = 0.4

# class list (YOLOE text prompt)
CLASSES = [
    "hair",
    "strand of hair",
    "thin fiber",
    "foreign object",
    "contamination",
]

# 出力形式：
#   semantic_{stem}.png : 0=bg, 1..K=クラスID(=boxes.cls+1)
#   overlay_{stem}.png  : 入力画像に半透明で重ねた可視化
SAVE_OVERLAY = True

os.makedirs(OUT_DIR, exist_ok=True)

model = YOLOE("yoloe-26x-seg.pt")
model.set_classes(CLASSES)

def iter_images(dir_path: str):
    p = Path(dir_path)
    for f in sorted(p.rglob("*")):
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            yield f

def to_uint16_png(semantic: np.ndarray, out_path: str):
    """
    ラベル数が増えても安全なようにuint16で保存（0..65535）
    """
    cv2.imwrite(out_path, semantic.astype(np.uint16))

def make_overlay(bgr: np.ndarray, semantic: np.ndarray):
    """
    色指定なし要望だったので、クラスIDから擬似カラーを自動生成
    """
    h, w = semantic.shape
    overlay = bgr.copy()

    # ID -> 色 を deterministic に生成（HSVを回す）
    max_id = int(semantic.max())
    if max_id <= 0:
        return overlay

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    # hue: class_idごとにずらす（0..179）
    # sat/value: 固定
    hue = (semantic * 37) % 180
    hsv[..., 0] = hue.astype(np.uint8)
    hsv[..., 1] = np.where(semantic > 0, 200, 0).astype(np.uint8)
    hsv[..., 2] = np.where(semantic > 0, 255, 0).astype(np.uint8)

    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    alpha = np.where(semantic > 0, 0.45, 0.0).astype(np.float32)[..., None]
    overlay = (overlay * (1 - alpha) + color * alpha).astype(np.uint8)
    return overlay

# ----------------------------
# inference loop
# ----------------------------
num_images = 0
total_infer_sec = 0.0
num_model_timed = 0
total_model_infer_ms = 0.0

for img_path in iter_images(INPUT_DIR):
    # 1枚ずつ推論（大量画像ならstream/batchにしてもOK）
    t0 = time.perf_counter()
    r = model.predict(
        source=str(img_path),
        conf=CONF_TH,
        verbose=False,
        device=0
    )[0]
    infer_sec = time.perf_counter() - t0
    model_infer_ms = None
    if hasattr(r, "speed") and isinstance(r.speed, dict) and r.speed.get("inference") is not None:
        model_infer_ms = float(r.speed["inference"])
        num_model_timed += 1
        total_model_infer_ms += model_infer_ms
    num_images += 1
    total_infer_sec += infer_sec

    # マスクが無い場合に備える
    if r.masks is None or r.boxes is None or len(r.boxes) == 0:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[WARN] failed to read: {img_path}")
            continue
        semantic = np.zeros((bgr.shape[0], bgr.shape[1]), dtype=np.int32)
        print(f"\n[{img_path.name}] 検出なし")
    else:
        masks = r.masks.data.cpu().numpy()  # [N, H, W] (model input size)
        cls = r.boxes.cls.cpu().numpy().astype(int)  # [N]
        confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") and r.boxes.conf is not None else None
        xyxy = r.boxes.xyxy.cpu().numpy()  # [N, 4] bbox座標

        # 信頼度でフィルタ
        if confs is not None:
            keep = confs >= CONF_TH
            masks = masks[keep]
            cls = cls[keep]
            confs = confs[keep]
            xyxy = xyxy[keep]

        # 検出結果を出力
        print(f"\n[{img_path.name}] 検出数: {len(cls)}")
        for i in range(len(cls)):
            class_name = CLASSES[cls[i]] if cls[i] < len(CLASSES) else f"cls_{cls[i]}"
            c = confs[i] if confs is not None else -1
            x1, y1, x2, y2 = xyxy[i]
            area = (x2 - x1) * (y2 - y1)
            mask_pixels = int((masks[i] > MASK_TH).sum())
            print(f"  [{i}] class={class_name}  conf={c:.4f}  "
                  f"bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})  "
                  f"bbox_area={area:.0f}  mask_px={mask_pixels}")

        H, W = masks.shape[1], masks.shape[2]
        semantic = np.zeros((H, W), dtype=np.int32)  # 0=background

        # 後勝ちで上書き（必要なら面積/スコア順に並べる）
        for i in range(len(masks)):
            class_id = cls[i] + 1
            semantic[masks[i] > MASK_TH] = class_id

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[WARN] failed to read: {img_path}")
            continue
        semantic = cv2.resize(
            semantic.astype(np.uint16),
            (bgr.shape[1], bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)

    stem = img_path.stem
    out_sem = os.path.join(OUT_DIR, f"semantic_{stem}.png")
    to_uint16_png(semantic, out_sem)

    if SAVE_OVERLAY:
        overlay = make_overlay(bgr, semantic)
        out_ov = os.path.join(OUT_DIR, f"overlay_{stem}.png")
        cv2.imwrite(out_ov, overlay)

    time_info = f"wall={infer_sec * 1000:.2f} ms"
    if model_infer_ms is not None:
        time_info += f", model_infer={model_infer_ms:.2f} ms"
    print(f"  -> {out_sem} | {time_info}")

if num_images > 0:
    print(f"[SUMMARY] images={num_images}, avg_wall={total_infer_sec / num_images * 1000:.2f} ms")
    if num_model_timed > 0:
        print(f"[SUMMARY] images_with_speed={num_model_timed}, avg_model_infer={total_model_infer_ms / num_model_timed:.2f} ms")
