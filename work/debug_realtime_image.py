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
- 'd' : デバッグ表示 ON/OFF
- 'p' : デバッグprint ON/OFF
- 'q' or ESC : 終了

保存先:
- SAVE_DIR を任意のパスに変更してください
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


# -----------------------------
# 設定
# -----------------------------
@dataclass
class AppConfig:
    device: int = 0
    half: bool = False          # GPUならTrue推奨
    conf_th: float = 0.05       # model.predict に渡す conf（VP推論に影響する場合あり）
    mask_th: float = 0.15       # 表示用（overlayの二値化）※デバッグ統計とは別
    imgsz: int = 640

    debug_mode: bool = True
    debug_print: bool = True
    debug_topk_draw: int = 20

    save_dir: str = SAVE_DIR
    clear_save_dir: bool = CLEAR_SAVE_DIR

    window_name: str = "YOLOE VP Seg (VPE set once, then fast) - no tracking"


# -----------------------------
# ユーティリティ
# -----------------------------
def ensure_dir(path: str, clear: bool = False) -> None:
    """ディレクトリを作成する。clear=Trueなら既存を削除してから作り直す。"""
    if clear and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def ts_str() -> str:
    t = time.time()
    lt = time.localtime(t)
    ms = int((t - int(t)) * 1000)
    return time.strftime("%Y%m%d_%H%M%S", lt) + f"_{ms:03d}"


def draw_overlay(img_bgr: np.ndarray, text_lines: List[str], org: Tuple[int, int] = (10, 25), line_h: int = 26) -> None:
    x, y = org
    for i, t in enumerate(text_lines):
        yy = y + i * line_h
        cv2.putText(img_bgr, t, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img_bgr, t, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def overlay_mask_whiten(bgr: np.ndarray, mask01: np.ndarray, alpha: float = 0.35, draw_contour: bool = True) -> np.ndarray:
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


def save_reference_images(frame_bgr: np.ndarray, bbox_xyxy: np.ndarray, save_dir: str) -> Tuple[str, Optional[str]]:
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


def select_roi_from_current_frame(win: str, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    draw_overlay(frame_bgr, ["Select ROI then Enter/Space  (ESC/c to cancel)"], org=(10, 25))
    roi = cv2.selectROI(win, frame_bgr, fromCenter=False, showCrosshair=True)
    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        return None
    return np.array([x, y, x + w, y + h], dtype=np.int32)


def summarize_detections(r: Any, bins: int = 10) -> Dict[str, Any]:
    """しきい値なしで、生の検出状況を要約する（confとmask面積率）。"""
    info: Dict[str, Any] = {
        "n_boxes": 0,
        "has_masks": False,
        "conf_min": None,
        "conf_mean": None,
        "conf_med": None,
        "conf_max": None,
        "conf_top": [],
        "conf_hist": [0] * bins,
        "mask_area_min": None,
        "mask_area_mean": None,
        "mask_area_med": None,
        "mask_area_max": None,
        "mask_area_top": [],
    }

    if r is None or r.boxes is None or len(r.boxes) == 0:
        return info

    info["n_boxes"] = int(len(r.boxes))

    # --- conf ---
    confs = r.boxes.conf
    conf_np = None if confs is None else confs.detach().cpu().numpy().astype(np.float32)

    if conf_np is not None and len(conf_np) > 0:
        info["conf_min"] = float(conf_np.min())
        info["conf_mean"] = float(conf_np.mean())
        info["conf_med"] = float(np.median(conf_np))
        info["conf_max"] = float(conf_np.max())

        idx = np.argsort(-conf_np)
        topN = min(10, len(idx))
        info["conf_top"] = [float(conf_np[idx[i]]) for i in range(topN)]

        for c in conf_np:
            b = int(np.clip(c, 0.0, 0.999999) * bins)
            info["conf_hist"][b] += 1

    # --- masks ---
    has_masks = (r.masks is not None and r.masks.data is not None)
    info["has_masks"] = bool(has_masks)
    if has_masks:
        masks = r.masks.data.detach().cpu().numpy()
        areas: List[float] = []
        for m in masks:
            m_bin = (m > 0.5).astype(np.uint8)
            areas.append(float(m_bin.sum()) / float(m_bin.size))

        if areas:
            a = np.array(areas, dtype=np.float32)
            info["mask_area_min"] = float(a.min())
            info["mask_area_mean"] = float(a.mean())
            info["mask_area_med"] = float(np.median(a))
            info["mask_area_max"] = float(a.max())
            idx = np.argsort(-a)
            topN = min(10, len(idx))
            info["mask_area_top"] = [float(a[idx[i]]) for i in range(topN)]

    return info


# -----------------------------
# アプリ本体
# -----------------------------
class VPApp:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

        self.model = YOLOE("yoloe-26x-seg.pt")

        self.cam = cs_camera()
        self.payload = None

        self.vp_ready = False
        self.refer_rgb: Optional[np.ndarray] = None
        self.visual_prompts: Optional[Dict[str, Any]] = None

        self.draw_contour = False

        self.prev_t = time.perf_counter()
        self.ema_fps: Optional[float] = None

    # ---- camera lifecycle ----
    def open_camera(self) -> None:
        self.cam.open_cam()
        self.payload = self.cam.set_param(self.cam.WIDTH, self.cam.HEIGHT, exposure_time=4000, fps=30)
        self.cam.start_capture()

    def close_camera(self) -> None:
        self.cam.stop_capture()
        self.cam.close_cam()
        self.cam.destroy_handle()

    # ---- VP control ----
    def reset_vp(self) -> None:
        self.vp_ready = False
        self.refer_rgb = None
        self.visual_prompts = None
        print("[INFO] VP reset. Press 's' to set VPE again.")

    def setup_vpe_from_frame(self, frame_bgr: np.ndarray) -> bool:
        bbox_xyxy = select_roi_from_current_frame(self.cfg.window_name, frame_bgr.copy())
        if bbox_xyxy is None:
            print("[WARN] ROI selection canceled")
            return False

        cap_path, roi_path = save_reference_images(frame_bgr, bbox_xyxy, self.cfg.save_dir)
        print(f"[INFO] saved capture: {cap_path}")
        if roi_path:
            print(f"[INFO] saved roi    : {roi_path}")

        self.refer_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        bbox = bbox_xyxy.astype(np.float32)[None, :]  # (1,4)
        self.visual_prompts = {"bboxes": bbox, "cls": np.array([0], dtype=int)}

        print("[INFO] Setting VPE (one-time)...")
        try:
            _ = self.model.predict(
                source=self.refer_rgb,
                refer_image=self.refer_rgb,
                visual_prompts=self.visual_prompts,
                predictor=YOLOEVPSegPredictor,
                device=self.cfg.device,
                half=self.cfg.half,
                conf=self.cfg.conf_th,
                imgsz=self.cfg.imgsz,
                verbose=False,
            )
            self.vp_ready = True
            print("[INFO] VPE set done. Now running fast inference WITHOUT refer_image per frame.")
            return True
        except Exception as e:
            print("[ERROR] VPE setup failed:", repr(e))
            import traceback

            traceback.print_exc()
            self.vp_ready = False
            return False

    # ---- inference & render ----
    def predict_fast(self, frame_bgr: np.ndarray) -> Tuple[Optional[Any], float]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        try:
            r = self.model.predict(
                source=frame_rgb,
                device=self.cfg.device,
                half=self.cfg.half,
                conf=self.cfg.conf_th,
                imgsz=self.cfg.imgsz,
                verbose=False,
            )[0]
        except Exception as e:
            print("[ERROR] predict failed:", repr(e))
            import traceback

            traceback.print_exc()
            self.reset_vp()
            return None, 0.0

        infer_ms = (time.perf_counter() - t0) * 1000.0
        return r, infer_ms

    def update_fps(self) -> float:
        now = time.perf_counter()
        inst_fps = 1.0 / max(1e-6, (now - self.prev_t))
        self.prev_t = now
        self.ema_fps = inst_fps if self.ema_fps is None else (0.9 * self.ema_fps + 0.1 * inst_fps)
        return float(self.ema_fps)

    def draw_predictions(self, disp: np.ndarray, r: Any) -> Tuple[int, float, bool]:
        """しきい値なしで top-k だけ描画（デバッグ向け）。"""
        H, W = disp.shape[:2]
        has_masks = (r.masks is not None and r.masks.data is not None)

        n_inst = 0
        best_conf = 0.0

        if r.boxes is None or len(r.boxes) == 0:
            return n_inst, best_conf, has_masks

        boxes = r.boxes.xyxy.detach().cpu().numpy().astype(int)
        confs = r.boxes.conf.detach().cpu().numpy().astype(np.float32) if r.boxes.conf is not None else None
        masks = r.masks.data.detach().cpu().numpy() if has_masks else None

        order = np.argsort(-confs) if confs is not None else np.arange(len(boxes))
        topk = min(self.cfg.debug_topk_draw, len(boxes))

        for rank in range(topk):
            i = int(order[rank])
            c = float(confs[i]) if confs is not None else 0.0
            n_inst += 1
            best_conf = max(best_conf, c)

            if masks is not None and i < len(masks):
                m = masks[i]
                m_bin = (m > 0.5).astype(np.uint8) * 255
                m_bin = cv2.resize(m_bin, (W, H), interpolation=cv2.INTER_NEAREST)
                m01 = (m_bin > 0).astype(np.float32)
                if m01.sum() > 0:
                    disp[:] = overlay_mask_whiten(disp, m01, alpha=0.25, draw_contour=self.draw_contour)

            x1, y1, x2, y2 = boxes[i]
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                disp,
                f"#{rank+1} {c:.3f}",
                (x1, max(0, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return n_inst, best_conf, has_masks

    def build_overlay_lines(self, infer_ms: float, fps: float, n_inst: int, best_conf: float, has_masks: bool, dbg: Dict[str, Any]) -> List[str]:
        contour_label = "ON" if self.draw_contour else "OFF"
        lines = [
            f"infer: {infer_ms:.1f} ms  imgsz={self.cfg.imgsz}",
            f"FPS  : {fps:.1f}",
            f"inst(draw): {n_inst} (best={best_conf:.2f})",
            f"masks: {'OK' if has_masks else 'None'}  mask_th={self.cfg.mask_th}",
            f"conf : {self.cfg.conf_th}  fp16={self.cfg.half}  device={self.cfg.device}",
            f"contour: {contour_label}",
            "'r' : re-select ROI   'm' : toggle contour   'd' : toggle debug   'p' : toggle print   'q' : quit",
        ]

        if self.cfg.debug_mode:
            conf_min = "NA" if dbg["conf_min"] is None else f"{dbg['conf_min']:.2f}"
            conf_med = "NA" if dbg["conf_med"] is None else f"{dbg['conf_med']:.2f}"
            conf_max = "NA" if dbg["conf_max"] is None else f"{dbg['conf_max']:.2f}"
            conf_top = ",".join([f"{v:.2f}" for v in dbg["conf_top"][:5]])
            hist = " ".join([str(x) for x in dbg["conf_hist"]])
            mask_top = ",".join([f"{v*100:.1f}%" for v in dbg["mask_area_top"][:5]])

            lines += [
                f"[RAW] n_boxes={dbg['n_boxes']}  conf(min/med/max)={conf_min}/{conf_med}/{conf_max}",
                f"[RAW] conf_top5={conf_top}",
                f"[RAW] conf_hist(10bins)={hist}",
                f"[RAW] masks={dbg['has_masks']}  mask_area_top5={mask_top}",
                f"[DRAW] topk={self.cfg.debug_topk_draw} (no threshold)",
            ]
        return lines

    # ---- input handling ----
    def handle_key_vp_not_ready(self, key: int, frame_bgr: np.ndarray) -> bool:
        """VP未セット時のキー処理。Trueならループ終了。"""
        if key in (ord("q"), 27):
            return True

        if key == ord("s"):
            self.setup_vpe_from_frame(frame_bgr)

        if key == ord("d"):
            self.cfg.debug_mode = not self.cfg.debug_mode
            print(f"[INFO] debug_mode: {self.cfg.debug_mode}")

        if key == ord("p"):
            self.cfg.debug_print = not self.cfg.debug_print
            print(f"[INFO] debug_print: {self.cfg.debug_print}")

        return False

    def handle_key_running(self, key: int) -> bool:
        """推論中のキー処理。Trueならループ終了。"""
        if key in (ord("q"), 27):
            return True

        if key == ord("r"):
            self.reset_vp()

        if key == ord("m"):
            self.draw_contour = not self.draw_contour
            print(f"[INFO] contour drawing: {'ON' if self.draw_contour else 'OFF'}")

        if key == ord("d"):
            self.cfg.debug_mode = not self.cfg.debug_mode
            print(f"[INFO] debug_mode: {self.cfg.debug_mode}")

        if key == ord("p"):
            self.cfg.debug_print = not self.cfg.debug_print
            print(f"[INFO] debug_print: {self.cfg.debug_print}")

        return False

    # ---- main loop ----
    def run(self) -> None:
        ensure_dir(self.cfg.save_dir, clear=self.cfg.clear_save_dir)
        print(f"[INFO] SAVE_DIR = {self.cfg.save_dir}")

        cv2.namedWindow(self.cfg.window_name, cv2.WINDOW_NORMAL)
        self.open_camera()

        try:
            while True:
                frame = self.cam.grab_frame(self.payload)
                if frame is None:
                    continue

                key = cv2.waitKey(1) & 0xFF

                if not self.vp_ready:
                    disp = frame.copy()
                    draw_overlay(
                        disp,
                        [
                            "'s' : capture -> save -> select ROI -> set VPE (one-time)",
                            "'d' : toggle debug   'p' : toggle print",
                            "'q' : quit",
                            f"SAVE_DIR: {self.cfg.save_dir}",
                        ],
                    )
                    cv2.imshow(self.cfg.window_name, disp)

                    if self.handle_key_vp_not_ready(key, frame):
                        break
                    continue

                r, infer_ms = self.predict_fast(frame)
                if r is None:
                    # predict失敗 → reset_vp 済み
                    continue

                dbg = summarize_detections(r, bins=10)
                if self.cfg.debug_print:
                    print("[DBG]", dbg)

                disp = frame.copy()
                n_inst, best_conf, has_masks = self.draw_predictions(disp, r)

                fps = self.update_fps()
                lines = self.build_overlay_lines(infer_ms, fps, n_inst, best_conf, has_masks, dbg)
                draw_overlay(disp, lines)
                cv2.imshow(self.cfg.window_name, disp)

                if self.handle_key_running(key):
                    break

        finally:
            self.close_camera()
            cv2.destroyAllWindows()


def main() -> None:
    cfg = AppConfig()
    app = VPApp(cfg)
    app.run()


if __name__ == "__main__":
    main()