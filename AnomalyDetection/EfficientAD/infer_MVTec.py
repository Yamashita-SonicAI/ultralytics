#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from anomalib.models import EfficientAd


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ★固定保存先（ここに必ず保存される）
DEFAULT_BASE_OUT = Path("/home/sonicai/workspace/yamada-develop/yolo-dev/AnomalyDetection/EfficientAD")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    paths = []
    for p in sorted(input_path.rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return paths


def to_tensor_nchw_rgb_u8(img_rgb_u8: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(img_rgb_u8).permute(2, 0, 1).contiguous()
    x = x.float() / 255.0
    return x.unsqueeze(0)


def normalize01(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32)
    x -= x.min()
    m = x.max()
    if m > eps:
        x /= m
    return x


def make_weight_mask(h: int, w: int, margin: int = 32) -> np.ndarray:
    yy = np.ones(h, np.float32)
    xx = np.ones(w, np.float32)

    if margin > 0:
        ramp_y = np.linspace(0.0, 1.0, margin, dtype=np.float32)
        yy[:margin] = ramp_y
        yy[-margin:] = ramp_y[::-1]

        ramp_x = np.linspace(0.0, 1.0, margin, dtype=np.float32)
        xx[:margin] = ramp_x
        xx[-margin:] = ramp_x[::-1]

    w2d = np.outer(yy, xx)
    return np.clip(w2d, 1e-6, 1.0)


def overlay_heatmap_bgr(img_bgr: np.ndarray, heat01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat_u8 = (np.clip(heat01, 0, 1) * 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)  # BGR
    return cv2.addWeighted(img_bgr, 1.0 - alpha, heat_color, alpha, 0.0)


def generate_tiles(H: int, W: int, tile: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    assert tile > 0
    assert 0 <= overlap < tile
    stride = tile - overlap

    xs = list(range(0, max(1, W - tile + 1), stride))
    ys = list(range(0, max(1, H - tile + 1), stride))

    if len(xs) == 0:
        xs = [0]
    if len(ys) == 0:
        ys = [0]

    if xs[-1] != max(0, W - tile):
        xs.append(max(0, W - tile))
    if ys[-1] != max(0, H - tile):
        ys.append(max(0, H - tile))

    tiles = []
    for y in ys:
        for x in xs:
            x1, y1 = x, y
            x2, y2 = min(x + tile, W), min(y + tile, H)
            tiles.append((x1, y1, x2, y2))
    return tiles


def load_model(ckpt: Path, device: str) -> EfficientAd:
    model: EfficientAd = EfficientAd.load_from_checkpoint(str(ckpt))
    model.eval()
    model.to(device)
    return model


@torch.inference_mode()
def infer_tiles(
    model: EfficientAd,
    img_bgr: np.ndarray,
    tiles: List[Tuple[int, int, int, int]],
    model_image_size: int,
    batch_tiles: int,
    device: str,
) -> List[np.ndarray]:
    out_maps: List[np.ndarray] = []

    buf_imgs = []
    buf_meta = []  # (th, tw)

    def flush():
        nonlocal buf_imgs, buf_meta, out_maps
        if not buf_imgs:
            return
        x = torch.cat(buf_imgs, dim=0).to(device)
        pred = model(x)
        amap = pred.anomaly_map
        if amap.ndim == 4:
            amap = amap.squeeze(1)  # B,S,S
        amap = amap.detach().float().cpu().numpy()

        for i in range(amap.shape[0]):
            th, tw = buf_meta[i]
            a = cv2.resize(amap[i], (tw, th), interpolation=cv2.INTER_LINEAR)
            out_maps.append(a)

        buf_imgs = []
        buf_meta = []

    for (x1, y1, x2, y2) in tiles:
        tile_bgr = img_bgr[y1:y2, x1:x2]
        th, tw = tile_bgr.shape[:2]

        tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        tile_rgb_rs = cv2.resize(tile_rgb, (model_image_size, model_image_size), interpolation=cv2.INTER_AREA)

        ten = to_tensor_nchw_rgb_u8(tile_rgb_rs)
        buf_imgs.append(ten)
        buf_meta.append((th, tw))

        if len(buf_imgs) >= batch_tiles:
            flush()

    flush()
    return out_maps


def merge_anomaly_maps(
    H: int,
    W: int,
    tiles: List[Tuple[int, int, int, int]],
    tile_maps: List[np.ndarray],
    merge: str,
    weight_margin: int,
) -> np.ndarray:
    if merge == "max":
        merged = np.zeros((H, W), np.float32)
        for (x1, y1, x2, y2), amap in zip(tiles, tile_maps):
            merged[y1:y2, x1:x2] = np.maximum(merged[y1:y2, x1:x2], amap.astype(np.float32))
        return merged

    if merge == "weighted":
        acc = np.zeros((H, W), np.float32)
        wsum = np.zeros((H, W), np.float32)
        for (x1, y1, x2, y2), amap in zip(tiles, tile_maps):
            th, tw = (y2 - y1), (x2 - x1)
            margin = min(weight_margin, th // 2, tw // 2)
            w = make_weight_mask(th, tw, margin=margin)
            acc[y1:y2, x1:x2] += amap.astype(np.float32) * w
            wsum[y1:y2, x1:x2] += w
        return acc / np.clip(wsum, 1e-6, None)

    raise ValueError(f"Unknown merge mode: {merge}")


def score_from_map(anomaly01: np.ndarray, mode: str) -> float:
    if mode == "max":
        return float(np.max(anomaly01))
    if mode == "p99":
        return float(np.percentile(anomaly01, 99.0))
    raise ValueError(mode)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="EfficientAD .ckpt path")
    ap.add_argument("--mvtec_root", type=str, required=True, help="MVTec root dir")
    ap.add_argument("--category", type=str, required=True, help="e.g. bottle, hazelnut ...")

    # ★ out は任意指定にして、未指定なら固定パスへ
    ap.add_argument("--out", type=str, default=str(DEFAULT_BASE_OUT), help="base output dir (default fixed path)")

    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--model_image_size", type=int, default=256)
    ap.add_argument("--tile_size", type=int, default=896)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--batch_tiles", type=int, default=16)
    ap.add_argument("--merge", type=str, default="max", choices=["max", "weighted"])
    ap.add_argument("--weight_margin", type=int, default=32)
    ap.add_argument("--overlay_alpha", type=float, default=0.45)
    ap.add_argument("--save_map_npy", action="store_true")
    ap.add_argument("--score_mode", type=str, default="p99", choices=["max", "p99"])
    return ap.parse_args()


def main():
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Falling back to CPU.")
        device = "cpu"

    ckpt = Path(args.ckpt)
    mvtec_root = Path(args.mvtec_root)
    test_dir = mvtec_root / args.category / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"MVTec test dir not found: {test_dir}")

    # ★最終保存先：<base>/<category>/
    base_out = Path(args.out).expanduser().resolve()
    out_cat = base_out / args.category
    vis_root = out_cat / "vis"
    map_root = out_cat / "maps"
    ensure_dir(vis_root)
    ensure_dir(map_root)

    model = load_model(ckpt, device=device)

    images = list_images(test_dir)
    if not images:
        raise FileNotFoundError(f"No images found in: {test_dir}")

    print(f"[INFO] device={device}")
    print(f"[INFO] test_dir={test_dir}")
    print(f"[INFO] out_dir={out_cat} (base={base_out})")
    print(f"[INFO] images={len(images)}")

    for img_path in images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] cannot read: {img_path}")
            continue

        H, W = img_bgr.shape[:2]
        tiles = generate_tiles(H, W, tile=args.tile_size, overlap=args.overlap)

        t0 = time.perf_counter()
        tile_maps = infer_tiles(
            model=model,
            img_bgr=img_bgr,
            tiles=tiles,
            model_image_size=args.model_image_size,
            batch_tiles=args.batch_tiles,
            device=device,
        )

        merged = merge_anomaly_maps(
            H=H, W=W,
            tiles=tiles,
            tile_maps=tile_maps,
            merge=args.merge,
            weight_margin=args.weight_margin,
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0

        merged01 = normalize01(merged)
        score = score_from_map(merged01, mode=args.score_mode)

        # ★test_dir からの相対パスを保持して保存（good/xxx.png など）
        rel = img_path.relative_to(test_dir)  # e.g. good/000.png
        rel_parent = rel.parent               # e.g. good
        stem = img_path.stem

        vis_dir = vis_root / rel_parent
        ensure_dir(vis_dir)
        vis = overlay_heatmap_bgr(img_bgr, merged01, alpha=args.overlay_alpha)
        cv2.putText(
            vis,
            f"score({args.score_mode})={score:.4f} tiles={len(tiles)} infer={infer_ms:.1f}ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        vis_path = vis_dir / f"{stem}_vis.png"
        cv2.imwrite(str(vis_path), vis)

        if args.save_map_npy:
            map_dir = map_root / rel_parent
            ensure_dir(map_dir)
            np.save(str(map_dir / f"{stem}_anomaly.npy"), merged01)

        print(f"[OK] {rel} -> score={score:.4f} tiles={len(tiles)} infer={infer_ms:.1f}ms saved={vis_path}")

    print(f"[DONE] outputs in: {out_cat}")


if __name__ == "__main__":
    main()