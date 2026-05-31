#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Physics-aware deep learning for tropical cyclone track prediction
End-to-end pipeline: weakly supervised eye localisation + track forecasting
================================================================================

A single, self-contained reference implementation of the two-stage pipeline
described in the manuscript, operating exclusively on single-channel
geostationary infrared imagery from the Digital Typhoon V2 (Western Pacific)
archive.

  STAGE 1  -- Weakly supervised eye localisation (YOLOv8n)
    3.1.1  intensity-aware adaptive pseudo-labels (wind-radius derived box)
    3.1.2  centre-aware selection at inference
    3.1.3  azimuthal-equidistant geometry (4.88 km/px) + training

  STAGE 2  -- Physics-aware track forecasting (Transformer)
    3.2.1  Strict Tracks (frozen YOLO run once; no ground truth at input time)
    3.2.2  13-feature kinematic state vector, T=12 h window, H={1,3,6,9,12} h
    3.2.3  pre-LayerNorm Transformer (d=128, 8 heads, 3 layers -> 599,690 params)
           composite loss  L = L_dist + 0.2 L_dir + 0.05 L_speed
    3.3    persistence / linear baselines, Haversine error, bootstrap 95% CIs,
           JMA-grade and confidence-quintile stratification

  EXPLAINABILITY
    4.2.1  permutation feature importance
    4.2.2  attention rollout (Abnar & Zuidema 2020) + integrated gradients
    4.2.3  Stage-1 EigenCAM attention-eye alignment (model[14] P3 fusion)

The two stages share a single CONFIG so geometry, storm filtering and the
strict storm-level 70/15/15 split are identical across the whole pipeline.

Run:
    python cyclone_pipeline.py            # runs the stages enabled in RUN
    python cyclone_pipeline.py --self-test  # verify architecture w/o the dataset

NOTE: this is a clean reference implementation of the *method*. Reproducing the
exact published numbers requires the dataset and a trained Stage-1 checkpoint;
two minor loss details (the horizon-weighting scheme and the IG step count) are
documented design choices.
================================================================================
"""

from __future__ import annotations

import os
import glob
import math
import random
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# UNIFIED CONFIGURATION  (shared by both stages)
# ============================================================================
CONFIG = {
    # ---- dataset roots (Kaggle layout) ----
    "image_root":    "/kaggle/input/digital-typhoon-dataset-western-pacific-wp/image_png/image_png",
    "metadata_root": "/kaggle/input/digital-typhoon-dataset-western-pacific-wp/metadata/metadata",
    "work_root":     "/kaggle/working",

    # ---- geometry (azimuthal equidistant, 1250 km radius -> 1250/256 km/px) ----
    "km_per_pixel": 1250.0 / 256.0,    # = 4.8828125
    "nm_per_pixel": 0.926,             # for the adaptive box (paper Delta_nm/px)
    "img_size": 512,

    # ---- storm filtering (3.1 / 2.2) ----
    "min_frames": 30,
    "grades_keep": (3, 4, 5),          # TS / STS / TY
    "year_min": 1987,                  # GMS-3 onwards

    # ---- strict storm-level split (shared by both stages) ----
    "seed": 42,
    "split_ratios": (0.70, 0.15, 0.15),

    # ---- Stage-1 YOLO (3.1.3) ----
    "yolo_pretrained": "yolov8n.pt",
    "yolo_project": "runs_eye",
    "yolo_run_name": "yolo_eye_full_v2",
    "yolo_epochs": 50,
    "yolo_batch": 32,
    "yolo_lr0": 1e-3,
    "yolo_weight_decay": 5e-4,
    "yolo_warmup_epochs": 3,
    "max_per_storm": None,             # set an int for the capped "Pass 1"; None = full data

    # adaptive box clip range (3.1.1)
    "box_min_px": 32,
    "box_max_px": 128,
    "box_default_px": 64,

    # ---- inference / Strict Tracks ----
    "conf_thres": 0.05,                # full-test eval + Strict Tracks
    "tracks_csv": "yolo_tracks_strict.csv",

    # ---- trained Stage-1 weights used for Strict Tracks / XAI ----
    # After training: <work_root>/runs_eye/yolo_eye_full_v2/weights/best.pt
    "yolo_weights": None,              # None -> use freshly trained best.pt
}

# Stage-2 forecaster hyper-parameters (3.2.3)
T_CONFIG = {
    "seq_len": 12,
    "horizons": (1, 3, 6, 9, 12),
    "input_dim": 13,
    "d_model": 128,
    "nhead": 8,
    "num_layers": 3,
    "dim_feedforward": 512,
    "dropout": 0.1,

    "batch_size": 256,
    "epochs": 60,
    "warmup_epochs": 3,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "early_stop_patience": 10,

    "alpha_dir": 0.2,
    "beta_speed": 0.05,
    "speed_cap_kmh": 80.0,
    "dir_mask_deg": 0.05,
    "cadence_tol_h": 0.25,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# Exact 13-feature state vector order (3.2.2):
#   [ d_phi^(0), d_lambda^(0), v_lambda, v_phi, a_lambda, a_phi,
#     W, Wdot, c, sin(2*pi*h/24), cos(2*pi*h/24), sin(phi), cos(phi) ]
FEATURE_NAMES = [
    "d_lat_off_anchor", "d_lon_off_anchor",
    "v_lon", "v_lat", "a_lon", "a_lat",
    "wind", "dwind_dt", "conf",
    "sin_hour", "cos_hour", "sin_lat", "cos_lat",
]
N_FEATURES = len(FEATURE_NAMES)
N_HORIZON = len(T_CONFIG["horizons"])

EARTH_R_KM = 6371.0
KM_PER_DEG = 111.0


# ============================================================================
# Geometry helpers
# ============================================================================
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def pixel_offset_to_latlon(dx_px, dy_px, lat_ref):
    """Pixel offset from image centre -> (dlat, dlon) in degrees.
    Storm-centred azimuthal-equidistant crop; image y grows downward."""
    km = CONFIG["km_per_pixel"]
    dlat = -(dy_px * km) / KM_PER_DEG
    dlon = (dx_px * km) / (KM_PER_DEG * math.cos(math.radians(lat_ref)))
    return dlat, dlon


def storm_level_split(storm_ids, ratios=None, seed=None):
    """Strict storm-level 70/15/15 split. No storm in more than one split."""
    ratios = ratios or CONFIG["split_ratios"]
    seed = CONFIG["seed"] if seed is None else seed
    storms = sorted(set(storm_ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(storms)
    n = len(storms)
    n_tr, n_va = int(ratios[0] * n), int(ratios[1] * n)
    train = set(storms[:n_tr])
    val = set(storms[n_tr:n_tr + n_va])
    test = set(storms[n_tr + n_va:])
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert len(train) + len(val) + len(test) == n
    return train, val, test


# ============================================================================
# STAGE 1 -- data preparation
# ============================================================================
def build_frame_index():
    """Scan metadata + PNGs into one frame-level DataFrame, then apply the
    Stage-1 quality filter (grade in {3,4,5}, year >= 1987, >= 30 frames)."""
    rows = []
    for csv_path in sorted(glob.glob(os.path.join(CONFIG["metadata_root"], "*.csv"))):
        sid = os.path.basename(csv_path).replace(".csv", "")
        img_dir = os.path.join(CONFIG["image_root"], sid)
        if not os.path.isdir(img_dir):
            continue
        try:
            df = pd.read_csv(csv_path, dtype={"file_1": str})
        except Exception:
            continue
        avail = set(os.listdir(img_dir))
        for _, r in df.iterrows():
            fname = str(r.get("file_1", "")).replace(".h5", ".png")
            if fname not in avail:
                continue
            rows.append({
                "storm_id": sid,
                "frame_path": os.path.join(img_dir, fname),
                "frame_stem": fname[:-4],
                "year": int(r["year"]), "month": int(r["month"]),
                "day": int(r["day"]), "hour": int(r["hour"]),
                "lat": float(r["lat"]), "lng": float(r["lng"]),
                "wind": float(r.get("wind", 0.0) or 0.0),
                "grade": int(r.get("grade", 0)),
                "long30": float(r.get("long30", 0.0) or 0.0),
                "short30": float(r.get("short30", 0.0) or 0.0),
            })
    idx = pd.DataFrame(rows)
    idx = idx[(idx["grade"].isin(CONFIG["grades_keep"])) & (idx["year"] >= CONFIG["year_min"])]
    counts = idx["storm_id"].value_counts()
    keep = counts[counts >= CONFIG["min_frames"]].index
    idx = idx[idx["storm_id"].isin(keep)].reset_index(drop=True)
    print(f"Frame index: {len(idx)} frames from {idx['storm_id'].nunique()} storms "
          f"(after grade/year/min-frames filter)")
    return idx


def adaptive_box_size_px(row):
    """Intensity-aware adaptive pseudo-label box edge (3.1.1, Eq. 1).
    Derived from 30-knot wind radii when available; else a wind-speed heuristic."""
    r30_long, r30_short = row.get("long30", 0) or 0, row.get("short30", 0) or 0
    if r30_long > 0 and r30_short > 0:
        avg_r30_nm = (r30_long + r30_short) / 2.0
        box = 0.25 * (avg_r30_nm / CONFIG["nm_per_pixel"])
        return int(np.clip(box, CONFIG["box_min_px"], CONFIG["box_max_px"]))
    wind = row.get("wind", 0) or 0
    if wind >= 85:
        return 48
    if wind >= 64:
        return 56
    if wind >= 34:
        return 64
    return CONFIG["box_default_px"]


def materialize_yolo_dataset(frame_index):
    """Write images/<split> + labels/<split> with adaptive centred pseudo-labels."""
    from tqdm import tqdm
    out_root = os.path.join(CONFIG["work_root"], "yolo_eye_full")
    img_size = CONFIG["img_size"]
    train_s, val_s, test_s = storm_level_split(frame_index["storm_id"].unique())
    frame_index = frame_index.copy()
    frame_index["split"] = frame_index["storm_id"].apply(
        lambda s: "train" if s in train_s else ("val" if s in val_s else "test"))

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(out_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_root, "labels", split), exist_ok=True)

    cap = CONFIG["max_per_storm"]
    for split in ("train", "val", "test"):
        dfs = frame_index[frame_index["split"] == split]
        if cap is not None:
            dfs = dfs.groupby("storm_id", group_keys=False).apply(
                lambda g: g.sample(min(len(g), cap), random_state=CONFIG["seed"]))
        for _, r in tqdm(dfs.iterrows(), total=len(dfs), desc=f"materialise {split}"):
            name = f"{r['storm_id']}_{r['frame_stem']}.png"
            dst_img = os.path.join(out_root, "images", split, name)
            dst_lbl = os.path.join(out_root, "labels", split, name.replace(".png", ".txt"))
            if not os.path.exists(dst_img):
                try:
                    os.symlink(r["frame_path"], dst_img)
                except OSError:
                    import shutil
                    shutil.copy(r["frame_path"], dst_img)
            box_px = adaptive_box_size_px(r)
            bn = box_px / img_size
            with open(dst_lbl, "w") as f:
                f.write(f"0 0.5 0.5 {bn:.6f} {bn:.6f}\n")

    # data.yaml
    import yaml
    yaml_path = os.path.join(out_root, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump({"path": out_root, "train": "images/train", "val": "images/val",
                   "test": "images/test", "nc": 1, "names": ["cyclone_eye"]}, f)
    print(f"YOLO dataset at {out_root}")
    frame_index.to_parquet(os.path.join(CONFIG["work_root"], "frame_index.parquet"))
    return out_root


# ============================================================================
# STAGE 1 -- training (3.1.3)
# ============================================================================
def train_yolo(data_yaml=None):
    from ultralytics import YOLO
    data_yaml = data_yaml or os.path.join(CONFIG["work_root"], "yolo_eye_full", "data.yaml")
    model = YOLO(CONFIG["yolo_pretrained"])
    model.train(
        data=data_yaml,
        epochs=CONFIG["yolo_epochs"],
        imgsz=CONFIG["img_size"],
        batch=CONFIG["yolo_batch"],
        optimizer="AdamW",
        lr0=CONFIG["yolo_lr0"],
        weight_decay=CONFIG["yolo_weight_decay"],
        warmup_epochs=CONFIG["yolo_warmup_epochs"],
        cos_lr=True,
        # storm orientation is non-canonical: full rotation + flips on;
        # colour jitter / mosaic / mixup / copy-paste OFF to preserve the centred-eye prior
        degrees=180.0, fliplr=0.5, flipud=0.5,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        project=CONFIG["yolo_project"], name=CONFIG["yolo_run_name"],
        exist_ok=True, save=True, verbose=True, device=0,
    )
    best = os.path.join(CONFIG["work_root"], CONFIG["yolo_project"],
                        CONFIG["yolo_run_name"], "weights", "best.pt")
    print(f"Stage-1 weights -> {best}")
    return best


def _stage1_weights():
    if CONFIG["yolo_weights"]:
        return CONFIG["yolo_weights"]
    return os.path.join(CONFIG["work_root"], CONFIG["yolo_project"],
                        CONFIG["yolo_run_name"], "weights", "best.pt")


# ============================================================================
# STAGE 1 -- full-test evaluation (3.1, Table 1)
# ============================================================================
def evaluate_stage1(weights=None):
    import cv2
    from ultralytics import YOLO
    from tqdm import tqdm
    weights = weights or _stage1_weights()
    model = YOLO(weights)
    img_size = CONFIG["img_size"]
    km = CONFIG["km_per_pixel"]

    fi = pd.read_parquet(os.path.join(CONFIG["work_root"], "frame_index.parquet"))
    test = fi[fi["split"] == "test"]
    errors, grades = [], []
    n_fallback = 0

    paths = [(f"{r['storm_id']}_{r['frame_stem']}.png", r["grade"]) for _, r in test.iterrows()]
    img_root = os.path.join(CONFIG["work_root"], "yolo_eye_full", "images", "test")
    B = 64
    for i in tqdm(range(0, len(paths), B), desc="Stage-1 eval"):
        chunk = paths[i:i + B]
        fps = [os.path.join(img_root, n) for n, _ in chunk]
        res = model(fps, conf=CONFIG["conf_thres"], iou=0.5, verbose=False)
        for (name, grade), r in zip(chunk, res):
            H, W = r.orig_shape
            cx, cy = W / 2.0, H / 2.0
            if len(r.boxes) == 0:
                px, py = cx, cy
                n_fallback += 1
            else:
                best = min(r.boxes, key=lambda b: math.hypot(
                    float(b.xywh[0][0]) - cx, float(b.xywh[0][1]) - cy))
                px, py = float(best.xywh[0][0]), float(best.xywh[0][1])
            err_km = math.hypot(px - cx, py - cy) * km   # equidistant projection
            errors.append(err_km)
            grades.append(grade)

    errors = np.array(errors)
    grades = np.array(grades)
    print("\n=== STAGE 1 eye localisation (test set) ===")
    print(f"  n={len(errors)}  fallback={n_fallback} ({100*n_fallback/max(len(errors),1):.4f}%)")
    print(f"  median {np.median(errors):.2f} km | mean {errors.mean():.2f} km | "
          f"90th {np.percentile(errors,90):.2f} | 95th {np.percentile(errors,95):.2f}")
    for g, lab in [(3, "TS"), (4, "STS"), (5, "TY")]:
        sel = grades == g
        if sel.sum():
            print(f"  Grade {g} ({lab}, n={sel.sum():5d}): median {np.median(errors[sel]):.2f} | "
                  f"mean {errors[sel].mean():.2f} km")
    return errors, grades


# ============================================================================
# STAGE 2 -- 3.2.1 Strict Tracks
# ============================================================================
def generate_strict_yolo_tracks(out_csv=None, weights=None):
    import cv2
    from ultralytics import YOLO
    from tqdm import tqdm
    out_csv = out_csv or CONFIG["tracks_csv"]
    weights = weights or _stage1_weights()
    model = YOLO(weights)
    grades_keep = set(CONFIG["grades_keep"])

    csv_files = sorted(glob.glob(os.path.join(CONFIG["metadata_root"], "*.csv")))
    cleaned, valid = {}, []
    for csv_path in csv_files:
        sid = os.path.basename(csv_path).replace(".csv", "")
        try:
            df = pd.read_csv(csv_path, dtype={"file_1": str})
        except Exception:
            continue
        if len(df) == 0 or "grade" not in df.columns:
            continue
        df = df[(df["grade"].isin(grades_keep)) & (df["year"] >= CONFIG["year_min"])]
        if len(df) >= CONFIG["min_frames"]:
            cleaned[sid] = df
            valid.append(sid)

    train_s, val_s, test_s = storm_level_split(valid)
    split_of = lambda s: "train" if s in train_s else ("val" if s in val_s else "test")
    print(f"Strict Tracks over {len(valid)} storms "
          f"(train {len(train_s)}/val {len(val_s)}/test {len(test_s)})")

    rows, n_fb, n_tot = [], 0, 0
    for sid in tqdm(valid, desc="Strict Tracks"):
        df = cleaned[sid].sort_values(["year", "month", "day", "hour"])
        img_dir = os.path.join(CONFIG["image_root"], sid)
        if not os.path.isdir(img_dir):
            continue
        avail = set(os.listdir(img_dir))
        last_lat = last_lon = None
        for _, r in df.iterrows():
            fname = str(r.get("file_1", "")).replace(".h5", ".png")
            dt = datetime(int(r["year"]), int(r["month"]), int(r["day"]), int(r["hour"]))
            pred_lat = pred_lon = np.nan
            conf, fb = 0.0, 1
            if fname in avail:
                img = cv2.imread(os.path.join(img_dir, fname))
                if img is not None:
                    res = model(img, conf=CONFIG["conf_thres"], verbose=False)[0]
                    if len(res.boxes) > 0:
                        H, W = img.shape[:2]
                        cx, cy = W / 2.0, H / 2.0
                        best = min(res.boxes, key=lambda b: math.hypot(
                            (float(b.xyxy[0][0]) + float(b.xyxy[0][2])) / 2 - cx,
                            (float(b.xyxy[0][1]) + float(b.xyxy[0][3])) / 2 - cy))
                        bx = (float(best.xyxy[0][0]) + float(best.xyxy[0][2])) / 2
                        by = (float(best.xyxy[0][1]) + float(best.xyxy[0][3])) / 2
                        conf = float(best.conf[0])
                        dlat, dlon = pixel_offset_to_latlon(bx - cx, by - cy, r["lat"])
                        pred_lat, pred_lon = r["lat"] + dlat, r["lng"] + dlon
                        last_lat, last_lon = pred_lat, pred_lon
                        fb = 0
            if fb == 1:
                n_fb += 1
                if last_lat is not None:
                    pred_lat, pred_lon = last_lat, last_lon
                else:
                    pred_lat = pred_lon = np.nan
            n_tot += 1
            rows.append({"sid": sid, "dt": dt.isoformat(), "hour": int(r["hour"]),
                         "pred_lat": pred_lat, "pred_lon": pred_lon, "conf": conf,
                         "fallback": fb, "wind": float(r.get("wind", 0.0) or 0.0),
                         "grade": int(r.get("grade", 0)),
                         "gt_lat": float(r["lat"]), "gt_lon": float(r["lng"]),
                         "split": split_of(sid)})

    tdf = pd.DataFrame(rows).dropna(subset=["pred_lat", "pred_lon"])
    tdf.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(tdf)} rows); fallback {100*n_fb/max(n_tot,1):.4f}%")
    return tdf


# ============================================================================
# STAGE 2 -- 3.2.2 feature/sequence dataset
# ============================================================================
class StrictTrackDataset(Dataset):
    def __init__(self, df, mode="train", wind_stats=None):
        self.seq_len = T_CONFIG["seq_len"]
        self.horizons = list(T_CONFIG["horizons"])
        self.max_h = max(self.horizons)
        self.tol = T_CONFIG["cadence_tol_h"]
        self.samples = []
        if mode == "train":
            w = df["wind"].values.astype(float)
            self.wind_min, self.wind_max = float(np.min(w)), float(np.max(w))
        else:
            self.wind_min, self.wind_max = wind_stats
        self._wspan = max(self.wind_max - self.wind_min, 1e-6)
        for _, g in df.groupby("sid"):
            g = g.copy()
            g["dt"] = pd.to_datetime(g["dt"])
            self._build(g.sort_values("dt").reset_index(drop=True))
        self.wind_stats = (self.wind_min, self.wind_max)

    def _build(self, g):
        seq_len, horizons, max_h = self.seq_len, self.horizons, self.max_h
        lat = g["pred_lat"].values.astype(float)
        lon = g["pred_lon"].values.astype(float)
        wind = g["wind"].values.astype(float)
        conf = g["conf"].values.astype(float)
        hour = g["hour"].values.astype(float)
        grade = g["grade"].values.astype(int)
        gt_lat = g["gt_lat"].values.astype(float)
        gt_lon = g["gt_lon"].values.astype(float)
        times = g["dt"].values.astype("datetime64[s]").astype("float64") / 3600.0

        v_lat = np.diff(lat, prepend=lat[0]); v_lon = np.diff(lon, prepend=lon[0])
        a_lat = np.diff(v_lat, prepend=v_lat[0]); a_lon = np.diff(v_lon, prepend=v_lon[0])
        wdot = np.diff(wind, prepend=wind[0])
        n_wind = (wind - self.wind_min) / self._wspan
        sin_h = np.sin(2 * np.pi * hour / 24.0); cos_h = np.cos(2 * np.pi * hour / 24.0)
        sin_lat = np.sin(np.radians(lat)); cos_lat = np.cos(np.radians(lat))

        expected = np.array([0] + list(range(1, seq_len)) +
                            [seq_len - 1 + h for h in horizons], dtype=float)
        n = len(g)
        for i in range(n - (seq_len + max_h)):
            a = i + seq_len - 1
            need = list(range(i, a + 1)) + [a + h for h in horizons]
            if np.any(np.abs((times[need] - times[i]) - expected) > self.tol):
                continue
            win = slice(i, i + seq_len)
            x = np.stack([
                lat[win] - lat[i], lon[win] - lon[i],
                v_lon[win], v_lat[win], a_lon[win], a_lat[win],
                n_wind[win], wdot[win], conf[win],
                sin_h[win], cos_h[win], sin_lat[win], cos_lat[win],
            ], axis=1).astype(np.float32)
            fut = [a + h for h in horizons]
            y = np.stack([gt_lat[fut] - lat[a], gt_lon[fut] - lon[a]], axis=1).astype(np.float32)
            self.samples.append({
                "x": torch.from_numpy(x), "y": torch.from_numpy(y),
                "anchor_lat": float(lat[a]), "anchor_lon": float(lon[a]),
                "v_lat": float(v_lat[a]), "v_lon": float(v_lon[a]),
                "grade": int(grade[a]), "conf_mean": float(np.mean(conf[win])),
                "gt_lat": gt_lat[fut].astype(np.float32),
                "gt_lon": gt_lon[fut].astype(np.float32),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch):
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "anchor_lat": torch.tensor([b["anchor_lat"] for b in batch], dtype=torch.float32),
        "anchor_lon": torch.tensor([b["anchor_lon"] for b in batch], dtype=torch.float32),
        "v_lat": torch.tensor([b["v_lat"] for b in batch], dtype=torch.float32),
        "v_lon": torch.tensor([b["v_lon"] for b in batch], dtype=torch.float32),
        "grade": torch.tensor([b["grade"] for b in batch], dtype=torch.long),
        "conf_mean": torch.tensor([b["conf_mean"] for b in batch], dtype=torch.float32),
        "gt_lat": torch.stack([torch.from_numpy(b["gt_lat"]) for b in batch]),
        "gt_lon": torch.stack([torch.from_numpy(b["gt_lon"]) for b in batch]),
    }


# ============================================================================
# STAGE 2 -- 3.2.3 model (599,690 params) + composite loss
# ============================================================================
class PreLNEncoderLayer(nn.Module):
    """Pre-LayerNorm encoder layer; identical parameterisation to
    nn.TransformerEncoderLayer but exposes the head-averaged attention map."""

    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.last_attn = None

    def forward(self, x, collect_attn=False):
        h = self.norm1(x)
        attn_out, attn_w = self.self_attn(h, h, h, need_weights=collect_attn,
                                          average_attn_weights=True)
        if collect_attn:
            self.last_attn = attn_w.detach()
        x = x + self.dropout1(attn_out)
        h = self.norm2(x)
        x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(h)))))
        return x


class PhysicsAwareForecaster(nn.Module):
    def __init__(self, cfg=T_CONFIG):
        super().__init__()
        d = cfg["d_model"]
        self.n_horizon = len(cfg["horizons"])
        self.input_proj = nn.Linear(cfg["input_dim"], d)
        self.pos_emb = nn.Parameter(torch.randn(1, cfg["seq_len"], d) * 0.02)
        self.layers = nn.ModuleList([
            PreLNEncoderLayer(d, cfg["nhead"], cfg["dim_feedforward"], cfg["dropout"])
            for _ in range(cfg["num_layers"])])
        self.final_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, self.n_horizon * 2)

    def forward(self, x, collect_attn=False):
        h = self.input_proj(x) + self.pos_emb
        for layer in self.layers:
            h = layer(h, collect_attn=collect_attn)
        out = self.head(self.final_norm(h[:, -1, :]))
        return out.view(-1, self.n_horizon, 2)

    def attention_maps(self):
        return [layer.last_attn for layer in self.layers]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class PhysicsAwareLoss(nn.Module):
    """L = L_dist + alpha*L_dir + beta*L_speed  (Eq. 4)."""

    def __init__(self, target_scale, cfg=T_CONFIG):
        super().__init__()
        self.alpha, self.beta = cfg["alpha_dir"], cfg["beta_speed"]
        self.cap, self.mask_deg = cfg["speed_cap_kmh"], cfg["dir_mask_deg"]
        self.smooth = nn.SmoothL1Loss(reduction="none")
        nh = len(cfg["horizons"])
        horizons = torch.tensor(cfg["horizons"], dtype=torch.float32)
        self.register_buffer("hweight", (horizons / horizons.mean()).view(1, nh, 1))
        self.register_buffer("scale", target_scale.view(1, nh, 2) + 1e-6)
        gaps = torch.diff(torch.cat([torch.zeros(1), horizons]))  # [1,2,3,3,3]
        self.register_buffer("dt", gaps.view(1, nh))

    def forward(self, pred, target, anchor_lat):
        l_dist = (self.smooth(pred / self.scale, target / self.scale) * self.hweight).mean()

        p12, t12 = pred[:, -1, :], target[:, -1, :]
        cos = F.cosine_similarity(p12, t12, dim=1, eps=1e-8)
        mask = (torch.linalg.norm(t12, dim=1) >= self.mask_deg).float()
        l_dir = ((1.0 - cos) * mask).sum() / mask.sum().clamp(min=1.0)

        B = pred.shape[0]
        seq = torch.cat([torch.zeros(B, 1, 2, device=pred.device), pred], dim=1)
        step = seq[:, 1:, :] - seq[:, :-1, :]
        coslat = torch.cos(torch.deg2rad(anchor_lat)).view(B, 1)
        disp = torch.sqrt((step[:, :, 0] * KM_PER_DEG) ** 2 +
                          (step[:, :, 1] * KM_PER_DEG * coslat) ** 2 + 1e-8)
        l_speed = torch.relu(disp / self.dt.to(pred.device) - self.cap).mean()

        total = l_dist + self.alpha * l_dir + self.beta * l_speed
        return total, {"dist": float(l_dist.detach()), "dir": float(l_dir.detach()),
                       "speed": float(l_speed.detach())}


def _lr_lambda(epoch):
    warm, total = T_CONFIG["warmup_epochs"], T_CONFIG["epochs"]
    if epoch < warm:
        return (epoch + 1) / warm
    return 0.5 * (1.0 + math.cos(math.pi * (epoch - warm) / max(total - warm, 1)))


def train_forecaster(train_ds, val_ds, cfg=T_CONFIG):
    device = cfg["device"]
    ys = torch.stack([s["y"] for s in train_ds.samples])
    target_scale = ys.std(dim=0)
    model = PhysicsAwareForecaster(cfg).to(device)
    print(f"Forecaster parameters: {count_parameters(model):,}")
    criterion = PhysicsAwareLoss(target_scale, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    tl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate)
    vl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate)

    best_val, best_state, best_ep, bad = float("inf"), None, -1, 0
    for epoch in range(cfg["epochs"]):
        model.train()
        for b in tl:
            opt.zero_grad()
            loss, _ = criterion(model(b["x"].to(device)), b["y"].to(device),
                                b["anchor_lat"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
        sched.step()
        model.eval()
        ve = []
        with torch.no_grad():
            for b in vl:
                ve.append(_horizon_errors(model(b["x"].to(device)).cpu().numpy(), b)["model"][:, -1])
        v12 = float(np.concatenate(ve).mean())
        if v12 < best_val:
            best_val, best_ep, bad = v12, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"epoch {epoch+1:02d}/{cfg['epochs']}  val +12h={v12:7.2f} km  "
                  f"(best {best_val:.2f} @ {best_ep+1})")
        if bad >= cfg["early_stop_patience"]:
            print(f"early stop @ {epoch+1} (best {best_ep+1})")
            break
    model.load_state_dict(best_state)
    return model, target_scale


# ============================================================================
# STAGE 2 -- 3.3 baselines + evaluation
# ============================================================================
def _horizon_errors(pred_offsets, batch):
    horizons = np.array(T_CONFIG["horizons"], dtype=float)
    alat, alon = batch["anchor_lat"].numpy(), batch["anchor_lon"].numpy()
    vlat, vlon = batch["v_lat"].numpy(), batch["v_lon"].numpy()
    gt_lat, gt_lon = batch["gt_lat"].numpy(), batch["gt_lon"].numpy()
    m = haversine_km(gt_lat, gt_lon, alat[:, None] + pred_offsets[:, :, 0],
                     alon[:, None] + pred_offsets[:, :, 1])
    p = haversine_km(gt_lat, gt_lon, np.repeat(alat[:, None], len(horizons), 1),
                     np.repeat(alon[:, None], len(horizons), 1))
    l = haversine_km(gt_lat, gt_lon, alat[:, None] + vlat[:, None] * horizons[None, :],
                     alon[:, None] + vlon[:, None] * horizons[None, :])
    return {"model": m, "persist": p, "linear": l}


def _bootstrap_ci(values, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    values = np.asarray(values)
    means = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(n)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(model, test_ds, cfg=T_CONFIG):
    device, horizons = cfg["device"], list(cfg["horizons"])
    dl = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate)
    em, ep, el, grades, confs = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for b in dl:
            e = _horizon_errors(model(b["x"].to(device)).cpu().numpy(), b)
            em.append(e["model"]); ep.append(e["persist"]); el.append(e["linear"])
            grades.append(b["grade"].numpy()); confs.append(b["conf_mean"].numpy())
    em, ep, el = map(np.concatenate, (em, ep, el))
    grades, confs = np.concatenate(grades), np.concatenate(confs)

    print("\n=== STAGE 2 track forecasting (test set, n=%d) ===" % len(em))
    print(f"{'Method':<20}" + "".join(f"+{h:>2}h    " for h in horizons))
    for name, arr in [("Persistence", ep), ("Linear extrap.", el), ("Transformer (ours)", em)]:
        print(f"{name:<20}" + "".join(f"{arr[:, j].mean():7.2f} " for j in range(len(horizons))))
    lo, hi = _bootstrap_ci(em[:, -1])
    print(f"{'95% CI (+12h)':<20}[{lo:.2f}, {hi:.2f}]")

    print("\nStratified by JMA grade (mean km):")
    for g, lab in [(3, "TS"), (4, "STS"), (5, "TY")]:
        sel = grades == g
        if sel.sum():
            print(f"  Grade {g} ({lab}, n={sel.sum():5d})  " +
                  "  ".join(f"{em[sel, j].mean():6.2f}" for j in range(len(horizons))))

    print("\nStratified by Stage-1 confidence (quintiles, +12h mean km):")
    qe = np.quantile(confs, [0, .2, .4, .6, .8, 1.0])
    for q in range(5):
        sel = (confs >= qe[q]) & (confs <= qe[q + 1] if q == 4 else confs < qe[q + 1])
        if sel.sum():
            print(f"  Q{q+1} (n={sel.sum():5d}): {em[sel, -1].mean():6.2f} km")
    return {"model": em, "persist": ep, "linear": el, "grades": grades, "confs": confs}


# ============================================================================
# 4.2.1 permutation importance
# ============================================================================
def permutation_importance(model, test_ds, repeats=5, cfg=T_CONFIG, seed=0):
    device = cfg["device"]
    X = torch.stack([s["x"] for s in test_ds.samples]).to(device)
    base_batch = collate(test_ds.samples)
    model.eval()
    with torch.no_grad():
        base = _horizon_errors(model(X).cpu().numpy(), base_batch)["model"][:, -1].mean()
    rng = np.random.RandomState(seed)
    N = X.shape[0]
    print("\n=== Permutation feature importance (+12h inflation, km) ===")
    out = {}
    for f in range(X.shape[2]):
        deltas = []
        for _ in range(repeats):
            Xp = X.clone()
            Xp[:, :, f] = X[torch.from_numpy(rng.permutation(N)).to(device), :, f]
            with torch.no_grad():
                e = _horizon_errors(model(Xp).cpu().numpy(), base_batch)["model"][:, -1].mean()
            deltas.append(e - base)
        out[FEATURE_NAMES[f]] = (float(np.mean(deltas)), float(np.std(deltas)))
    for name, (m, s) in sorted(out.items(), key=lambda kv: -kv[1][0]):
        print(f"  {name:<18} +{m:6.2f} +/- {s:4.2f} km")
    return out


# ============================================================================
# 4.2.2 attention rollout + integrated gradients
# ============================================================================
def attention_rollout(model, test_ds, n=2048, cfg=T_CONFIG):
    device = cfg["device"]
    X = torch.stack([s["x"] for s in test_ds.samples[:n]]).to(device)
    model.eval()
    with torch.no_grad():
        _ = model(X, collect_attn=True)
        mats = model.attention_maps()
    T = X.shape[1]
    eye = torch.eye(T, device=device).unsqueeze(0)
    roll = None
    for A in mats:
        Ah = 0.5 * A + 0.5 * eye
        Ah = Ah / Ah.sum(dim=-1, keepdim=True)
        roll = Ah if roll is None else torch.bmm(Ah, roll)
    mass = roll[:, -1, :].mean(dim=0).cpu().numpy()
    mass = mass / mass.sum()
    last3, first3 = mass[-3:].sum(), mass[:3].sum()
    print("\n=== Attention rollout (n=%d) ===" % X.shape[0])
    print(f"  anchor-step mass: {mass[-1]*100:.1f}% | last3 {last3:.3f} | "
          f"first3 {first3:.3f} | recency {last3/max(first3,1e-9):.2f}x")
    return mass


def integrated_gradients(model, sample, steps=50, cfg=T_CONFIG):
    device = cfg["device"]
    model.eval()
    x = sample["x"].unsqueeze(0).to(device)
    baseline = torch.zeros_like(x)
    total = torch.zeros_like(x)
    for k in range(1, steps + 1):
        xk = (baseline + (k / steps) * (x - baseline)).clone().requires_grad_(True)
        scalar = torch.linalg.norm(model(xk)[0, -1, :])
        total += torch.autograd.grad(scalar, xk)[0]
    return ((x - baseline) * total / steps).squeeze(0).detach().cpu().numpy()


def ig_case_studies(model, test_ds, cfg=T_CONFIG):
    device = cfg["device"]
    batch = collate(test_ds.samples)
    with torch.no_grad():
        X = torch.stack([s["x"] for s in test_ds.samples]).to(device)
        e12 = _horizon_errors(model(X).cpu().numpy(), batch)["model"][:, -1]
    order = np.argsort(e12)
    cases = {"best": order[0], "median": order[len(order) // 2], "worst": order[-1]}
    print("\n=== Integrated gradients (best/median/worst +12h) ===")
    out = {}
    for label, idx in cases.items():
        attr = integrated_gradients(model, test_ds.samples[idx], cfg=cfg)
        share = np.abs(attr).sum(axis=0)
        share = 100 * share / max(share.sum(), 1e-9)
        top = int(np.argmax(share))
        out[label] = {"idx": int(idx), "err12": float(e12[idx]), "attr": attr, "share": share}
        print(f"  {label:<7} err={e12[idx]:7.2f} km  top: {FEATURE_NAMES[top]} ({share[top]:.1f}%)")
    return out


# ============================================================================
# 4.2.3 Stage-1 EigenCAM attention-eye alignment
# ============================================================================
def _gcam_mask(cam, border):
    m = cam.copy()
    m[:border, :] = 0; m[-border:, :] = 0; m[:, :border] = 0; m[:, -border:] = 0
    return m


def stage1_eigencam_alignment(n_frames=2000, weights=None):
    import cv2
    from ultralytics import YOLO
    try:
        from pytorch_grad_cam import EigenCAM
    except Exception as e:
        print(f"[skip] pytorch-grad-cam unavailable: {e}")
        return None
    weights = weights or _stage1_weights()
    yolo = YOLO(weights)
    target = yolo.model.model[14]                      # P3 fusion Concat

    class _Wrap(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            r = self.m(x)
            return r[0] if isinstance(r, tuple) else r

    cam = EigenCAM(model=_Wrap(yolo.model), target_layers=[target])
    sz, km = CONFIG["img_size"], CONFIG["km_per_pixel"]
    border = int(0.10 * sz)

    valid = [os.path.basename(p).replace(".csv", "")
             for p in glob.glob(os.path.join(CONFIG["metadata_root"], "*.csv"))]
    _, _, test_s = storm_level_split(valid)
    frames = []
    for sid in test_s:
        d = os.path.join(CONFIG["image_root"], sid)
        if os.path.isdir(d):
            frames += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png")]
    random.seed(CONFIG["seed"]); random.shuffle(frames)

    dists = []
    for fp in frames[:n_frames]:
        img = cv2.imread(fp)
        if img is None:
            continue
        img = cv2.resize(img, (sz, sz))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        try:
            m = _gcam_mask(cam(input_tensor=torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0))[0], border)
        except Exception:
            continue
        ys, xs = np.mgrid[0:sz, 0:sz]
        s = m.sum()
        if s <= 0:
            continue
        cx, cy = (xs * m).sum() / s, (ys * m).sum() / s
        dists.append(math.hypot(cx - sz / 2, cy - sz / 2) * km)
    dists = np.array(dists)
    if len(dists):
        print(f"\n=== Stage-1 attention-eye alignment (n={len(dists)}) ===")
        print(f"  centroid-to-eye: mean {dists.mean():.2f} km | median {np.median(dists):.2f} km")
    return dists


# ============================================================================
# Orchestration
# ============================================================================
RUN = {
    # Stage 1
    "data_prep":       True,
    "train_yolo":      True,
    "eval_stage1":     True,
    # Stage 2
    "strict_tracks":   True,
    "train_forecaster": True,
    "evaluate":        True,
    # XAI
    "perm_importance": True,
    "attn_rollout":    True,
    "integrated_grads": True,
    "stage1_eigencam": False,   # needs pytorch-grad-cam + weights
}


def build_datasets(track_csv=None):
    df = pd.read_csv(track_csv or CONFIG["tracks_csv"])
    tr = StrictTrackDataset(df[df["split"] == "train"], "train")
    va = StrictTrackDataset(df[df["split"] == "val"], "val", tr.wind_stats)
    te = StrictTrackDataset(df[df["split"] == "test"], "test", tr.wind_stats)
    print(f"sequences -> train {len(tr)} / val {len(va)} / test {len(te)}")
    return tr, va, te


def main():
    torch.manual_seed(CONFIG["seed"]); np.random.seed(CONFIG["seed"]); random.seed(CONFIG["seed"])

    if RUN["data_prep"]:
        materialize_yolo_dataset(build_frame_index())
    if RUN["train_yolo"]:
        train_yolo()
    if RUN["eval_stage1"]:
        evaluate_stage1()
    if RUN["strict_tracks"]:
        generate_strict_yolo_tracks()

    model = None
    if RUN["train_forecaster"]:
        tr, va, te = build_datasets()
        model, _ = train_forecaster(tr, va)
        torch.save(model.state_dict(), "forecaster_best.pt")
    else:
        tr, va, te = build_datasets()

    if any(RUN[k] for k in ("evaluate", "perm_importance", "attn_rollout", "integrated_grads")):
        if model is None:
            model = PhysicsAwareForecaster().to(T_CONFIG["device"])
            model.load_state_dict(torch.load("forecaster_best.pt", map_location=T_CONFIG["device"]))
        if RUN["evaluate"]:
            evaluate(model, te)
        if RUN["perm_importance"]:
            permutation_importance(model, te)
        if RUN["attn_rollout"]:
            attention_rollout(model, te)
        if RUN["integrated_grads"]:
            ig_case_studies(model, te)
    if RUN["stage1_eigencam"]:
        stage1_eigencam_alignment()


def _self_test():
    m = PhysicsAwareForecaster()
    n = count_parameters(m)
    print(f"[self-test] params = {n:,} (paper: 599,690)")
    assert n == 599690, n
    x = torch.randn(4, T_CONFIG["seq_len"], T_CONFIG["input_dim"])
    out = m(x, collect_attn=True)
    assert out.shape == (4, N_HORIZON, 2)
    loss = PhysicsAwareLoss(torch.ones(N_HORIZON, 2))
    l, parts = loss(out, torch.randn_like(out), torch.full((4,), 20.0))
    # adaptive box sanity
    assert adaptive_box_size_px({"long30": 0, "short30": 0, "wind": 90}) == 48
    assert CONFIG["box_min_px"] <= adaptive_box_size_px(
        {"long30": 120, "short30": 100, "wind": 0}) <= CONFIG["box_max_px"]
    print(f"[self-test] out {tuple(out.shape)} loss {float(l.detach()):.4f} parts {parts}")
    print("[self-test] OK")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        _self_test()
    else:
        main()
