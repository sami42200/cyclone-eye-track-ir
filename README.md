# Physics-aware deep learning for tropical cyclone track prediction

Weakly supervised eye localisation and multi-horizon track forecasting from
**single-channel geostationary infrared imagery** — an end-to-end, two-stage
pipeline built on the Digital Typhoon V2 (Western Pacific) archive.

This repository is a clean, self-contained reference implementation of the
method described in the accompanying manuscript. Both stages share one
configuration, so geometry, storm filtering and the strict storm-level split are
identical across the whole pipeline.

---

## Overview

**Stage 1 — Weakly supervised eye localisation (YOLOv8n).**
Every Digital Typhoon frame is storm-centred, so the eye lies at the image
centre. Instead of a single fixed label, an *intensity-aware adaptive*
pseudo-label box is derived from the JMA 30-knot wind radii (tighter boxes for
stronger storms with smaller eyes). At inference, a deterministic *centre-aware
selection* rule keeps the predicted box closest to the image centre; if no box
clears the confidence threshold, the image centre is used as a flagged fallback.
Pixel offsets convert to lat/lon through the azimuthal-equidistant geometry at a
calibrated **4.88 km/pixel**.

**Stage 2 — Physics-aware track forecasting (Transformer).**
The frozen YOLOv8n is run once over every frame to build a **Strict Tracks**
dataset of `(φ̂, λ̂, confidence)` tuples — the forecaster never sees ground truth
at input time. From a 12-hour sliding window it builds a **13-feature kinematic
state vector** and a compact **pre-LayerNorm Transformer** forecasts storm-centre
offsets at five lead times `{+1, +3, +6, +9, +12} h`. A composite physics-aware
loss adds directional and climatological speed-sanity constraints to a
horizon-weighted distance term. Persistence and linear-extrapolation baselines
are reported alongside, with bootstrap 95% confidence intervals and
intensity/confidence stratification.

**Explainability.** Four quantitative analyses: permutation feature importance,
attention rollout, integrated gradients, and a Stage-1 EigenCAM attention-eye
alignment metric.

---

## Headline results (paper)

Stage 1 eye localisation (test set, 141 storms, 17,237 frames):

| | Median | Mean | 90th pct | 95th pct |
|---|---|---|---|---|
| Overall | 17.66 km | 19.48 km | 34.82 | 41.37 |
| Grade 3 (TS) | 22.24 | 23.34 | — | — |
| Grade 4 (STS) | 21.33 | 22.68 | — | — |
| Grade 5 (TY) | 12.37 | 14.46 | — | — |

Stage 2 track forecasting (mean Haversine error, km):

| Method | +1 h | +3 h | +6 h | +9 h | +12 h |
|---|---|---|---|---|---|
| Persistence | 25.76 | 57.50 | 111.72 | 167.42 | 223.88 |
| Linear extrap. | 31.99 | 64.41 | 117.59 | 172.93 | 229.46 |
| **Transformer (ours)** | **18.77** | **27.07** | **43.11** | **61.82** | **82.46** |

+12 h 95% CI `[81.38, 83.55]` km — a 63.2% reduction over persistence and 64.1%
over linear extrapolation. Typhoon-grade +12 h mean error: 69.17 km.

---

## Repository structure

```
.
├── cyclone_pipeline.py     # full end-to-end pipeline (Stage 1 + Stage 2 + XAI)
├── requirements.txt        # Python dependencies
└── README.md
```

`cyclone_pipeline.py` is one module with a clearly sectioned layout:
data prep → Stage-1 training/eval → Strict Tracks → Stage-2 training/eval → XAI.

---

## Dataset

**Digital Typhoon Dataset V2 — Western Pacific.**

- Official source (NII): <https://agora.ex.nii.ac.jp/digital-typhoon/>
- Kaggle mirror: `digital-typhoon-dataset-western-pacific-wp`
- 1,116 cyclones, 192,956 single-channel IR frames (1978–present), `512 × 512`,
  azimuthal-equidistant storm-centred crop, 1250 km radius (4.88 km/px).
- License: **CC BY 4.0**, doi `10.20783/DIAS.664` — cite when used.

Expected layout:

```
<image_root>/<storm_id>/<frame>.png
<metadata_root>/<storm_id>.csv     # year, month, day, hour, lat, lng, wind, grade, long30, short30, file_1
```

After the quality filter (JMA grade ∈ {3,4,5}, year ≥ 1987, ≥ 30 frames/storm)
and the strict storm-level **70/15/15** split (seed 42), the working corpus is
931 storms / 112,347 frames.

---

## Installation

```bash
git clone https://github.com/<your-username>/cyclone-eye-track-pipeline.git
cd cyclone-eye-track-pipeline
pip install -r requirements.txt
```

A CUDA GPU is recommended. On Kaggle most dependencies are pre-installed; you
typically only need `pip install grad-cam` for the EigenCAM analysis.

---

## Usage

Verify the model architecture (no dataset required) — confirms the exact
599,690-parameter Transformer and that the loss / attention hooks are wired:

```bash
python cyclone_pipeline.py --self-test
```

Run the pipeline. Point `CONFIG["image_root"]` / `CONFIG["metadata_root"]` at
your data, then:

```bash
python cyclone_pipeline.py
```

Choose which stages run via the `RUN` dictionary near the bottom of the file:

```python
RUN = {
    "data_prep":        True,   # Stage 1: build YOLO dataset (adaptive pseudo-labels)
    "train_yolo":       True,   # Stage 1: fine-tune YOLOv8n
    "eval_stage1":      True,   # Stage 1: full-test localisation error
    "strict_tracks":    True,   # Stage 2: build Strict Tracks
    "train_forecaster": True,   # Stage 2: train the physics-aware Transformer
    "evaluate":         True,   # Stage 2: baselines + bootstrap CIs + stratification
    "perm_importance":  True,   # XAI 4.2.1
    "attn_rollout":     True,   # XAI 4.2.2
    "integrated_grads": True,   # XAI 4.2.2
    "stage1_eigencam":  False,  # XAI 4.2.3 (needs pytorch-grad-cam + weights)
}
```

Individual functions are importable:

```python
from cyclone_pipeline import (
    build_frame_index, materialize_yolo_dataset, train_yolo, evaluate_stage1,
    generate_strict_yolo_tracks, build_datasets, train_forecaster, evaluate,
    permutation_importance, attention_rollout, ig_case_studies,
    stage1_eigencam_alignment, PhysicsAwareForecaster,
)
```

---

## Configuration

Shared geometry/filtering/split and Stage-1 training live in `CONFIG`; the
Stage-2 forecaster lives in `T_CONFIG`.

| `CONFIG` key | Default | Meaning |
|---|---|---|
| `km_per_pixel` | `1250/256 ≈ 4.88` | azimuthal-equidistant calibration |
| `nm_per_pixel` | `0.926` | adaptive-box scale (Δ_nm/px) |
| `grades_keep` / `year_min` / `min_frames` | `{3,4,5}` / `1987` / `30` | quality filter |
| `split_ratios` / `seed` | `(.70,.15,.15)` / `42` | strict storm-level split |
| `box_min_px` / `box_max_px` | `32` / `128` | adaptive-box clip range |
| `yolo_epochs` / `yolo_batch` / `yolo_lr0` | `50` / `32` / `1e-3` | Stage-1 training |
| `conf_thres` | `0.05` | detection threshold |

| `T_CONFIG` key | Default |
|---|---|
| `seq_len` / `horizons` | `12` / `{1,3,6,9,12}` |
| `input_dim` | `13` |
| `d_model` / `nhead` / `num_layers` / `dim_feedforward` | `128` / `8` / `3` / `512` |
| `epochs` / `batch_size` / `lr` / `weight_decay` | `60` / `256` / `5e-4` / `1e-4` |
| `alpha_dir` / `beta_speed` / `speed_cap_kmh` | `0.2` / `0.05` / `80` |

---

## Method details (mapped to the manuscript)

- **3.1.1 Adaptive pseudo-labels** — `adaptive_box_size_px()`:
  `b = clip(0.25 · R̄₃₀ / 0.926, 32, 128)` px; wind-speed fallback
  (48/56/64 px for ≥85/≥64/else kt).
- **3.1.2 Centre-aware selection** — box nearest the image centre; centre
  fallback when no detection clears `conf_thres`.
- **3.1.3 Geometry/training** — 4.88 km/px; YOLOv8n, 50 epochs, AdamW
  (lr 1e-3, wd 5e-4), 3-epoch warmup + cosine, batch 32, ±180° rotation and
  h/v flips, colour-jitter/mosaic/mixup/copy-paste off.
- **3.2.1 Strict Tracks** — `generate_strict_yolo_tracks()` runs the frozen
  detector once; persistence fallback; no ground truth at input time.
- **3.2.2 Feature vector** — exact 13-dim `x_t`: anchor offsets, velocity,
  acceleration, wind, wind tendency, confidence, diurnal sin/cos, Coriolis
  lat sin/cos; `T=12` window, `H={1,3,6,9,12}`, ±0.25 h cadence enforced.
- **3.2.3 Model + loss** — pre-LayerNorm encoder (`d=128, 8 heads, 3 layers,
  FF=512, GELU, dropout 0.1` → **599,690 params**), learnable positional
  encoding, anchor-step LayerNorm read-out; `L = L_dist + 0.2·L_dir +
  0.05·L_speed`. AdamW lr 5e-4, wd 1e-4, warmup+cosine over 60 epochs,
  batch 256, grad-clip 1.0.
- **3.3 Baselines/eval** — persistence, linear extrapolation, Haversine error,
  bootstrap 95% CIs (n=2000), JMA-grade + confidence-quintile stratification.
- **4.2.1–4.2.3 XAI** — permutation importance (5 repeats/feature), attention
  rollout (residual-corrected, renormalised, recency ratio), integrated
  gradients (best/median/worst), Stage-1 EigenCAM centroid-to-eye distance on
  `model[14]` (P3 fusion) with a 10% border mask.

---

## Reproducibility notes

This is a reference implementation of the **method**. To reproduce the exact
published figures you also need:

1. The Digital Typhoon V2 data at the configured paths.
2. A trained Stage-1 checkpoint (run `train_yolo`, or set `CONFIG["yolo_weights"]`
   to your `best.pt`). Pre-trained weights are not bundled here; add them to a
   release or a data DOI for full reproduction.
3. Two intentionally explicit design choices are documented in code: the
   horizon-weighting scheme inside `L_dist` and the integrated-gradients step
   count. Adjust them in `T_CONFIG` / `integrated_gradients()` if your run used
   different settings.

Exact numbers depend on the trained detector and data; the architecture itself
is verified bit-for-bit against the paper via `--self-test`.

---

## Citation

If you use this code, please cite the paper and the dataset:

```bibtex
@article{tc_physics_aware_2025,
  title  = {Physics-aware deep learning for tropical cyclone track prediction:
            weakly supervised eye localisation and trajectory forecasting from
            single-channel infrared imagery},
  journal = {Remote Sensing Letters},
  year   = {2025}
}

@misc{digital_typhoon_v2,
  title  = {Digital Typhoon: Long-term Satellite Image Dataset for the
            Spatio-Temporal Modeling of Tropical Cyclones},
  author = {Kitamoto, A. and Hwang, J. and Vuillod, B. and Gautier, L. and
            Tian, Y. and Clanuwat, T.},
  year   = {2023},
  note   = {National Institute of Informatics, doi:10.20783/DIAS.664},
  howpublished = {\url{https://agora.ex.nii.ac.jp/digital-typhoon/}}
}
```

---

## License

- **Code:** MIT.
- **Data:** Digital Typhoon V2 — CC BY 4.0 (NII), not covered by MIT.
- **YOLOv8 / Ultralytics:** AGPL-3.0 — review Ultralytics' terms before
  redistribution or commercial use.

## Acknowledgments

The Digital Typhoon Project (National Institute of Informatics, Japan); the
open-source Ultralytics (YOLOv8), PyTorch, and pytorch-grad-cam maintainers;
and the Kaggle research platform for compute.
