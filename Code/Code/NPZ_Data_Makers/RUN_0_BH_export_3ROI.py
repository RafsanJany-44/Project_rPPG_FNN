# RUN_0_BH_export_3ROI_best.py
# python RUN_0_BH_export_3ROI_best.py
#
# Best-possible BH NPZ export from available files:
# - X: [T, 3*K] multi-ROI RGB means per frame
# - Y: [T] GT waveform interpolated onto exact frame timestamps
# - t: [T] exact frame timestamps from timestamps.csv
#
# Notes:
# - We use real frame timestamps from timestamps.csv
# - wave.csv contains the GT waveform but no explicit GT timestamps in the sample files
# - Therefore, we reconstruct GT time as uniformly sampled over the measured recording duration
# - This is the strongest defensible alignment given the available BH files

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from roi_face_mediapipe_advanced import MediaPipeFaceMeshRoi
from rgb_extractor_multi_roi import _mean_rgb_robust_from_mask


# ---------------------------
# Paths / settings
# ---------------------------
ROOT_DIR = Path("/media/data/rPPG/rPPG_Data/Pub_BH-rPPG_FULL")
OUT_DIR = Path("/media/data/rPPG/Code/GitHub/Project_rPPG/Dataset/3_ROI/BH_RAW")
ROI_NAMES = ["forehead_strip", "left_cheek_big", "right_cheek_big"]


# ---------------------------
# Robust CSV readers
# ---------------------------
def read_single_col_numeric_csv(path: Path) -> np.ndarray:
    """
    Reads a one-column CSV robustly.
    Handles cases like:
      - no header
      - one text header
      - scientific notation
    Returns numeric values only.
    """
    df = pd.read_csv(path, header=None)
    vals = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    vals = vals[np.isfinite(vals)].astype(np.float64)
    if vals.size == 0:
        raise ValueError(f"No numeric data found in: {path}")
    return vals


def read_wave_csv(path: Path) -> np.ndarray:
    """
    wave.csv may have header 'Wave' or may be plain numeric.
    """
    try:
        df = pd.read_csv(path)
        if df.shape[1] >= 1:
            vals = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
            vals = vals[np.isfinite(vals)].astype(np.float64)
            if vals.size > 0:
                return vals
    except Exception:
        pass

    return read_single_col_numeric_csv(path)


def read_sensor_csv(path: Path) -> pd.DataFrame:
    """
    sensor.csv is optional for metadata/QC only.
    """
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


# ---------------------------
# BH time loading
# ---------------------------
def load_bh_timing_and_gt(seq_dir: Path):
    """
    Best-possible BH timing reconstruction from available files.

    Inputs:
      timestamps.csv : real frame timestamps (appears to be in milliseconds)
      wave.csv       : GT waveform values, no explicit GT timestamps in sample

    We do:
      1) load real frame timestamps
      2) convert ms -> s if needed
      3) normalize to start at 0
      4) reconstruct GT time uniformly over the measured recording duration

    Why:
      sample files do not expose explicit GT timestamps, so this is the
      strongest defensible reconstruction using real measured video time.
    """
    ts_path = seq_dir / "timestamps.csv"
    wave_path = seq_dir / "wave.csv"
    sensor_path = seq_dir / "sensor.csv"

    if not ts_path.exists():
        raise FileNotFoundError(f"Missing timestamps.csv: {ts_path}")
    if not wave_path.exists():
        raise FileNotFoundError(f"Missing wave.csv: {wave_path}")

    t_frame_raw = read_single_col_numeric_csv(ts_path)
    gt_wave = read_wave_csv(wave_path)
    sensor_df = read_sensor_csv(sensor_path) if sensor_path.exists() else pd.DataFrame()

    # timestamps.csv in your sample looks like milliseconds:
    # 0, 69.33, 131.09, ...
    # If very large, assume ms -> s.
    if np.nanmedian(np.diff(t_frame_raw)) > 1.0:
        t_frame_s = t_frame_raw / 1000.0
        ts_unit = "ms_to_s"
    else:
        t_frame_s = t_frame_raw.copy()
        ts_unit = "already_seconds"

    # normalize to 0
    t_frame_s = t_frame_s - t_frame_s[0]

    # basic cleanup
    good = np.isfinite(t_frame_s)
    t_frame_s = t_frame_s[good]

    if t_frame_s.size < 2:
        raise ValueError(f"Too few frame timestamps in: {ts_path}")

    # enforce strict monotonicity if tiny duplicates exist
    # keep first occurrence of each strictly increasing step
    keep = np.ones(len(t_frame_s), dtype=bool)
    keep[1:] = np.diff(t_frame_s) > 0
    t_frame_s = t_frame_s[keep]

    if t_frame_s.size < 2:
        raise ValueError(f"Frame timestamps are not usable after cleanup: {ts_path}")

    # Reconstruct GT time over measured recording duration.
    # This is equivalent to assuming wave.csv is uniformly sampled over the same acquisition span.
    dur_s = float(t_frame_s[-1] - t_frame_s[0])
    if gt_wave.size < 2 or dur_s <= 0:
        raise ValueError(f"Invalid GT waveform or duration in: {seq_dir}")

    t_gt_s = np.linspace(0.0, dur_s, gt_wave.size, dtype=np.float64)

    # effective GT sampling rate (for metadata/debug)
    fs_gt_eff = float((gt_wave.size - 1) / dur_s)

    info = {
        "ts_unit": ts_unit,
        "video_duration_s_measured": dur_s,
        "n_frame_timestamps": int(t_frame_s.size),
        "n_gt_samples_raw": int(gt_wave.size),
        "fs_gt_effective_hz": fs_gt_eff,
        "sensor_rows": int(len(sensor_df)) if not sensor_df.empty else 0,
    }

    return t_frame_s, t_gt_s, gt_wave.astype(np.float64), info, sensor_df


# ---------------------------
# ROI extraction from PNG frames
# ---------------------------
def extract_rgb_from_png_sequence(seq_dir: Path, roi_extractor, roi_names):
    img_dir = seq_dir / seq_dir.name
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing frame folder: {img_dir}")

    img_files = sorted(img_dir.glob("Frame_*.png"))
    if len(img_files) == 0:
        raise FileNotFoundError(f"No PNG frames found in: {img_dir}")

    X_list = []
    valid_names = []

    for img_path in img_files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        masks = roi_extractor.extract_masks_by_name(
            frame,
            fallback_full_image_on_fail=False
        )

        row = []
        for name in roi_names:
            m = masks.get(name, None)
            rgb = _mean_rgb_robust_from_mask(frame, m) if m is not None else np.zeros(3, dtype=np.float32)
            row.extend(rgb)

        X_list.append(row)
        valid_names.append(img_path.name)

    if len(X_list) == 0:
        raise ValueError(f"No readable PNG frames in: {img_dir}")

    X = np.asarray(X_list, dtype=np.float32)
    return X, valid_names


# ---------------------------
# QC
# ---------------------------
def compute_basic_qc(t_frame, t_gt, gt_wave, X, Y, valid_names, info):
    qc = {}

    qc["n_frames_rgb"] = int(X.shape[0])
    qc["n_valid_png"] = int(len(valid_names))
    qc["n_gt_samples"] = int(gt_wave.size)

    qc["t_frame_monotonic"] = bool(np.all(np.diff(t_frame) > 0)) if len(t_frame) >= 2 else False
    qc["t_gt_monotonic"] = bool(np.all(np.diff(t_gt) > 0)) if len(t_gt) >= 2 else False

    if len(t_frame) >= 2:
        dt = np.diff(t_frame)
        qc["t_frame_dt_median"] = float(np.median(dt))
        qc["t_frame_dt_std"] = float(np.std(dt))
        qc["fps_est_median"] = float(1.0 / np.median(dt)) if np.median(dt) > 0 else np.nan
        qc["video_duration_s"] = float(t_frame[-1] - t_frame[0])
    else:
        qc["t_frame_dt_median"] = np.nan
        qc["t_frame_dt_std"] = np.nan
        qc["fps_est_median"] = np.nan
        qc["video_duration_s"] = np.nan

    if len(t_gt) >= 2:
        qc["gt_duration_s"] = float(t_gt[-1] - t_gt[0])
    else:
        qc["gt_duration_s"] = np.nan

    if np.isfinite(qc["video_duration_s"]) and qc["video_duration_s"] > 0 and np.isfinite(qc["gt_duration_s"]):
        qc["duration_ratio_gt_over_video"] = float(qc["gt_duration_s"] / qc["video_duration_s"])
    else:
        qc["duration_ratio_gt_over_video"] = np.nan

    qc["X_rows"] = int(X.shape[0])
    qc["X_cols"] = int(X.shape[1]) if X.ndim == 2 else 0
    qc["Y_len"] = int(len(Y))
    qc["len_match"] = bool(len(t_frame) == len(Y) == X.shape[0])

    qc["gt_std_raw"] = float(np.std(gt_wave)) if gt_wave.size else np.nan
    qc["y_std_resampled"] = float(np.std(Y)) if Y.size else np.nan

    qc["ts_unit"] = info.get("ts_unit", "unknown")
    qc["fs_gt_effective_hz"] = info.get("fs_gt_effective_hz", np.nan)

    return qc


# ---------------------------
# Export one sequence
# ---------------------------
def export_one_sequence(seq_dir: Path, out_dir: Path, roi_names):
    seq_name = seq_dir.name

    # 1) load timing + GT
    t_frame_raw, t_gt, gt_wave, info, sensor_df = load_bh_timing_and_gt(seq_dir)

    # 2) extract RGB from PNGs
    roi_extractor = MediaPipeFaceMeshRoi()
    X, valid_names = extract_rgb_from_png_sequence(seq_dir, roi_extractor, roi_names)

    # 3) match lengths between readable PNG frames and timestamps
    # We assume Frame_00000.png corresponds to timestamps[0], etc.
    T = min(len(t_frame_raw), X.shape[0])
    if T < 2:
        raise ValueError(f"Too few aligned samples in: {seq_name}")

    t_frame = t_frame_raw[:T]
    X = X[:T, :]
    valid_names = valid_names[:T]

    # 4) resample GT onto exact frame times
    Y = np.interp(
        t_frame,
        t_gt,
        gt_wave,
        left=gt_wave[0],
        right=gt_wave[-1]
    ).astype(np.float32)

    # 5) meta + qc
    meta = {
        "dataset": "BH-rPPG",
        "seq": seq_name,
        "roi_names": roi_names,
        "channel_order": "RGB per ROI in roi_names order",
        "K": int(len(roi_names)),
        "C": int(X.shape[1]),
        "alignment_method": "real_frame_timestamps + duration_anchored_uniform_gt",
        "timestamps_source": "timestamps.csv",
        "gt_source": "wave.csv",
        "fs_gt_effective_hz": float(info["fs_gt_effective_hz"]),
        "sensor_has_columns": list(sensor_df.columns) if not sensor_df.empty else [],
    }

    qc = compute_basic_qc(
        t_frame=t_frame,
        t_gt=t_gt,
        gt_wave=gt_wave,
        X=X,
        Y=Y,
        valid_names=valid_names,
        info=info,
    )

    out_path = out_dir / f"{seq_name}.npz"
    np.savez_compressed(
        out_path,
        X=X.astype(np.float32),
        Y=Y.astype(np.float32),
        t=t_frame.astype(np.float64),
        meta=json.dumps(meta),
        qc=json.dumps(qc),
    )

    return qc


# ---------------------------
# Main
# ---------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted([p for p in ROOT_DIR.iterdir() if p.is_dir() and "_" in p.name])

    rows = []
    for seq_dir in tqdm(seq_dirs, desc="BH export", ncols=100):
        try:
            qc = export_one_sequence(seq_dir, OUT_DIR, ROI_NAMES)
            rows.append({"seq": seq_dir.name, "status": "OK", **qc})
            print(f"[OK]  {seq_dir.name}")
        except Exception as e:
            rows.append({"seq": seq_dir.name, "status": "FAIL", "error": str(e)})
            print(f"[FAIL] {seq_dir.name}: {e}")

    df = pd.DataFrame(rows)
    qc_csv = OUT_DIR / "bh_export_qc.csv"
    df.to_csv(qc_csv, index=False)

    ok = int((df["status"] == "OK").sum()) if "status" in df.columns else 0
    print(f"[DONE] BH export finished: OK={ok}/{len(df)}")
    print(f"[DONE] QC log saved: {qc_csv}")
    print(f"[DONE] NPZ files saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()