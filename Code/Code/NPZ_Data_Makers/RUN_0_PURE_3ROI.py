# main/RUN_PURE_3ROI.py
# python RUN_PURE_3ROI.py

import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from roi_face_mediapipe_advanced import MediaPipeFaceMeshRoi
from rgb_extractor_multi_roi import extract_rgb_timeseries_multi_roi_from_video


# ---------------------------
# Our PURE JSON parsing
# ---------------------------
def load_pure_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    full = data.get("/FullPackage", [])
    img = data.get("/Image", [])

    img_ts_ns = np.array([x["Timestamp"] for x in img], dtype=np.int64)

    gt_ts_ns = []
    gt_wave = []
    for x in full:
        ts = x.get("Timestamp", None)
        val = x.get("Value", {})
        if ts is None:
            continue
        if "waveform" not in val:
            continue
        gt_ts_ns.append(ts)
        gt_wave.append(val.get("waveform", 0))

    gt_ts_ns = np.array(gt_ts_ns, dtype=np.int64)
    gt_wave = np.array(gt_wave, dtype=np.float64)
    return img_ts_ns, gt_ts_ns, gt_wave


def ns_to_rel_seconds(ts_ns: np.ndarray) -> np.ndarray:
    ts_ns = ts_ns.astype(np.int64)
    return (ts_ns - ts_ns[0]) * 1e-9


def resample_gt_to_frame_times(t_frame_s: np.ndarray, t_gt_s: np.ndarray, gt_wave: np.ndarray):
    if t_frame_s.size == 0 or t_gt_s.size == 0:
        return np.zeros_like(t_frame_s, dtype=np.float32)

    order = np.argsort(t_gt_s)
    t_gt_s2 = t_gt_s[order]
    gt2 = gt_wave[order].astype(np.float64)

    y = np.interp(t_frame_s, t_gt_s2, gt2, left=gt2[0], right=gt2[-1])
    return y.astype(np.float32)


# ---------------------------
# Our QC (NO POS/CHROM gating)
# ---------------------------
def _safe_stats_1d(x: np.ndarray):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan)
    return dict(mean=float(np.mean(x)), std=float(np.std(x)), min=float(np.min(x)), max=float(np.max(x)))


def compute_time_integrity_qc(t_img_s: np.ndarray, t_frame_s: np.ndarray, t_gt_s: np.ndarray):
    """
    We measure alignment integrity without using rPPG algorithms:
    - Are timestamps monotonic?
    - Are frame time steps stable?
    - Does GT duration roughly match video duration?
    """
    qc = {}

    qc["n_img_ts"] = int(len(t_img_s))
    qc["n_frames_extracted"] = int(len(t_frame_s))
    qc["n_gt_ts"] = int(len(t_gt_s))

    # frame time stats
    if len(t_frame_s) >= 3:
        dt = np.diff(t_frame_s)
        dt_pos = dt[dt > 0]
        qc["t_frame_monotonic"] = bool(np.all(dt > 0))
        qc["t_frame_dt_median"] = float(np.median(dt_pos)) if dt_pos.size else np.nan
        qc["t_frame_dt_std"] = float(np.std(dt_pos)) if dt_pos.size else np.nan
        qc["fps_est_median"] = float(1.0 / qc["t_frame_dt_median"]) if qc["t_frame_dt_median"] and np.isfinite(qc["t_frame_dt_median"]) else np.nan
    else:
        qc["t_frame_monotonic"] = False
        qc["t_frame_dt_median"] = np.nan
        qc["t_frame_dt_std"] = np.nan
        qc["fps_est_median"] = np.nan

    # durations
    vid_dur = float(t_frame_s[-1] - t_frame_s[0]) if len(t_frame_s) >= 2 else np.nan
    gt_dur  = float(t_gt_s[-1] - t_gt_s[0]) if len(t_gt_s) >= 2 else np.nan
    qc["video_duration_s"] = vid_dur
    qc["gt_duration_s"] = gt_dur
    qc["duration_ratio_gt_over_video"] = float(gt_dur / vid_dur) if np.isfinite(vid_dur) and vid_dur > 0 and np.isfinite(gt_dur) else np.nan

    # mismatch between advertised image timestamps and extracted frames
    qc["frame_count_mismatch"] = int(len(t_img_s) - len(t_frame_s))

    return qc


def compute_gt_integrity_qc(gt_wave: np.ndarray, y_on_frames: np.ndarray):
    """
    We measure GT quality (sensor signal sanity), not vision:
    - NaN rate
    - variance / flatline
    - large jumps ratio (basic artifact indicator)
    """
    qc = {}

    gt = np.asarray(gt_wave, dtype=np.float64).reshape(-1)
    y = np.asarray(y_on_frames, dtype=np.float64).reshape(-1)

    qc["gt_nan_frac"] = float(np.mean(~np.isfinite(gt))) if gt.size else 1.0
    qc["y_nan_frac"] = float(np.mean(~np.isfinite(y))) if y.size else 1.0

    gt_fin = gt[np.isfinite(gt)]
    y_fin = y[np.isfinite(y)]

    qc["gt_std"] = float(np.std(gt_fin)) if gt_fin.size else np.nan
    qc["y_std"] = float(np.std(y_fin)) if y_fin.size else np.nan

    # jump ratio on resampled Y (robust-ish)
    if y_fin.size >= 3:
        dy = np.diff(y_fin)
        mad = np.median(np.abs(dy - np.median(dy))) + 1e-8
        jump = np.abs(dy) > (10.0 * mad)  # conservative
        qc["y_jump_frac"] = float(np.mean(jump))
    else:
        qc["y_jump_frac"] = np.nan

    # basic range stats (for debugging only)
    s = _safe_stats_1d(y)
    qc["y_min"] = s["min"]
    qc["y_max"] = s["max"]

    return qc


def compute_roi_integrity_qc(roi_extractor: MediaPipeFaceMeshRoi, video_path: Path, roi_names, max_check_frames: int = 300):
    """
    We measure whether our ROIs exist (pixel coverage), without judging pulse quality.
    We only scan the first N frames for speed.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"roi_qc_error": f"cannot_open_video:{video_path}"}

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(max_check_frames, n_total if n_total > 0 else max_check_frames)

    K = len(roi_names)
    pix_counts = np.zeros((n, K), dtype=np.int64)
    face_fail = 0

    i = 0
    for _ in range(n):
        ok, frame_bgr = cap.read()
        if not ok:
            break

        masks = roi_extractor.extract_masks_by_name(frame_bgr, fallback_full_image_on_fail=False)
        # if all masks are empty, likely face fail
        any_nonzero = False

        for k, name in enumerate(roi_names):
            m = masks.get(name, None)
            c = int(np.count_nonzero(m)) if m is not None else 0
            pix_counts[i, k] = c
            if c > 0:
                any_nonzero = True

        if not any_nonzero:
            face_fail += 1

        i += 1

    cap.release()

    if i == 0:
        return {"roi_qc_error": "no_frames_read"}

    pix_counts = pix_counts[:i, :]
    qc = {
        "roi_check_frames": int(i),
        "face_fail_frac_est": float(face_fail / i),
    }

    for k, name in enumerate(roi_names):
        c = pix_counts[:, k].astype(np.float64)
        qc[f"{name}_pix_med"] = float(np.median(c))
        qc[f"{name}_pix_min"] = float(np.min(c))
        qc[f"{name}_pix_zero_frac"] = float(np.mean(c <= 0))

    return qc


# ---------------------------
# Export
# ---------------------------
def export_one_sequence(seq_dir: Path, out_dir: Path, roi_names, save_npz: bool = True):
    seq_name = seq_dir.name
    parent = seq_dir.parent
    json_path = parent / f"{seq_name}.json"
    video_path = seq_dir / f"{seq_name}.avi"

    if not json_path.exists():
        raise FileNotFoundError(f"Missing json: {json_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")

    img_ts_ns, gt_ts_ns, gt_wave = load_pure_json(json_path)
    t_img_s = ns_to_rel_seconds(img_ts_ns)
    t_gt_s  = ns_to_rel_seconds(gt_ts_ns)

    roi = MediaPipeFaceMeshRoi()

    # Multi-ROI RGB extraction (frame times from /Image timestamps)
    t_frame_s, X = extract_rgb_timeseries_multi_roi_from_video(
        video_path=video_path,
        frame_times_s=t_img_s,
        roi_extractor=roi,
        roi_names=roi_names,
        show_progress=False,
    )

    # Resample GT to those exact frame times
    Y = resample_gt_to_frame_times(t_frame_s, t_gt_s, gt_wave)

    # QC (safe)
    qc_time = compute_time_integrity_qc(t_img_s=t_img_s, t_frame_s=t_frame_s, t_gt_s=t_gt_s)
    qc_gt   = compute_gt_integrity_qc(gt_wave=gt_wave, y_on_frames=Y)

    # ROI QC (fast scan only, does not gate)
    qc_roi  = compute_roi_integrity_qc(roi, video_path, roi_names, max_check_frames=300)

    qc = {**qc_time, **qc_gt, **qc_roi}

    meta = {
        "dataset": "PURE",
        "seq": seq_name,
        "roi_names": roi_names,
        "channel_order": "RGB per ROI in roi_names order",
        "K": len(roi_names),
        "C": int(3 * len(roi_names)),
    }

    if save_npz:
        out_path = out_dir / f"{seq_name}.npz"
        np.savez_compressed(
            out_path,
            X=X.astype(np.float32),
            Y=Y.astype(np.float32),
            t=t_frame_s.astype(np.float64),
            meta=json.dumps(meta),
            qc=json.dumps(qc),
        )

    return qc


def main():
    ROOT = Path(r"C:\Users\user\Desktop\New folder\raw")
    OUT  = Path(r"C:\Users\user\Desktop\New folder\PURE_DL_SEQ_NPZ")
    OUT.mkdir(parents=True, exist_ok=True)

    ROI_NAMES = ["forehead_strip", "left_cheek_big", "right_cheek_big"]

    seq_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir() and "-" in p.name])

    rows = []
    for seq_dir in tqdm(seq_dirs, desc="PURE export", ncols=100):
        try:
            qc = export_one_sequence(seq_dir, OUT, ROI_NAMES, save_npz=True)
            row = {"seq": seq_dir.name, "status": "OK", **qc}
        except Exception as e:
            row = {"seq": seq_dir.name, "status": "FAIL", "error": str(e)}
        rows.append(row)

    df = pd.DataFrame(rows)
    qc_csv = OUT / "pure_export_qc.csv"
    df.to_csv(qc_csv, index=False)

    ok = int((df["status"] == "OK").sum()) if "status" in df.columns else 0
    print(f"[DONE] our PURE export finished: OK={ok}/{len(seq_dirs)}")
    print(f"[DONE] our QC log saved: {qc_csv}")
    print(f"[DONE] our sequence files saved in: {OUT}")


if __name__ == "__main__":
    # We import cv2 only here to keep top clean in some environments
    import cv2
    main()
