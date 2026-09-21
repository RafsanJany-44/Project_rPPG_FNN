# main/RUN_0_UBFC_3ROI.py
# python RUN_0_UBFC_3ROI.py
#
# We export UBFC sequences for deep learning (Option A):
# - X: [T, 3*K] multi-ROI RGB means per frame (K=3)
# - Y: [T] GT waveform resampled to frame times (t = frame_idx / fps)
# - t: [T] frame time in seconds
#
# Important:
# - We do NOT use POS/CHROM for pass/fail gating.
# - We export sequences unless a hard error occurs.
# - We save a QC CSV to disk so we can analyze drift/quality later.

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from roi_face_mediapipe_advanced import MediaPipeFaceMeshRoi
from rgb_extractor_multi_roi import extract_rgb_timeseries_multi_roi_from_video



# ---------------------------
# ---------------------------
def load_gt_ground_truth_txt(path: Path):
    data = np.loadtxt(str(path), dtype=np.float64)

    if data.ndim == 1:
        trace = data
        t = np.arange(trace.size, dtype=np.float64) / 64.0
        hr = np.full_like(trace, np.nan)
        return t, trace, hr

    if data.shape[0] == 3 and data.shape[1] > 3:
        trace = data[0, :]
        hr = data[1, :]
        t = data[2, :]
        return t.astype(np.float64), trace.astype(np.float64), hr.astype(np.float64)

    if data.shape[1] == 3 and data.shape[0] > 3:
        trace = data[:, 0]
        hr = data[:, 1]
        t = data[:, 2]
        return t.astype(np.float64), trace.astype(np.float64), hr.astype(np.float64)

    flat = data.reshape(-1)
    trace = flat
    t = np.arange(trace.size, dtype=np.float64) / 64.0
    hr = np.full_like(trace, np.nan)
    return t, trace.astype(np.float64), hr


def load_gt_gtdump_xmp(path: Path):
    data = np.loadtxt(str(path), delimiter=",", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] >= 4:
        t = data[:, 0] / 1000.0
        hr = data[:, 1]
        trace = data[:, 3]
        return t.astype(np.float64), trace.astype(np.float64), hr.astype(np.float64)

    trace = data[:, -1]
    t = np.arange(trace.size, dtype=np.float64) / 64.0
    hr = np.full_like(trace, np.nan)
    return t, trace.astype(np.float64), hr


def load_ubfc_gt(seq_dir: Path):
    xmp = seq_dir / "gtdump.xmp"
    if xmp.exists():
        return load_gt_gtdump_xmp(xmp)

    txts = sorted(seq_dir.glob("ground_truth*.txt"))
    if len(txts) > 0:
        return load_gt_ground_truth_txt(txts[0])

    raise FileNotFoundError(f"Our GT file is missing in: {seq_dir}")


def find_video_file(seq_dir: Path):
    avis = sorted(seq_dir.glob("*.avi"))
    if len(avis) == 0:
        raise FileNotFoundError(f"Our video is missing in: {seq_dir}")
    return avis[0]


# ---------------------------
# Our alignment: resample GT onto frame time
# ---------------------------
def resample_gt_to_frame_times(t_frame: np.ndarray, t_gt: np.ndarray, gt_trace: np.ndarray):
    order = np.argsort(t_gt)
    t2 = t_gt[order].astype(np.float64)
    x2 = gt_trace[order].astype(np.float64)

    good = np.isfinite(t2) & np.isfinite(x2)
    t2 = t2[good]
    x2 = x2[good]
    if t2.size < 2:
        return np.zeros_like(t_frame, dtype=np.float32)

    y = np.interp(t_frame, t2, x2, left=x2[0], right=x2[-1])
    return y.astype(np.float32)


# ---------------------------
# Our QC (NO POS/CHROM gating)
# ---------------------------
def compute_time_integrity_qc(t_frame: np.ndarray, t_gt: np.ndarray, fps: float):
    qc = {}

    qc["n_frames"] = int(len(t_frame))
    qc["n_gt_samples"] = int(len(t_gt))
    qc["fps_from_video"] = float(fps)

    # frame dt stability
    if len(t_frame) >= 3:
        dt = np.diff(t_frame)
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

    # duration match (drift proxy)
    vid_dur = float(t_frame[-1] - t_frame[0]) if len(t_frame) >= 2 else np.nan
    gt_dur = float(t_gt[-1] - t_gt[0]) if len(t_gt) >= 2 else np.nan
    qc["video_duration_s"] = vid_dur
    qc["gt_duration_s"] = gt_dur
    qc["duration_ratio_gt_over_video"] = float(gt_dur / vid_dur) if np.isfinite(vid_dur) and vid_dur > 0 and np.isfinite(gt_dur) else np.nan

    return qc


def compute_gt_integrity_qc(gt_trace: np.ndarray, y_on_frames: np.ndarray):
    qc = {}

    gt = np.asarray(gt_trace, dtype=np.float64).reshape(-1)
    y = np.asarray(y_on_frames, dtype=np.float64).reshape(-1)

    qc["gt_nan_frac"] = float(np.mean(~np.isfinite(gt))) if gt.size else 1.0
    qc["y_nan_frac"] = float(np.mean(~np.isfinite(y))) if y.size else 1.0

    gt_fin = gt[np.isfinite(gt)]
    y_fin = y[np.isfinite(y)]

    qc["gt_std"] = float(np.std(gt_fin)) if gt_fin.size else np.nan
    qc["y_std"] = float(np.std(y_fin)) if y_fin.size else np.nan

    if y_fin.size >= 3:
        dy = np.diff(y_fin)
        mad = np.median(np.abs(dy - np.median(dy))) + 1e-8
        jump = np.abs(dy) > (10.0 * mad)
        qc["y_jump_frac"] = float(np.mean(jump))
    else:
        qc["y_jump_frac"] = np.nan

    return qc


def compute_roi_integrity_qc(roi_extractor: MediaPipeFaceMeshRoi, video_path: Path, roi_names, max_check_frames: int = 300):
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

        # We prefer no-full-image fallback for QC (we want to know failures).
        masks = roi_extractor.extract_masks_by_name(frame_bgr, fallback_full_image_on_fail=False)

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
    qc = {"roi_check_frames": int(i), "face_fail_frac_est": float(face_fail / i)}

    for k, name in enumerate(roi_names):
        c = pix_counts[:, k].astype(np.float64)
        qc[f"{name}_pix_med"] = float(np.median(c))
        qc[f"{name}_pix_min"] = float(np.min(c))
        qc[f"{name}_pix_zero_frac"] = float(np.mean(c <= 0))

    return qc


# ---------------------------
# Export one UBFC sequence
# ---------------------------
def export_one_sequence(seq_dir: Path, out_dir: Path, roi_names, zscore_gt):
    seq_dir = Path(seq_dir)
    seq_name = seq_dir.name

    video_path = find_video_file(seq_dir)
    t_gt, gt_trace, _ = load_ubfc_gt(seq_dir)

    gt_trace = gt_trace.astype(np.float64)
    if zscore_gt and np.std(gt_trace) > 1e-12:
        gt_trace = (gt_trace - np.mean(gt_trace)) / np.std(gt_trace)

    roi = MediaPipeFaceMeshRoi()

    # 1) get fps from video
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if not np.isfinite(fps) or fps <= 1:
        fps = 30.0

    # 2) extract RGB (PURE-style extractor returns (t_arr, X))
    # we pass dummy times just to read the full video length
    # (our extractor reads frames sequentially and stops naturally)
    dummy_times = np.arange(10**9, dtype=np.float64)  # large; loop breaks at end of video
    t_dummy, X = extract_rgb_timeseries_multi_roi_from_video(
        video_path=video_path,
        frame_times_s=dummy_times,
        roi_extractor=roi,
        roi_names=roi_names,
        show_progress=True,
    )

    # 3) build t_frame from length(X)
    T = int(X.shape[0])
    t_frame = (np.arange(T, dtype=np.float64) / fps)

    # Our aligned target
    Y = resample_gt_to_frame_times(t_frame, t_gt, gt_trace)

    # Our QC (we do not gate)
    qc_time = compute_time_integrity_qc(t_frame=t_frame, t_gt=t_gt, fps=fps)
    qc_gt = compute_gt_integrity_qc(gt_trace=gt_trace, y_on_frames=Y)
    qc_roi = compute_roi_integrity_qc(roi, video_path, roi_names, max_check_frames=300)

    qc = {**qc_time, **qc_gt, **qc_roi}

    meta = {
        "dataset": "UBFC",
        "seq": seq_name,
        "video": str(video_path),
        "roi_names": roi_names,
        "channel_order": "RGB per ROI in roi_names order",
        "K": len(roi_names),
        "C": int(3 * len(roi_names)),
        "fps": float(fps),
        "zscore_gt": bool(zscore_gt),
    }

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


def main():
    ROOT = Path(r"D:\Data\UBFC\Dataset_3")
    OUT = Path(r"C:\Users\user\Documents\GitHub\Project_rPPG\Dataset\3_ROI\UBFC_RAW")
    OUT.mkdir(parents=True, exist_ok=True)

    ROI_NAMES = ["forehead_strip", "left_cheek_big", "right_cheek_big"]

    seq_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()])

    rows = []
    for seq_dir in tqdm(seq_dirs, desc="UBFC export", ncols=100):
        try:
            qc = export_one_sequence(seq_dir, OUT, ROI_NAMES, zscore_gt=False)
            rows.append({"seq": seq_dir.name, "status": "OK", **qc})
            print(f"[OK]  {seq_dir.name}.....Do not Worry about the Progress Bar! I promise its all fine.")
        except Exception as e:
            rows.append({"seq": seq_dir.name, "status": "FAIL", "error": str(e)})
            print(f"[FAIL] {seq_dir.name}: {e}")

    df = pd.DataFrame(rows)
    qc_csv = OUT / "ubfc_export_qc.csv"
    df.to_csv(qc_csv, index=False)

    ok = int((df["status"] == "OK").sum()) if "status" in df.columns else 0
    print(f"[DONE] our UBFC export finished: OK={ok}/{len(seq_dirs)}")
    print(f"[DONE] our QC log saved: {qc_csv}")
    print(f"[DONE] our sequence files saved in: {OUT}")


if __name__ == "__main__":
    main()
