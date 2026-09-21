# main/RUN_UBFCPhys_3ROI.py
# python main/RUN_UBFCPhys_3ROI.py
#run in server 
# We export UBFC-Phys sequences for deep learning (Option A):
# - X: [T, 3*K] multi-ROI RGB means per frame (K=3 recommended)
# - Y: [T] GT BVP resampled to video frame times
# - t: [T] frame time (seconds) built from frame_idx / fps (handled by our shared extractor)
#
# Important:
# - We do NOT use POS/CHROM for pass/fail gating.
# - We export sequences unless a hard error occurs.
# - We save a QC CSV to disk.

import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from roi_face_mediapipe_advanced import MediaPipeFaceMeshRoi
from rgb_extractor_multi_roi import extract_rgb_timeseries_multi_roi_from_video


# ---------------------------
# Our UBFC-Phys GT loader (BVP CSV)
# ---------------------------
def load_bvp_csv(bvp_csv: Path) -> np.ndarray:
    """
    Our UBFC-Phys BVP files are one value per line.
    Our sampling rate is 64 Hz per dataset readme.
    """
    return np.loadtxt(str(bvp_csv), dtype=np.float64).reshape(-1)


def resample_to_frame_times(t_frame: np.ndarray, fs_gt: float, gt_sig: np.ndarray) -> np.ndarray:
    """
    Our Option 1 alignment:
    - We assume both streams start at t=0 within each task file.
    - We build GT time axis by sample_idx / fs_gt and interpolate onto t_frame.
    """
    n = int(gt_sig.size)
    t_gt = np.arange(n, dtype=np.float64) / float(fs_gt)

    good = np.isfinite(t_gt) & np.isfinite(gt_sig)
    t2 = t_gt[good]
    x2 = gt_sig[good]
    if t2.size < 2:
        return np.zeros_like(t_frame, dtype=np.float32)

    y = np.interp(t_frame, t2, x2, left=x2[0], right=x2[-1])
    return y.astype(np.float32)


# ---------------------------
# Our path helper (your proven naming)
# ---------------------------
def get_task_paths(subject_dir: Path, task: str):
    """
    task: "T1" or "T2" or "T3"
    Files:
      vid_s<id>_<task>.avi
      bvp_s<id>_<task>.csv
    """
    subject_dir = Path(subject_dir)
    sid = subject_dir.name  # e.g., "s1"
    vid = subject_dir / f"vid_{sid}_{task}.avi"
    bvp = subject_dir / f"bvp_{sid}_{task}.csv"

    if not vid.exists():
        raise FileNotFoundError(f"Our video is missing: {vid}")
    if not bvp.exists():
        raise FileNotFoundError(f"Our BVP is missing: {bvp}")
    return vid, bvp


# ---------------------------
# Our QC (NO POS/CHROM)
# ---------------------------
def compute_time_integrity_qc(t_frame: np.ndarray, fs_gt: float, gt_len: int):
    qc = {}

    qc["n_frames"] = int(len(t_frame))
    qc["video_duration_s"] = float(t_frame[-1] - t_frame[0]) if len(t_frame) >= 2 else np.nan

    gt_dur = float((gt_len - 1) / fs_gt) if gt_len >= 2 else np.nan
    qc["gt_duration_s"] = gt_dur
    qc["duration_ratio_gt_over_video"] = float(gt_dur / qc["video_duration_s"]) if np.isfinite(gt_dur) and np.isfinite(qc["video_duration_s"]) and qc["video_duration_s"] > 0 else np.nan

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

    return qc


def compute_gt_integrity_qc(gt_sig: np.ndarray, y_on_frames: np.ndarray):
    qc = {}

    gt = np.asarray(gt_sig, dtype=np.float64).reshape(-1)
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


def compute_roi_integrity_qc_from_X(X: np.ndarray, roi_names):
    """
    We estimate ROI integrity from X without re-reading video:
    - If a ROI often returns [0,0,0], it likely had empty mask / face failure.
    This is not perfect, but it is cheap and consistent.
    """
    qc = {}
    K = len(roi_names)
    if X is None or X.size == 0:
        qc["roi_zero_rgb_frac_est"] = np.nan
        return qc

    T = X.shape[0]
    for k, name in enumerate(roi_names):
        rgb = X[:, k*3:(k+1)*3]
        is_zero = np.all(np.isclose(rgb, 0.0), axis=1)
        qc[f"{name}_zero_rgb_frac_est"] = float(np.mean(is_zero)) if T > 0 else np.nan

    return qc


# ---------------------------
# Export one subject-task
# ---------------------------
def export_one_subject_task(
    subject_dir: Path,
    task: str,
    out_dir: Path,
    roi_names,
    fs_gt: float = 64.0,
    zscore_gt: bool = True,
):
    subject_dir = Path(subject_dir)
    subj = subject_dir.name

    video_path, bvp_path = get_task_paths(subject_dir, task)

    # Our GT BVP
    gt_bvp = load_bvp_csv(bvp_path).astype(np.float64)
    if zscore_gt and np.std(gt_bvp) > 1e-12:
        gt_bvp = (gt_bvp - np.mean(gt_bvp)) / np.std(gt_bvp)

    # Our multi-ROI RGB (shared extractor)
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


    # Our alignment: resample GT to frame times
    Y = resample_to_frame_times(t_frame, fs_gt=fs_gt, gt_sig=gt_bvp)

    # Our QC (no gating)
    qc_time = compute_time_integrity_qc(t_frame=t_frame, fs_gt=fs_gt, gt_len=int(gt_bvp.size))
    qc_gt = compute_gt_integrity_qc(gt_sig=gt_bvp, y_on_frames=Y)
    qc_roi = compute_roi_integrity_qc_from_X(X=X, roi_names=roi_names)
    qc = {**qc_time, **qc_gt, **qc_roi}

    meta = {
        "dataset": "UBFC-Phys",
        "subj": subj,
        "task": task,
        "video": str(video_path),
        "bvp": str(bvp_path),
        "roi_names": roi_names,
        "channel_order": "RGB per ROI in roi_names order",
        "K": int(len(roi_names)),
        "C": int(3 * len(roi_names)),
        "fs_gt": float(fs_gt),
        "zscore_gt": bool(zscore_gt),
    }

    out_path = out_dir / f"{subj}_{task}.npz"
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
    ROOT = Path(r"C:\Users\user\Desktop\New folder\raw")
    OUT = Path(r"C:\Users\user\Desktop\New folder\UBFCPhys_DL_SEQ_NPZ")
    OUT.mkdir(parents=True, exist_ok=True)

    ROI_NAMES = ["forehead_strip", "left_cheek_big", "right_cheek_big"]
    TASKS = ("T1", "T2", "T3")

    subject_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("s")])

    rows = []
    for sdir in subject_dirs:
        for task in TASKS:
            seq_id = f"{sdir.name}_{task}"
            try:
                qc = export_one_subject_task(
                    subject_dir=sdir,
                    task=task,
                    out_dir=OUT,
                    roi_names=ROI_NAMES,
                    fs_gt=64.0,
                    zscore_gt=True,
                )
                rows.append({"seq": seq_id, "status": "OK", **qc})
                print(f"[OK]  {seq_id}")
            except Exception as e:
                rows.append({"seq": seq_id, "status": "FAIL", "error": str(e)})
                print(f"[FAIL] {seq_id}: {e}")

    df = pd.DataFrame(rows)
    qc_csv = OUT / "ubfcphys_export_qc.csv"
    df.to_csv(qc_csv, index=False)

    ok = int((df["status"] == "OK").sum()) if "status" in df.columns else 0
    total = int(len(df))
    print(f"[DONE] our UBFC-Phys export finished: OK={ok}/{total}")
    print(f"[DONE] our QC log saved: {qc_csv}")
    print(f"[DONE] our sequence files saved in: {OUT}")


if __name__ == "__main__":
    main()
