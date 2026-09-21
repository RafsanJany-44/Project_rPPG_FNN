# main/RUN_COHFACE_export_3ROI.py
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import cv2
from tqdm import tqdm

from roi_face_mediapipe_advanced import MediaPipeFaceMeshRoi
from rgb_extractor_multi_roi import extract_rgb_timeseries_multi_roi_from_video

def load_cohface_gt(hdf5_path: Path):
    """
    Extracts pulse signal from HDF5 container.
    BVP is sampled at 256 Hz.
    """
    with h5py.File(str(hdf5_path), 'r') as f:
        # Standard COHFACE structure: signals are in 'pulse'
        gt_sig = np.array(f['pulse'], dtype=np.float64).flatten()
        fs_gt = 256.0
        t_gt = np.arange(len(gt_sig)) / fs_gt
    return t_gt, gt_sig

def export_one_sequence(seq_dir: Path, out_dir: Path, roi_names: list):
    seq_name = seq_dir.name  # e.g., Subj_1_0
    
    # Identify video and hdf5 files
    # Format: data_1_0.avi and data_1_0.hdf5
    parts = seq_name.split('_')
    suffix = f"{parts[1]}_{parts[2]}"
    video_path = seq_dir / f"data_{suffix}.avi"
    hdf5_path = seq_dir / f"data_{suffix}.hdf5"

    if not video_path.exists() or not hdf5_path.exists():
        raise FileNotFoundError(f"Missing files in {seq_dir}")

    t_gt, gt_sig = load_cohface_gt(hdf5_path)
    
    roi_extractor = MediaPipeFaceMeshRoi()
    
    # COHFACE video is recorded at 20 fps
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if not np.isfinite(fps) or fps <= 1:
        fps = 20.0

    # Extract RGB
    # Passing None to frame_times_s builds t = i/fps internally
    t_frame, X = extract_rgb_timeseries_multi_roi_from_video(
        video_path=video_path,
        frame_times_s=None,
        roi_extractor=roi_extractor,
        roi_names=roi_names,
        show_progress=False
    )

    # Align GT to frame times
    Y = np.interp(t_frame, t_gt, gt_sig, left=gt_sig[0], right=gt_sig[-1])

    meta = {
        "dataset": "COHFACE",
        "seq": seq_name,
        "roi_names": roi_names,
        "K": len(roi_names),
        "C": int(X.shape[1]),
        "fps": fps
    }

    # Placeholder for existing QC logic
    qc = {"fs": fps, "duration_ratio_gt_over_video": (t_gt[-1]/t_frame[-1] if len(t_frame) else 1.0)}

    out_path = out_dir / f"{seq_name}.npz"
    np.savez_compressed(
        out_path,
        X=X.astype(np.float32),
        Y=Y.astype(np.float32),
        t=t_frame.astype(np.float64),
        meta=json.dumps(meta),
        qc=json.dumps(qc)
    )

def main():
    ROOT = Path("/media/data/rPPG/rPPG_Data/cohface_sorted")
    OUT = Path("/media/data/rPPG/Code/GitHub/Project_rPPG/Dataset/3_ROI/COHFACE_RAW")
    OUT.mkdir(parents=True, exist_ok=True)
    
    ROI_NAMES = ["forehead_strip", "left_cheek_big", "right_cheek_big"]
    seq_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("Subj_")])

    for seq_dir in tqdm(seq_dirs, desc="COHFACE export"):
        try:
            export_one_sequence(seq_dir, OUT, ROI_NAMES)
        except Exception as e:
            print(f"Error in {seq_dir.name}: {e}")

if __name__ == "__main__":
    main()