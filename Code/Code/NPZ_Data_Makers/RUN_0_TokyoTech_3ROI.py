# RUN_0_TokyoTech_3ROI.py
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io
import cv2
from tqdm import tqdm

from roi_face_mediapipe_advanced import MediaPipeFaceMeshRoi
from rgb_extractor_multi_roi import extract_rgb_timeseries_multi_roi_from_video

# --- Global Settings ---
ROOT_DIR = Path("/media/data/rPPG/rPPG_Data/TokyoTechDataset")
OUT_DIR = Path("/media/data/rPPG/Code/GitHub/Project_rPPG/Dataset/3_ROI/TokyoTech_RAW")
ROI_NAMES = ["forehead_strip", "left_cheek_big", "right_cheek_big"]

FS_PPG = 2048.0
SYNC_OFFSET = 0.32  # timeStampVideo = timeStampcPPG - 0.32
FRAG_DUR = 20.0     # 20 seconds per AVI

def get_fragment_metadata(filename: str):
    """
    Maps filenames to index and state.
    0001 -> 1, 0601 -> 2, 1201 -> 3, etc.
    """
    try:
        start_frame = int(filename.split('to')[0])
        idx = (start_frame // 600) + 1
    except:
        return None, None

    if 1 <= idx <= 3:
        state = "Pre_Relax"
    elif 4 <= idx <= 6:
        state = "Exercise"
    elif 7 <= idx <= 9:
        state = "Post_Relax"
    else:
        state = "Unknown"
    
    return idx, state

def process_subject(subj_id: str):
    subj_dir = ROOT_DIR / subj_id
    video_dir = subj_dir / "30fps"
    ppg_mat_path = subj_dir / "contactPPG.mat"

    if not ppg_mat_path.exists():
        return

    # Load Master PPG (2048 Hz)
    mat_data = scipy.io.loadmat(str(ppg_mat_path))
    gt_full = mat_data['dataA'].flatten().astype(np.float64)
    t_gt_full = np.arange(len(gt_full)) / FS_PPG

    avi_files = sorted(list(video_dir.glob("*.avi")))
    roi_extractor = MediaPipeFaceMeshRoi()

    for avi_path in avi_files:
        frag_idx, state = get_fragment_metadata(avi_path.name)
        if frag_idx is None: 
            continue

        save_name = f"Subj_{subj_id}_{state}_Frag{frag_idx}"
        
        # Extract RGB (3-ROI Standard)
        # Frame_times_s=None builds internal t = i/30.0
        t_video, X = extract_rgb_timeseries_multi_roi_from_video(
            video_path=avi_path,
            frame_times_s=None,
            roi_extractor=roi_extractor,
            roi_names=ROI_NAMES,
            show_progress=False
        )

        # Synchronize and Interpolate
        # Video time 't' maps to PPG time 't + start_in_session + 0.32'
        v_start_in_session = (frag_idx - 1) * FRAG_DUR
        t_lookup = t_video + v_start_in_session + SYNC_OFFSET

        Y = np.interp(t_lookup, t_gt_full, gt_full, left=gt_full[0], right=gt_full[-1])

        meta = {
            "dataset": "TokyoTech",
            "subj": subj_id,
            "state": state,
            "frag": frag_idx,
            "roi_names": ROI_NAMES,
            "fs_video": 30.0,
            "fs_ppg_raw": FS_PPG,
            "sync_offset_s": SYNC_OFFSET
        }

        qc = {
            "n_frames": int(X.shape[0]),
            "v_offset_s": float(v_start_in_session),
            "ppg_lookup_start_s": float(t_lookup[0])
        }

        out_path = OUT_DIR / f"{save_name}.npz"
        np.savez_compressed(
            out_path,
            X=X.astype(np.float32),
            Y=Y.astype(np.float32),
            t=t_video.astype(np.float64),
            meta=json.dumps(meta),
            qc=json.dumps(qc)
        )

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subj_dirs = sorted([d.name for d in ROOT_DIR.iterdir() if d.is_dir() and d.name.isdigit()])

    for s_id in tqdm(subj_dirs, desc="Total Progress"):
        process_subject(s_id)

if __name__ == "__main__":
    main()