import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Project internal logic for ROI extraction
from roi_face_mediapipe_advanced import MediaPipeFaceMeshRoi
from rgb_extractor_multi_roi import _mean_rgb_robust_from_mask


# --- Configuration ubfc ---
# NPZ_PATH = Path("/media/data/rPPG/Code/GitHub/Project_rPPG/Dataset/3_ROI/UBFCPhys_RAW/s1_T1.npz")
# DATA_PATH = Path("/media/data/rPPG/rPPG_Data/UBFC-PHYS/s1/vid_s1_T1.avi")


# --- Configuration tokyotech ---
# NPZ_PATH = Path("/media/data/rPPG/Code/GitHub/Project_rPPG/Dataset/3_ROI/TokyoTech_RAW/Subj_01_Pre_Relax_Frag1.npz")
# DATA_PATH = Path("/media/data/rPPG/rPPG_Data/TokyoTechDataset/01/30fps/0001to0600.avi")


# --- Configuration BH ---
NPZ_PATH = Path("/media/data/rPPG/Code/GitHub/Project_rPPG/Dataset/3_ROI/BH_RAW/0_0.npz")
DATA_PATH = Path("/media/data/rPPG/rPPG_Data/Pub_BH-rPPG_FULL/0_0/0_0")


# --- Configuration COHFACE ---
#NPZ_PATH = Path("/media/data/rPPG/Code/GitHub/Project_rPPG/Dataset/3_ROI/COHFACE_RAW/Subj_1_0.npz")
#DATA_PATH = Path("/media/data/rPPG/rPPG_Data/cohface_sorted/Subj_1_0/data_1_0.avi")



ROI_NAMES = ["forehead_strip", "left_cheek_big", "right_cheek_big"]
ROI_DISPLAY_NAMES = ["Forehead", "Left Cheek", "Right Cheek"]
LINE_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR

CHANNEL_NAMES = ["R", "G", "B"]

# Output folder for HTML plots
OUT_DIR = Path("./comparison_plotly_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_comparison_engine():
    """
    Executes comparison between stored NPZ data and live frame extraction.
    Supports both video files and image sequences.
    """
    # Load Stored Data
    with np.load(NPZ_PATH, allow_pickle=True) as data:
        X_npz = data["X"]
        t_npz = data["t"]
        meta = json.loads(str(data["meta"]))

    # Initialize Live Extraction
    roi_extractor = MediaPipeFaceMeshRoi()

    # Determine if input is a video file or an image folder
    is_sequence = DATA_PATH.is_dir()

    if is_sequence:
        frame_files = sorted(list(DATA_PATH.glob("*.png")))
        total_source_frames = len(frame_files)
    else:
        cap = cv2.VideoCapture(str(DATA_PATH))
        total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    X_live = []
    frame_idx = 0
    total_npz_frames = int(X_npz.shape[0])

    # Process until either source ends or NPZ length is reached
    max_frames = min(total_source_frames, total_npz_frames)

    print(f"Engine Status: Comparing {NPZ_PATH.name}")
    print("Controls: Press [Space] to Pause | Press [q] to Stop and Analyze.")

    paused = False
    while frame_idx < max_frames:
        if not paused:
            # Frame acquisition logic
            if is_sequence:
                frame = cv2.imread(str(frame_files[frame_idx]))
                ok = frame is not None
            else:
                ok, frame = cap.read()

            if not ok:
                break

            display_frame = frame.copy()
            masks = roi_extractor.extract_masks_by_name(frame)

            current_row = []
            for i, name in enumerate(ROI_NAMES):
                m = masks.get(name)

                # Extract robust mean RGB
                rgb = _mean_rgb_robust_from_mask(frame, m) if m is not None else np.zeros(3, dtype=np.float32)
                current_row.extend(rgb)

                # Draw ROI boundaries
                if m is not None:
                    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(display_frame, contours, -1, LINE_COLORS[i], 1)

            X_live.append(current_row)

            # Overlay frame information
            cv2.putText(
                display_frame,
                f"Frame: {frame_idx}/{max_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Comparison Engine: Frame-by-Frame Validation", display_frame)

        # Monitor for interaction
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            print(f"Extraction stopped at frame {frame_idx}. Analyzing available data.")
            break
        elif key == ord(" "):
            paused = not paused

        if not paused:
            frame_idx += 1

    if not is_sequence:
        cap.release()
    cv2.destroyAllWindows()

    X_live = np.array(X_live, dtype=np.float32)
    processed_len = len(X_live)

    if processed_len == 0:
        print("No frames were processed.")
        return

    # Align stored data for statistical comparison
    X_npz_subset = X_npz[:processed_len]
    t_subset = t_npz[:processed_len]

    # Original decision metric kept exactly as before:
    # Forehead Green channel = index 1
    corr, _ = pearsonr(X_live[:, 1], X_npz_subset[:, 1])
    rmse = np.sqrt(np.mean((X_live[:, 1] - X_npz_subset[:, 1]) ** 2))

    final_status = "PASS" if corr > 0.99 and rmse < 0.1 else "FAIL"

    print("\n" + "=" * 50)
    print("DATA INTEGRITY REPORT")
    print("=" * 50)
    print(f"Processed Frames: {processed_len}")
    print(f"Pearson Correlation (Forehead G): {corr:.6f}")
    print(f"RMSE (Forehead G): {rmse:.6f}")
    print(f"Decision: {final_status}")
    print("=" * 50)

    # Extra per-channel quick report
    print("\nPer-channel summary:")
    for roi_idx, roi_name in enumerate(ROI_DISPLAY_NAMES):
        for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
            col = roi_idx * 3 + ch_idx
            x1 = X_live[:, col]
            x2 = X_npz_subset[:, col]

            if np.std(x1) < 1e-12 or np.std(x2) < 1e-12:
                ch_corr = np.nan
            else:
                ch_corr, _ = pearsonr(x1, x2)

            ch_rmse = np.sqrt(np.mean((x1 - x2) ** 2))
            print(f"{roi_name:12s} {ch_name}: corr={ch_corr:.6f}, rmse={ch_rmse:.6f}")

    # Keep existing matplotlib-style plots
    generate_validation_plots_matplotlib(X_npz_subset, X_live, t_subset, corr, rmse)

    # New Plotly HTML exports
    generate_validation_plots_plotly(X_npz_subset, X_live, t_subset, corr, rmse)


def generate_validation_plots_matplotlib(X_npz, X_live, t, corr, rmse):
    """
    Original-style matplotlib verification plot
    (kept for compatibility).
    """
    plt.figure(figsize=(14, 8))

    # Signal Comparison (Forehead Green Channel)
    plt.subplot(2, 1, 1)
    plt.plot(t, X_npz[:, 1], label="Stored Signal", color="black", alpha=0.6)
    plt.plot(t, X_live[:, 1], label="Live Re-extracted Signal", color="red", linestyle="--")
    plt.title(f"Temporal Phase Check (Forehead G) | Corr: {corr:.4f}")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)

    # Absolute Residual Error
    plt.subplot(2, 1, 2)
    error = X_npz[:, 1] - X_live[:, 1]
    plt.fill_between(t, error, color="purple", alpha=0.2, label="Residual Delta")
    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"RMSE Consistency Check (Forehead G): {rmse:.4f}")
    plt.ylabel("Delta")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def generate_validation_plots_plotly(X_npz, X_live, t, corr, rmse):
    """
    Creates one single interactive Plotly HTML plot:
    - no subplots
    - no dashed lines
    - all 3 ROIs
    - all RGB channels
    - Stored and Live both shown as normal solid lines
    """

    base_name = NPZ_PATH.stem

    fig = go.Figure()

    for roi_idx, roi_name in enumerate(ROI_DISPLAY_NAMES):
        for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
            col_idx = roi_idx * 3 + ch_idx

            # Stored signal
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=X_npz[:, col_idx],
                    mode="lines",
                    name=f"{roi_name} {ch_name} Stored",
                )
            )

            # Live signal
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=X_live[:, col_idx],
                    mode="lines",
                    name=f"{roi_name} {ch_name} Live",
                )
            )

    fig.update_layout(
        title=f"Stored vs Live RGB Signals | All 3 ROIs | {base_name}",
        xaxis_title="Time (s)",
        yaxis_title="Intensity",
        height=900,
        width=1600,
        template="plotly_white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02
        )
    )

    out_html = OUT_DIR / f"{base_name}_single_all_roi_rgb_live_vs_stored.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"[Saved] {out_html}")


if __name__ == "__main__":
    run_comparison_engine()