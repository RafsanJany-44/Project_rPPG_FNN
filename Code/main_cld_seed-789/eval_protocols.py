"""
eval_protocols.py — rPPG evaluation across multiple measurement protocols
─────────────────────────────────────────────────────────────────────────

Purpose
    Run the SAME trained checkpoint through several evaluation "protocols"
    (measurement recipes) without retraining.  The model output waveform is
    identical for every protocol; only the post-processing differs:
    windowing, FFT resolution, bandpass, harmonic handling, and aggregation.

Protocols
    "old"     Old Approach          — 8s windows, 240-pt FFT (7.5 BPM bins),
                                       0.67-3.0 Hz, per-window aggregation.
                                       Reproduces the original eval.py behaviour.

    "prism"   PRISM Protocol        — 10s non-overlapping windows,
                                       16384-pt zero-padded FFT (~0.11 BPM),
                                       0.75-2.5 Hz, per-window aggregation.
                                       NOTE: this is ONLY the PRISM evaluation
                                       protocol (Steps 6-10 from the paper),
                                       NOT the PRISM algorithm (no dual-band,
                                       no alpha/lambda search, no harmonic check).

    "toolbox" rPPG-Toolbox Protocol — full-video waveform (non-overlapping
                                       windows concatenated), zero-padded FFT,
                                       0.75-2.5 Hz, one HR per video,
                                       per-video aggregation.

    "all"     Runs old, prism, and toolbox in sequence, each into its own
              output folder.

Configuration
    Set the module-level PROTOCOL global below.  It is NOT a command-line arg.

Output
    Each protocol writes to its own folder:
        SAVE_DIR / "EVAL_PROTOCOL_old"
        SAVE_DIR / "EVAL_PROTOCOL_prism"
        SAVE_DIR / "EVAL_PROTOCOL_toolbox"
    Every table, CSV, and plot produced by the original eval.py is preserved.

Design note
    The build_signals_for_video() function centralises the per-protocol
    difference in how windows are formed and how many HR estimates a video
    yields.  Everything downstream (metrics, tables, plots) reads a common
    per-video record so the reporting code is shared across protocols.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr
from tqdm import tqdm

from model_ZOO import build_model
from mega_eval_additions import (collect_deep_features_for_subject,
                                  write_deep_analysis)


# ═══════════════════════════════════════════════════════════════════════════
#  PROTOCOL SELECTION  —  edit this global to choose which protocol(s) to run
# ═══════════════════════════════════════════════════════════════════════════
#   "old"      Old Approach (original eval.py behaviour)
#   "prism"    PRISM Protocol
#   "toolbox"  rPPG-Toolbox Protocol
#   "all"      run all three, each into its own folder
PROTOCOL = "all"
# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
#  OLD PROTOCOL FFT RESOLUTION OVERRIDE
#  ─────────────────────────────────────
#  The old protocol defaults to nfft = signal length (240 pts at 30 Hz),
#  giving 7.5 BPM bin resolution.  Set this to a target resolution in BPM
#  to override with a computed zero-pad length instead.
#
#  Examples:
#    RESOLUTION_OLD_BPM = None     # default — no override, keeps 7.5 BPM bins
#    RESOLUTION_OLD_BPM = 1.0     # zero-pad to ~1800 pts → 1.0 BPM resolution
#    RESOLUTION_OLD_BPM = 0.11    # zero-pad to ~16364 pts → matches PRISM
#
#  The nfft is computed as:  nfft = round(fs * 60 / resolution)
#  using fs = 30.0 Hz (standard across PURE, UBFC, TokyoTech).
# ═══════════════════════════════════════════════════════════════════════════
RESOLUTION_OLD_BPM = 1.0



# ── Paths & shared hyper-parameters ──────────────────────────────────────────


Model_Register_Name = "three_branch_mmi_se_attention"


SAVE_DIR = Path(f"/media/data/rPPG/Code/GitHub/Project_rPPG_Result/Result_Lab_SE_Attention/" + Model_Register_Name + "/")


CKPT_PATH = SAVE_DIR / "last_model.pt"
MANIFEST_PATH = Path(
    "/media/data/rPPG/Code/GitHub/Project_rPPG/Dataset/3_ROI/"
    "PURE-x-UBFC-x-Tokoyo/manifest_split_BALANCED.csv"
)
ROI_INDEX = "avg"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# G-channel SNR threshold separating reliable from unreliable windows.
G_SNR_THRESHOLD = 1.7


# ═══════════════════════════════════════════════════════════════════════════
#  PROTOCOL DEFINITIONS
#  Each protocol is a plain dict of measurement parameters.  No model or loss
#  parameter appears here — protocols only change how the waveform is measured.
# ═══════════════════════════════════════════════════════════════════════════
PROTOCOL_CONFIGS: Dict[str, dict] = {

    "old": {
        "display_name":   "Old Approach",
        "window_s":       8.0,
        "stride_s":       1.0,
        "aggregation":    "window",     # per-window -> per-subject
        "nfft":           None,         # None -> FFT length = signal length (240)
        "bp_low_hz":      0.67,
        "bp_high_hz":     3.0,
        "bp_order":       3,
        "bpm_min":        40.0,
        "bpm_max":        180.0,
        "bpm_max_gt":     150.0,
        "use_hann":       False,
    },

    "prism": {
        "display_name":   "PRISM Protocol",
        "window_s":       10.0,
        "stride_s":       10.0,         # non-overlapping
        "aggregation":    "window",     # per-window -> overall
        "nfft":           16384,        # zero-pad -> ~0.11 BPM resolution
        "bp_low_hz":      0.75,         # standard evaluation bandpass
        "bp_high_hz":     2.5,          # standard evaluation bandpass
        "bp_order":       2,            # 2nd-order Butterworth (community standard)
        "bpm_min":        45.0,         # symmetric with GT range
        "bpm_max":        150.0,        # symmetric with GT range
        "bpm_max_gt":     150.0,        # symmetric with prediction range
        "use_hann":       False,
    },

    "toolbox": {
        "display_name":   "rPPG-Toolbox Protocol",
        "window_s":       10.0,         # inference window; predictions concatenated
        "stride_s":       10.0,         # non-overlapping -> concatenate (option A)
        "aggregation":    "video",      # one HR per video -> overall
        "nfft":           16384,        # zero-pad; full-video signal is already long
        "bp_low_hz":      0.75,
        "bp_high_hz":     2.5,
        "bp_order":       2,
        "bpm_min":        45.0,
        "bpm_max":        150.0,
        "bpm_max_gt":     150.0,
        "use_hann":       False,
    },
}


# ── Utility helpers (shared, unchanged from eval.py) ──────────────────────────

def safe_name(x: str) -> str:
    x = str(x)
    x = re.sub(r"[^\w\-.]+", "_", x)
    return x[:180]


def load_meta(meta_raw):
    if isinstance(meta_raw, np.ndarray):
        meta_raw = meta_raw.item()
    if isinstance(meta_raw, (bytes, bytearray)):
        meta_raw = meta_raw.decode("utf-8", errors="ignore")
    if isinstance(meta_raw, str):
        try:
            return json.loads(meta_raw)
        except Exception:
            return {}
    if isinstance(meta_raw, dict):
        return meta_raw
    return {}


def resolve_npz_path(path_value: str, manifest_path: Path) -> Path:
    p = Path(str(path_value))
    if p.is_absolute():
        return p
    candidates = [manifest_path.parent / p, Path.cwd() / p]
    for parent in manifest_path.resolve().parents:
        candidates.append(parent / p)
    for c in candidates:
        if c.exists():
            return c
    return manifest_path.parent / p


def infer_dataset_name(npz_path: Path, meta: dict, row=None) -> str:
    if row is not None and "dataset" in row and pd.notna(row["dataset"]):
        return str(row["dataset"])
    if "dataset" in meta:
        return str(meta["dataset"])
    for part in npz_path.parts[::-1]:
        if part.endswith("_RAW"):
            return part.replace("_RAW", "")
    return npz_path.parent.name


def select_roi_rgb_np(x: np.ndarray, roi_index):
    if roi_index == "avg":
        T, C = x.shape
        if C % 3 != 0:
            raise ValueError(f"C={C} is not divisible by 3.")
        n_rois = C // 3
        x = x.reshape(T, n_rois, 3)
        return x.mean(axis=1)
    roi_index = int(roi_index)
    s = roi_index * 3
    e = s + 3
    if x.shape[1] < e:
        raise ValueError(f"Input has C={x.shape[1]}, ROI_INDEX={roi_index} needs [{s}:{e}]")
    return x[:, s:e]


def zscore_np(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    y = y - np.mean(y)
    s = np.std(y)
    return y / max(s, eps)


def infer_fs(meta: dict, t: np.ndarray) -> float:
    fs = float(meta.get("fps", np.nan))
    if np.isfinite(fs) and fs > 1.0:
        return fs
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0:
        return 30.0
    return float(1.0 / np.median(dt))


def load_subject_npz(npz_path: Path):
    with np.load(npz_path, allow_pickle=True) as z:
        X    = z["X"].astype(np.float32)
        Y    = z["Y"].astype(np.float32)
        t    = z["t"].astype(np.float64)
        meta = load_meta(z["meta"])
    return X, Y, t, meta


def load_model(ckpt_path: str | Path):
    model = build_model(Model_Register_Name).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model.to(DEVICE), ckpt


def compute_pearson(pred: np.ndarray, target: np.ndarray) -> float:
    pred   = pred.reshape(-1)
    target = target.reshape(-1)
    if len(pred) != len(target) or len(pred) < 2:
        return float("nan")
    try:
        corr, _ = pearsonr(pred, target)
        return float(corr)
    except Exception:
        return float("nan")


# ── Protocol-parameterised signal helpers ────────────────────────────────────

def bandpass_filter(
    sig: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> np.ndarray:
    """Butterworth bandpass.  Cutoffs and order come from the active protocol."""
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    if len(sig) < (order * 3 + 1):
        return sig.copy()
    nyq  = 0.5 * fs
    low  = low_hz  / nyq
    high = high_hz / nyq
    if not (0 < low < high < 1):
        return sig.copy()
    b, a = butter(order, [low, high], btype="band")
    try:
        return filtfilt(b, a, sig)
    except Exception:
        return sig.copy()


def chrom_signal(rgb: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    """CHROM baseline signal, then bandpass with the active protocol's cutoffs."""
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"Expected rgb shape [T,3], got {rgb.shape}")
    eps  = 1e-8
    r    = rgb[:, 0];  g = rgb[:, 1];  b = rgb[:, 2]
    r_n  = r / (np.mean(r) + eps)
    g_n  = g / (np.mean(g) + eps)
    b_n  = b / (np.mean(b) + eps)
    x_c  = 3.0 * r_n - 2.0 * g_n
    y_c  = 1.5 * r_n + g_n - 1.5 * b_n
    alpha = np.std(x_c) / (np.std(y_c) + eps)
    s    = x_c - alpha * y_c
    return bandpass_filter(
        s, fs, cfg["bp_low_hz"], cfg["bp_high_hz"], cfg["bp_order"]
    ).astype(np.float64)


def fft_peak_bpm(
    sig: np.ndarray,
    fs: float,
    bpm_min: float,
    bpm_max: float,
    nfft: int | None,
    use_hann: bool,
) -> float:
    """
    FFT-argmax HR estimate.

    nfft      None -> transform length equals signal length (coarse, Old Approach)
              int  -> zero-pad transform to nfft points (fine, PRISM / Toolbox)
    use_hann  apply a Hann taper before the transform (reduces spectral leakage)
    """
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    if len(sig) < 4 or not np.isfinite(fs) or fs <= 0:
        return float("nan")
    sig = sig - np.mean(sig)
    if use_hann:
        sig = sig * np.hanning(len(sig))

    n = int(nfft) if nfft is not None else len(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec  = np.abs(np.fft.rfft(sig, n=n)) ** 2

    fmin = bpm_min / 60.0
    fmax = bpm_max / 60.0
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return float("nan")
    peak_f = freqs[mask][np.argmax(spec[mask])]
    return float(peak_f * 60.0)


def estimate_hr(
    sig: np.ndarray,
    fs: float,
    cfg: dict,
    is_gt: bool,
) -> float:
    """
    Turn a bandpass-filtered waveform into a BPM value.

    Pure protocol operation: FFT-argmax with the active protocol's nfft,
    frequency range, and optional Hann taper.  No dual-band check, no
    harmonic correction — those belong to PRISM's algorithm, not the protocol.
    """
    bpm_max = cfg["bpm_max_gt"] if is_gt else cfg["bpm_max"]
    return fft_peak_bpm(sig, fs, cfg["bpm_min"], bpm_max, cfg["nfft"], cfg["use_hann"])


# ── Model inference on a single window ────────────────────────────────────────

def predict_window(model: nn.Module, rgb_window: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    """Run the model on one window, then bandpass with the active protocol."""
    xt = torch.from_numpy(rgb_window.astype(np.float32)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(xt).squeeze(0).detach().cpu().numpy()
    return bandpass_filter(
        pred.astype(np.float64), fs,
        cfg["bp_low_hz"], cfg["bp_high_hz"], cfg["bp_order"]
    ).astype(np.float64)


def predict_full_video(model: nn.Module, X_roi: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    """
    Build a full-video waveform by predicting on non-overlapping windows and
    concatenating them end-to-end (option A).  Used by the video-level
    aggregation path.  The concatenated raw prediction is bandpass-filtered
    once as a whole so the spectrum reflects the entire recording.
    """
    win_len = int(round(cfg["window_s"] * fs))
    if len(X_roi) < win_len:
        return np.array([], dtype=np.float64)

    starts = list(range(0, len(X_roi) - win_len + 1, win_len))  # non-overlapping
    pieces = []
    for s in starts:
        e  = s + win_len
        xt = torch.from_numpy(X_roi[s:e].astype(np.float32)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(xt).squeeze(0).detach().cpu().numpy()
        pieces.append(pred.astype(np.float64))

    if not pieces:
        return np.array([], dtype=np.float64)

    full = np.concatenate(pieces)
    return bandpass_filter(
        full, fs, cfg["bp_low_hz"], cfg["bp_high_hz"], cfg["bp_order"]
    ).astype(np.float64)


def chrom_full_video(X_roi: np.ndarray, fs: float, cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Full-video CHROM waveform via non-overlapping concatenation."""
    win_len = int(round(cfg["window_s"] * fs))
    if len(X_roi) < win_len:
        return np.array([]), np.array([])
    starts = list(range(0, len(X_roi) - win_len + 1, win_len))
    pieces = [chrom_signal(X_roi[s:s + win_len], fs, cfg) for s in starts]
    used_len = starts[-1] + win_len if starts else 0
    return np.concatenate(pieces) if pieces else np.array([]), np.arange(used_len)


# ── Failure-type labelling (bins scaled to the active FFT resolution) ─────────

def bin_width_bpm(cfg: dict, fs: float, window_s: float) -> float:
    """Frequency-bin width in BPM under the active FFT settings."""
    n = cfg["nfft"] if cfg["nfft"] is not None else int(round(window_s * fs))
    return (fs / n) * 60.0


def _failure_type(pred_bpm: float, gt_bpm: float, bin_bpm: float) -> str:
    if not (np.isfinite(pred_bpm) and np.isfinite(gt_bpm)):
        return "nan"
    err   = abs(pred_bpm - gt_bpm)
    ratio = gt_bpm / pred_bpm if pred_bpm > 0 else 0.0
    tol   = max(4.0, 0.5 * bin_bpm)
    if err < tol:
        return "correct"
    if 1.7 < ratio < 2.3:
        return "sub_harm_half"
    if 2.5 < ratio < 3.5:
        return "sub_harm_third"
    if 0.3 < ratio < 0.6:
        return "super_harm_2x"
    if 0.6 < ratio < 0.8:
        return "super_harm_1p5x"
    if err <= 1.0 * bin_bpm:
        return "1bin"
    if err <= 2.0 * bin_bpm:
        return "2bin"
    if err <= 3.0 * bin_bpm:
        return "3bin"
    return "large_error"


def _failure_tag(failure_type: str, bin_bpm: float) -> str:
    return {
        "correct":          "\u2713 CORRECT",
        "sub_harm_half":    "\u26a0 SUB-HARMONIC  model = gt\u00f72",
        "sub_harm_third":   "\u26a0 SUB-HARMONIC  model = gt\u00f73",
        "super_harm_2x":    "\u26a0 SUPER-HARMONIC  model = 2\u00d7gt",
        "super_harm_1p5x":  "\u26a0 SUPER-HARMONIC  model = 1.5\u00d7gt",
        "1bin":             f"\u2194 QUANTISATION  1 bin ({bin_bpm:.2f} BPM)",
        "2bin":             f"\u2194 QUANTISATION  2 bins ({2*bin_bpm:.2f} BPM)",
        "3bin":             f"\u2194 QUANTISATION  3 bins ({3*bin_bpm:.2f} BPM)",
        "large_error":      "\u2717 LARGE ERROR",
        "nan":              "?",
    }.get(failure_type, failure_type)


# ── PSD top-peak extraction (uses active FFT resolution) ──────────────────────

def extract_top_psd_peaks(sig: np.ndarray, fs: float, cfg: dict, top_k: int,
                          bpm_max: float) -> dict:
    sig = np.asarray(sig, dtype=np.float64).reshape(-1) - np.mean(sig)
    if cfg["use_hann"] and len(sig) >= 4:
        sig = sig * np.hanning(len(sig))
    n = cfg["nfft"] if cfg["nfft"] is not None else len(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    bpm   = freqs * 60.0
    psd   = np.abs(np.fft.rfft(sig, n=n)) ** 2
    mask  = (bpm >= cfg["bpm_min"]) & (bpm <= bpm_max)
    bpm, psd = bpm[mask], psd[mask]
    if psd.size == 0:
        return {}
    psd_norm = psd / (np.max(psd) + 1e-12)
    idxs = np.argsort(psd_norm)[::-1][:top_k]
    peaks = {}
    for i, idx in enumerate(idxs, start=1):
        peaks[f"top{i}_bpm"]   = float(bpm[idx])
        peaks[f"top{i}_power"] = float(psd_norm[idx])
    return peaks


# ── Subject evaluation (branches on aggregation mode) ─────────────────────────

def evaluate_one_subject(
    npz_path: Path,
    model: nn.Module,
    cfg: dict,
) -> Dict:
    """
    Evaluate one video under the active protocol.

    For "window" aggregation: one HR estimate per window, then averaged.
    For "video"  aggregation: one HR estimate for the whole concatenated video.

    The returned dict keeps the same keys as the original eval.py so all
    downstream tables and plots work unchanged.
    """
    X, Y, t, meta = load_subject_npz(npz_path)
    fs      = infer_fs(meta, t)
    X_roi   = select_roi_rgb_np(X, ROI_INDEX)
    seq_name= str(meta.get("seq", npz_path.stem))
    window_s = cfg["window_s"]
    stride_s = cfg["stride_s"]
    win_len  = int(round(window_s * fs))
    step_len = int(round(stride_s * fs))
    bin_bpm  = bin_width_bpm(cfg, fs, window_s)

    if len(Y) < win_len or len(X_roi) < win_len:
        return {
            "seq": seq_name, "n_windows": 0,
            "pred_gt_hr_mae": np.nan, "chrom_gt_hr_mae": np.nan,
            "pred_gt_pearson": np.nan, "chrom_gt_pearson": np.nan,
            "fs": fs, "error": "sequence_too_short",
            "window_df": pd.DataFrame(),
        }

    # ── VIDEO-LEVEL AGGREGATION (rPPG-Toolbox Protocol) ──────────────────────
    if cfg["aggregation"] == "video":
        pred_full  = predict_full_video(model, X_roi, fs, cfg)
        chrom_full, used_idx = chrom_full_video(X_roi, fs, cfg)
        used_len   = len(used_idx)
        gt_full    = Y[:used_len].astype(np.float64)

        bpm_gt    = estimate_hr(gt_full, fs, cfg, is_gt=True)
        bpm_pred  = estimate_hr(pred_full, fs, cfg, is_gt=False)
        bpm_chrom = estimate_hr(chrom_full, fs, cfg, is_gt=False)

        model_err = abs(bpm_pred  - bpm_gt) if np.isfinite(bpm_gt) and np.isfinite(bpm_pred)  else np.nan
        chrom_err = abs(bpm_chrom - bpm_gt) if np.isfinite(bpm_gt) and np.isfinite(bpm_chrom) else np.nan
        L = min(len(pred_full), len(gt_full), len(chrom_full))
        pred_corr = compute_pearson(pred_full[:L],  gt_full[:L]) if L > 1 else np.nan
        chrom_corr= compute_pearson(chrom_full[:L], gt_full[:L]) if L > 1 else np.nan

        window_rows = [{
            "seq": seq_name, "window_idx": 0,
            "start_idx": 0, "end_idx": used_len,
            "start_time": float(t[0]), "end_time": float(t[min(used_len, len(t)) - 1]),
            "fs": fs,
            "gt_bpm": bpm_gt, "model_bpm": bpm_pred, "chrom_bpm": bpm_chrom,
            "model_err": model_err, "chrom_err": chrom_err,
            "model_minus_chrom": (model_err - chrom_err)
                if np.isfinite(model_err) and np.isfinite(chrom_err) else np.nan,
            "model_pearson": pred_corr, "chrom_pearson": chrom_corr,
            "model_better": (model_err < chrom_err)
                if np.isfinite(model_err) and np.isfinite(chrom_err) else False,
        }]

        return {
            "seq": seq_name,
            "n_windows": 1,
            "pred_gt_hr_mae":  float(model_err) if np.isfinite(model_err) else np.nan,
            "chrom_gt_hr_mae": float(chrom_err) if np.isfinite(chrom_err) else np.nan,
            "pred_gt_pearson":  float(pred_corr)  if np.isfinite(pred_corr)  else np.nan,
            "chrom_gt_pearson": float(chrom_corr) if np.isfinite(chrom_corr) else np.nan,
            "fs": fs, "error": "",
            "window_df": pd.DataFrame(window_rows),
        }

    # ── WINDOW-LEVEL AGGREGATION (Old Approach, PRISM Protocol) ──────────────
    starts = list(range(0, len(Y) - win_len + 1, step_len))
    pred_hr_errs, chrom_hr_errs = [], []
    pred_pearsons, chrom_pearsons = [], []
    window_rows = []

    for w_idx, s in enumerate(starts):
        e       = s + win_len
        rgb_w   = X_roi[s:e]
        gt_w    = Y[s:e].astype(np.float64)
        pred_w  = predict_window(model, rgb_w, fs, cfg)
        chrom_w = chrom_signal(rgb_w, fs, cfg)

        bpm_gt    = estimate_hr(gt_w, fs, cfg, is_gt=True)
        bpm_pred  = estimate_hr(pred_w, fs, cfg, is_gt=False)
        bpm_chrom = estimate_hr(chrom_w, fs, cfg, is_gt=False)

        model_err = abs(bpm_pred  - bpm_gt) if np.isfinite(bpm_gt) and np.isfinite(bpm_pred)  else np.nan
        chrom_err = abs(bpm_chrom - bpm_gt) if np.isfinite(bpm_gt) and np.isfinite(bpm_chrom) else np.nan
        pred_corr = compute_pearson(pred_w, gt_w)
        chrom_corr= compute_pearson(chrom_w, gt_w)

        if np.isfinite(model_err):  pred_hr_errs.append(model_err)
        if np.isfinite(chrom_err):  chrom_hr_errs.append(chrom_err)
        if np.isfinite(pred_corr):  pred_pearsons.append(pred_corr)
        if np.isfinite(chrom_corr): chrom_pearsons.append(chrom_corr)

        window_rows.append({
            "seq": seq_name, "window_idx": w_idx,
            "start_idx": s,  "end_idx": e,
            "start_time": float(t[s]), "end_time": float(t[e - 1]),
            "fs": fs,
            "gt_bpm": bpm_gt, "model_bpm": bpm_pred, "chrom_bpm": bpm_chrom,
            "model_err": model_err, "chrom_err": chrom_err,
            "model_minus_chrom": (model_err - chrom_err)
                if np.isfinite(model_err) and np.isfinite(chrom_err) else np.nan,
            "model_pearson": pred_corr, "chrom_pearson": chrom_corr,
            "model_better": (model_err < chrom_err)
                if np.isfinite(model_err) and np.isfinite(chrom_err) else False,
        })

    return {
        "seq": seq_name,
        "n_windows": len(starts),
        "pred_gt_hr_mae":  float(np.mean(pred_hr_errs))  if pred_hr_errs  else np.nan,
        "chrom_gt_hr_mae": float(np.mean(chrom_hr_errs)) if chrom_hr_errs else np.nan,
        "pred_gt_pearson":  float(np.mean(pred_pearsons))  if pred_pearsons  else np.nan,
        "chrom_gt_pearson": float(np.mean(chrom_pearsons)) if chrom_pearsons else np.nan,
        "fs": fs, "error": "",
        "window_df": pd.DataFrame(window_rows),
    }


# ── PSD top-peak collection (Fix 4 columns, protocol-aware) ───────────────────

def collect_psd_top_peaks_for_subject(
    npz_path: Path, model: nn.Module, cfg: dict,
    dataset_name: str, split_name: str, seq_name: str, subject_id: str,
) -> list:
    X, Y, t, meta = load_subject_npz(npz_path)
    fs       = infer_fs(meta, t)
    X_roi    = select_roi_rgb_np(X, ROI_INDEX)
    win_len  = int(round(cfg["window_s"] * fs))
    step_len = int(round(cfg["stride_s"] * fs))
    starts   = list(range(0, len(Y) - win_len + 1, step_len))
    bin_bpm  = bin_width_bpm(cfg, fs, cfg["window_s"])
    rows     = []

    def _normalize(ch: np.ndarray) -> np.ndarray:
        return ch / (np.mean(ch) + 1e-8)

    for w_idx, s in enumerate(starts):
        e     = s + win_len
        rgb_w = X_roi[s:e]
        gt_w  = Y[s:e].astype(np.float64)

        signals = {
            "R":     _normalize(rgb_w[:, 0]),
            "G":     _normalize(rgb_w[:, 1]),
            "B":     _normalize(rgb_w[:, 2]),
            "GT":    gt_w,
            "CHROM": chrom_signal(rgb_w, fs, cfg),
            "MODEL": predict_window(model, rgb_w, fs, cfg),
        }

        bpm_gt    = estimate_hr(signals["GT"], fs, cfg, is_gt=True)
        bpm_model = estimate_hr(signals["MODEL"], fs, cfg, is_gt=False)
        bpm_chrom = estimate_hr(signals["CHROM"], fs, cfg, is_gt=False)

        row = {
            "dataset": dataset_name, "split": split_name,
            "seq": seq_name, "subject_id": subject_id,
            "window_idx": w_idx, "start_idx": s, "end_idx": e,
            "fs": fs,
            "gt_bpm": bpm_gt, "model_bpm": bpm_model, "chrom_bpm": bpm_chrom,
        }

        for name, sig in signals.items():
            limit = cfg["bpm_max_gt"] if name == "GT" else cfg["bpm_max"]
            peaks = extract_top_psd_peaks(sig, fs, cfg, top_k=3, bpm_max=limit)
            for k, v in peaks.items():
                row[f"{name}_{k}"] = v

        g_top1 = row.get("G_top1_power", 1.0)
        g_top2 = row.get("G_top2_power", 1.0)
        row["G_snr"] = float(g_top1 / (g_top2 + 1e-8))
        row["model_failure_type"] = _failure_type(bpm_model, bpm_gt, bin_bpm)
        row["chrom_failure_type"] = _failure_type(bpm_chrom, bpm_gt, bin_bpm)

        rows.append(row)

    return rows


# ── PSD diagnostic slider (protocol-aware) ────────────────────────────────────

def make_psd_diagnostic_slider(
    npz_path: Path, model: nn.Module, out_html: Path, cfg: dict,
    dataset_name: str, split_name: str,
):
    X, Y, t, meta = load_subject_npz(npz_path)
    seq_name = str(meta.get("seq", npz_path.stem))
    fs       = infer_fs(meta, t)
    X_roi    = select_roi_rgb_np(X, ROI_INDEX)
    win_len  = int(round(cfg["window_s"] * fs))
    step_len = int(round(cfg["stride_s"] * fs))
    bin_bpm  = bin_width_bpm(cfg, fs, cfg["window_s"])

    if len(X_roi) < win_len:
        return

    starts     = list(range(0, len(X_roi) - win_len + 1, step_len))
    n_windows  = len(starts)
    TRACES_PER_WIN = 9

    fig          = go.Figure()
    slider_steps = []
    title_text   = (f"<b>PSD Diagnostic [{cfg['display_name']}] | {dataset_name} | "
                    f"{split_name} | {seq_name}</b>")

    def _normalize(ch: np.ndarray) -> np.ndarray:
        return ch / (np.mean(ch) + 1e-8)

    def _get_psd(sig: np.ndarray):
        sig = np.asarray(sig, dtype=np.float64).reshape(-1) - np.mean(sig)
        if cfg["use_hann"] and len(sig) >= 4:
            sig = sig * np.hanning(len(sig))
        n = cfg["nfft"] if cfg["nfft"] is not None else len(sig)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        bpm_f = freqs * 60.0
        psd   = np.abs(np.fft.rfft(sig, n=n)) ** 2
        mask  = (bpm_f >= cfg["bpm_min"]) & (bpm_f <= cfg["bpm_max"])
        bpm_f = bpm_f[mask];  psd = psd[mask]
        psd   = psd / (np.max(psd) + 1e-12)
        return bpm_f, psd

    SIGNAL_COLORS = {
        "R_norm": "#FF0000", "G_norm": "#00CC33", "B_norm": "#1E00FF",
        "GT": "#111111", "Model": "#9D3DFC", "CHROM": "#FFA500",
    }

    for w_idx, s in enumerate(starts):
        e       = s + win_len
        rgb_w   = X_roi[s:e]
        gt_w    = Y[s:e].astype(np.float64)
        pred_w  = predict_window(model, rgb_w, fs, cfg)
        chrom_w = chrom_signal(rgb_w, fs, cfg)

        bpm_gt    = estimate_hr(gt_w, fs, cfg, is_gt=True)
        bpm_pred  = estimate_hr(pred_w, fs, cfg, is_gt=False)
        bpm_chrom = estimate_hr(chrom_w, fs, cfg, is_gt=False)

        g_peaks  = extract_top_psd_peaks(_normalize(rgb_w[:, 1]), fs, cfg, top_k=2,
                                         bpm_max=cfg["bpm_max"])
        g_snr    = g_peaks.get("top1_power", 1.0) / (g_peaks.get("top2_power", 1.0) + 1e-8)
        snr_flag = "\U0001F7E2" if g_snr >= G_SNR_THRESHOLD else "\U0001F534"

        model_ft = _failure_type(bpm_pred,  bpm_gt, bin_bpm)
        chrom_ft = _failure_type(bpm_chrom, bpm_gt, bin_bpm)

        visible = (w_idx == 0)

        signals = {
            "R_norm": _normalize(rgb_w[:, 0]),
            "G_norm": _normalize(rgb_w[:, 1]),
            "B_norm": _normalize(rgb_w[:, 2]),
            "GT":     gt_w,
            "Model":  pred_w,
            "CHROM":  chrom_w,
        }
        for name, sig in signals.items():
            bpm_axis, psd_vals = _get_psd(sig)
            peak_bpm = bpm_axis[np.argmax(psd_vals)] if len(psd_vals) else float("nan")
            fig.add_trace(go.Scatter(
                x=bpm_axis, y=psd_vals, mode="lines+markers",
                name=f"{name} | peak={peak_bpm:.1f}", visible=visible,
                line=dict(color=SIGNAL_COLORS.get(name, "gray")),
            ))

        vline_defs = [
            (bpm_gt,       f"GT={bpm_gt:.1f} BPM",      "blue",   "dash"),
            (bpm_gt / 2.0, f"gt\u00f72={bpm_gt/2:.1f} BPM", "red", "dashdot"),
            (min(bpm_gt * 2.0, cfg["bpm_max"]),
                           f"gt\u00d72={bpm_gt*2:.1f} BPM", "purple", "dot"),
        ]
        for x_val, label, col, dash in vline_defs:
            if not np.isfinite(x_val):
                fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name=label,
                                         visible=visible, showlegend=False))
            else:
                fig.add_trace(go.Scatter(
                    x=[x_val, x_val], y=[0.0, 1.05], mode="lines", name=label,
                    visible=visible, line=dict(color=col, dash=dash, width=1.5),
                    showlegend=True,
                ))

        visibility = [False] * (TRACES_PER_WIN * n_windows)
        for offset in range(TRACES_PER_WIN):
            visibility[TRACES_PER_WIN * w_idx + offset] = True

        slider_steps.append(dict(
            method="update",
            args=[{"visible": visibility},
                  {"title": (f"{title_text}<br><span style='font-size:12px'>"
                             f"Win {w_idx} | GT={bpm_gt:.1f}  Model={bpm_pred:.1f}  "
                             f"CHROM={bpm_chrom:.1f} BPM | {snr_flag} G_SNR={g_snr:.2f} | "
                             f"Model: {_failure_tag(model_ft, bin_bpm)} | "
                             f"CHROM: {_failure_tag(chrom_ft, bin_bpm)}</span>")}],
            label=f"Win {w_idx}",
        ))

    fig.update_layout(
        title=title_text, xaxis_title="Heart Rate (BPM)",
        yaxis_title="Normalised PSD Power", template="plotly_white",
        width=1500, height=750,
        sliders=[dict(active=0, currentvalue={"prefix": "Window: "},
                      pad={"t": 50}, steps=slider_steps)],
        legend=dict(orientation="v", x=1.02, y=1.0),
        margin=dict(l=80, r=230, t=120, b=100),
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs=True)


# ── Signal comparison slider (protocol-aware) ─────────────────────────────────

def make_signal_comparison_slider(
    npz_path: Path, model: nn.Module, out_html: Path, cfg: dict,
    dataset_name: str, split_name: str,
):
    X, Y, t, meta = load_subject_npz(npz_path)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    seq_name = str(meta.get("seq", npz_path.stem))
    fs       = infer_fs(meta, t)
    X_roi    = select_roi_rgb_np(X, ROI_INDEX)
    win_len  = int(round(cfg["window_s"] * fs))
    step_len = int(round(cfg["stride_s"] * fs))

    if len(X_roi) < win_len:
        return

    starts    = list(range(0, len(X_roi) - win_len + 1, step_len))
    n_windows = len(starts)
    fig       = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35],
                              subplot_titles=("BVP Waveforms", "Frequency Spectra"),
                              vertical_spacing=0.12)
    slider_steps = []
    title_text   = f"<b>[{cfg['display_name']}] {dataset_name} | {split_name} | {seq_name}</b>"

    for w_idx, s in enumerate(starts):
        e       = s + win_len
        rgb_w   = X_roi[s:e]
        gt_w    = Y[s:e].astype(np.float64)
        t_w     = t[s:e]
        pred_w  = predict_window(model, rgb_w, fs, cfg)
        chrom_w = chrom_signal(rgb_w, fs, cfg)

        pred_bpm  = estimate_hr(pred_w, fs, cfg, is_gt=False)
        gt_bpm    = estimate_hr(gt_w, fs, cfg, is_gt=True)
        chrom_bpm = estimate_hr(chrom_w, fs, cfg, is_gt=False)

        pred_mae  = abs(pred_bpm  - gt_bpm) if np.isfinite(pred_bpm)  and np.isfinite(gt_bpm) else np.nan
        chrom_mae = abs(chrom_bpm - gt_bpm) if np.isfinite(chrom_bpm) and np.isfinite(gt_bpm) else np.nan
        pred_corr = compute_pearson(pred_w, gt_w)
        chrom_corr= compute_pearson(chrom_w, gt_w)
        visible   = (w_idx == 0)

        fig.add_trace(go.Scatter(x=t_w, y=zscore_np(gt_w), mode="lines",
            name=f"GT | HR={gt_bpm:.1f} BPM",
            line=dict(color="#2E86AB", width=2), visible=visible, legendgroup="gt"),
            row=1, col=1)
        fig.add_trace(go.Scatter(x=t_w, y=zscore_np(pred_w), mode="lines",
            name=f"Model | HR={pred_bpm:.1f} | MAE={pred_mae:.2f} | r={pred_corr:.3f}",
            line=dict(color="#F18F01", width=2), visible=visible, legendgroup="model"),
            row=1, col=1)
        fig.add_trace(go.Scatter(x=t_w, y=zscore_np(chrom_w), mode="lines",
            name=f"CHROM | HR={chrom_bpm:.1f} | MAE={chrom_mae:.2f} | r={chrom_corr:.3f}",
            line=dict(color="#A23B72", width=2, dash="dot"), visible=visible, legendgroup="chrom"),
            row=1, col=1)

        n = cfg["nfft"] if cfg["nfft"] is not None else len(gt_w)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        bpm_f = freqs * 60.0
        mask  = (bpm_f >= cfg["bpm_min"]) & (bpm_f <= cfg["bpm_max"])
        gt_fft   = np.abs(np.fft.rfft(gt_w    - np.mean(gt_w),    n=n)) ** 2
        pred_fft = np.abs(np.fft.rfft(pred_w  - np.mean(pred_w),  n=n)) ** 2
        chrom_fft= np.abs(np.fft.rfft(chrom_w - np.mean(chrom_w), n=n)) ** 2

        fig.add_trace(go.Scatter(x=bpm_f[mask], y=gt_fft[mask]/ (gt_fft[mask].max()+1e-12),
            mode="lines", name="GT Spectrum", line=dict(color="#2E86AB", width=2),
            visible=visible, showlegend=False, legendgroup="gt"), row=2, col=1)
        fig.add_trace(go.Scatter(x=bpm_f[mask], y=pred_fft[mask]/ (pred_fft[mask].max()+1e-12),
            mode="lines", name="Model Spectrum", line=dict(color="#F18F01", width=2),
            visible=visible, showlegend=False, legendgroup="model"), row=2, col=1)
        fig.add_trace(go.Scatter(x=bpm_f[mask], y=chrom_fft[mask]/ (chrom_fft[mask].max()+1e-12),
            mode="lines", name="CHROM Spectrum", line=dict(color="#A23B72", width=2, dash="dot"),
            visible=visible, showlegend=False, legendgroup="chrom"), row=2, col=1)

        visibility = [False] * (6 * n_windows)
        for offset in range(6):
            visibility[6 * w_idx + offset] = True
        slider_steps.append(dict(method="update",
            args=[{"visible": visibility}, {"title": title_text}],
            label=f"Win {w_idx}"))

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Normalised Amplitude", row=1, col=1)
    fig.update_xaxes(title_text="Heart Rate (BPM)", row=2, col=1)
    fig.update_yaxes(title_text="Normalised Power", row=2, col=1)
    fig.update_layout(title=title_text, template="plotly_white", width=1600, height=900,
        sliders=[dict(active=0, currentvalue={"prefix": "Window: "},
                      pad={"t": 50}, steps=slider_steps)],
        legend=dict(orientation="v", xanchor="left", x=1.01, y=1.0),
        margin=dict(l=80, r=180, t=100, b=100), font=dict(size=11))
    fig.write_html(str(out_html), include_plotlyjs=True)


# ── Video-level signal comparison (for rPPG-Toolbox protocol) ─────────────────

def make_signal_comparison_video(
    npz_path: Path, model: nn.Module, out_html: Path, cfg: dict,
    dataset_name: str, split_name: str,
):
    """
    Single full-video signal comparison plot (no slider).
    Used when cfg["aggregation"] == "video".

    Row 1 — z-scored GT, Model, and CHROM waveforms across the full video.
    Row 2 — FFT spectra of each, computed on the full concatenated signal.
    """
    X, Y, t, meta = load_subject_npz(npz_path)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    seq_name = str(meta.get("seq", npz_path.stem))
    fs       = infer_fs(meta, t)
    X_roi    = select_roi_rgb_np(X, ROI_INDEX)

    # build full-video signals via non-overlapping window concatenation
    pred_full  = predict_full_video(model, X_roi, fs, cfg)
    chrom_full_sig, used_idx = chrom_full_video(X_roi, fs, cfg)
    used_len = len(used_idx)

    if pred_full.size == 0 or chrom_full_sig.size == 0 or used_len < 2:
        return

    gt_full = Y[:used_len].astype(np.float64)
    t_full  = t[:used_len]

    # trim to common length
    L = min(len(pred_full), len(gt_full), len(chrom_full_sig), len(t_full))
    pred_full    = pred_full[:L]
    gt_full      = gt_full[:L]
    chrom_full_sig = chrom_full_sig[:L]
    t_full       = t_full[:L]

    # extract HR from full-video signals
    gt_bpm    = estimate_hr(gt_full,      fs, cfg, is_gt=True)
    pred_bpm  = estimate_hr(pred_full,    fs, cfg, is_gt=False)
    chrom_bpm = estimate_hr(chrom_full_sig, fs, cfg, is_gt=False)

    pred_mae  = abs(pred_bpm  - gt_bpm)  if np.isfinite(pred_bpm)  and np.isfinite(gt_bpm) else np.nan
    chrom_mae = abs(chrom_bpm - gt_bpm)  if np.isfinite(chrom_bpm) and np.isfinite(gt_bpm) else np.nan
    pred_corr = compute_pearson(pred_full, gt_full)
    chrom_corr= compute_pearson(chrom_full_sig, gt_full)

    title_text = (f"<b>[{cfg['display_name']}] {dataset_name} | {split_name} | {seq_name}</b>"
                  f"<br><span style='font-size:11px'>"
                  f"GT={gt_bpm:.1f}  Model={pred_bpm:.1f} (MAE={pred_mae:.2f}, r={pred_corr:.3f})  "
                  f"CHROM={chrom_bpm:.1f} (MAE={chrom_mae:.2f}, r={chrom_corr:.3f})"
                  f"</span>")

    fig = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35],
                        subplot_titles=("Full-Video BVP Waveforms", "Full-Video Frequency Spectra"),
                        vertical_spacing=0.12)

    # row 1 — time-domain waveforms
    fig.add_trace(go.Scatter(x=t_full, y=zscore_np(gt_full), mode="lines",
        name=f"GT | HR={gt_bpm:.1f} BPM",
        line=dict(color="#2E86AB", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_full, y=zscore_np(pred_full), mode="lines",
        name=f"Model | HR={pred_bpm:.1f} | MAE={pred_mae:.2f} | r={pred_corr:.3f}",
        line=dict(color="#F18F01", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_full, y=zscore_np(chrom_full_sig), mode="lines",
        name=f"CHROM | HR={chrom_bpm:.1f} | MAE={chrom_mae:.2f} | r={chrom_corr:.3f}",
        line=dict(color="#A23B72", width=2, dash="dot")), row=1, col=1)

    # row 2 — frequency spectra
    n = cfg["nfft"] if cfg["nfft"] is not None else L
    freqs  = np.fft.rfftfreq(n, d=1.0 / fs)
    bpm_f  = freqs * 60.0
    mask   = (bpm_f >= cfg["bpm_min"]) & (bpm_f <= cfg["bpm_max"])
    gt_fft    = np.abs(np.fft.rfft(gt_full      - np.mean(gt_full),      n=n)) ** 2
    pred_fft  = np.abs(np.fft.rfft(pred_full    - np.mean(pred_full),    n=n)) ** 2
    chrom_fft = np.abs(np.fft.rfft(chrom_full_sig - np.mean(chrom_full_sig), n=n)) ** 2

    fig.add_trace(go.Scatter(x=bpm_f[mask], y=gt_fft[mask]/(gt_fft[mask].max()+1e-12),
        mode="lines", name="GT Spectrum", line=dict(color="#2E86AB", width=2),
        showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=bpm_f[mask], y=pred_fft[mask]/(pred_fft[mask].max()+1e-12),
        mode="lines", name="Model Spectrum", line=dict(color="#F18F01", width=2),
        showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=bpm_f[mask], y=chrom_fft[mask]/(chrom_fft[mask].max()+1e-12),
        mode="lines", name="CHROM Spectrum", line=dict(color="#A23B72", width=2, dash="dot"),
        showlegend=False), row=2, col=1)

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Normalised Amplitude", row=1, col=1)
    fig.update_xaxes(title_text="Heart Rate (BPM)", row=2, col=1)
    fig.update_yaxes(title_text="Normalised Power", row=2, col=1)
    fig.update_layout(title=title_text, template="plotly_white", width=1600, height=900,
        legend=dict(orientation="v", xanchor="left", x=1.01, y=1.0),
        margin=dict(l=80, r=180, t=120, b=100), font=dict(size=11))
    fig.write_html(str(out_html), include_plotlyjs=True)


def make_psd_diagnostic_video(
    npz_path: Path, model: nn.Module, out_html: Path, cfg: dict,
    dataset_name: str, split_name: str,
):
    """
    Single full-video PSD diagnostic plot (no slider).
    Used when cfg["aggregation"] == "video".

    Shows the PSD of R, G, B (normalised), GT, Model, and CHROM signals
    computed on the full concatenated video.
    """
    X, Y, t, meta = load_subject_npz(npz_path)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    seq_name = str(meta.get("seq", npz_path.stem))
    fs       = infer_fs(meta, t)
    X_roi    = select_roi_rgb_np(X, ROI_INDEX)

    pred_full  = predict_full_video(model, X_roi, fs, cfg)
    chrom_full_sig, used_idx = chrom_full_video(X_roi, fs, cfg)
    used_len = len(used_idx)

    if pred_full.size == 0 or chrom_full_sig.size == 0 or used_len < 2:
        return

    gt_full = Y[:used_len].astype(np.float64)
    L = min(len(pred_full), len(gt_full), len(chrom_full_sig))
    pred_full      = pred_full[:L]
    gt_full        = gt_full[:L]
    chrom_full_sig = chrom_full_sig[:L]

    # normalised RGB channels for the used portion of the video
    eps = 1e-8
    rgb_used = X_roi[:used_len].astype(np.float64)
    r_norm = rgb_used[:L, 0] / (np.mean(rgb_used[:L, 0]) + eps)
    g_norm = rgb_used[:L, 1] / (np.mean(rgb_used[:L, 1]) + eps)
    b_norm = rgb_used[:L, 2] / (np.mean(rgb_used[:L, 2]) + eps)

    gt_bpm    = estimate_hr(gt_full,      fs, cfg, is_gt=True)
    pred_bpm  = estimate_hr(pred_full,    fs, cfg, is_gt=False)
    chrom_bpm = estimate_hr(chrom_full_sig, fs, cfg, is_gt=False)

    title_text = (f"<b>PSD Diagnostic [{cfg['display_name']}] | {dataset_name} | "
                  f"{split_name} | {seq_name}</b>"
                  f"<br><span style='font-size:12px'>"
                  f"GT={gt_bpm:.1f}  Model={pred_bpm:.1f}  CHROM={chrom_bpm:.1f} BPM"
                  f"</span>")

    def _get_psd(sig):
        sig = np.asarray(sig, dtype=np.float64).reshape(-1) - np.mean(sig)
        n = cfg["nfft"] if cfg["nfft"] is not None else len(sig)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        bpm_f = freqs * 60.0
        psd   = np.abs(np.fft.rfft(sig, n=n)) ** 2
        mask  = (bpm_f >= cfg["bpm_min"]) & (bpm_f <= cfg["bpm_max"])
        bpm_f = bpm_f[mask]; psd = psd[mask]
        psd   = psd / (np.max(psd) + 1e-12)
        return bpm_f, psd

    SIGNAL_COLORS = {
        "R_norm": "#FF0000", "G_norm": "#00CC33", "B_norm": "#1E00FF",
        "GT": "#111111", "Model": "#9D3DFC", "CHROM": "#FFA500",
    }
    signals = {
        "R_norm": r_norm, "G_norm": g_norm, "B_norm": b_norm,
        "GT": gt_full, "Model": pred_full, "CHROM": chrom_full_sig,
    }

    fig = go.Figure()
    for name, sig in signals.items():
        bpm_axis, psd_vals = _get_psd(sig)
        peak_bpm = bpm_axis[np.argmax(psd_vals)] if len(psd_vals) else float("nan")
        fig.add_trace(go.Scatter(
            x=bpm_axis, y=psd_vals, mode="lines",
            name=f"{name} | peak={peak_bpm:.1f}",
            line=dict(color=SIGNAL_COLORS.get(name, "gray")),
        ))

    # vertical markers for GT, GT/2, GT*2
    vline_defs = [
        (gt_bpm,                            f"GT={gt_bpm:.1f} BPM",      "blue",   "dash"),
        (gt_bpm / 2.0,                      f"GT/2={gt_bpm/2:.1f} BPM",  "red",    "dashdot"),
        (min(gt_bpm * 2.0, cfg["bpm_max"]), f"GT*2={gt_bpm*2:.1f} BPM",  "purple", "dot"),
    ]
    for x_val, label, col, dash in vline_defs:
        if np.isfinite(x_val):
            fig.add_trace(go.Scatter(
                x=[x_val, x_val], y=[0.0, 1.05], mode="lines", name=label,
                line=dict(color=col, dash=dash, width=1.5), showlegend=True))

    fig.update_layout(
        title=title_text, xaxis_title="Heart Rate (BPM)",
        yaxis_title="Normalised PSD Power", template="plotly_white",
        width=1500, height=750,
        legend=dict(orientation="v", x=1.02, y=1.0),
        margin=dict(l=80, r=230, t=120, b=100))
    fig.write_html(str(out_html), include_plotlyjs=True)


# ── Bar / scatter / correlation plots (unchanged logic, protocol in title) ────

def get_split_color(split: str):
    split = str(split).lower()
    if split == "train":       return "#000AE3", "#FF6903"
    if split in ["val","valid","validation"]: return "#7B00BA", "#FF6903"
    if split in ["test","testing"]:           return "#6D45FF", "#FF6903"
    return "#007DE3", "#E36500"


def make_hr_mae_comparison_bar(df: pd.DataFrame, out_html: Path, plot_title: str):
    if len(df) == 0:
        return
    df_plot  = df.copy().sort_values(["split","dataset","seq"]).reset_index(drop=True)
    x_ids    = [f"x_{i}" for i in range(len(df_plot))]
    tick_text, colors_pred, colors_chrom = [], [], []
    for _, row in df_plot.iterrows():
        color = get_split_color(row["split"])
        colors_pred.append(color[0]); colors_chrom.append(color[1])
        tick_text.append(f"<span style='color:{color[0]}'>{row['seq']}</span>")

    pred_summary  = " | ".join([f"{r['split']} Model={r['pred_gt_hr_mae']:.2f}"
                                for _, r in df_plot.groupby("split")["pred_gt_hr_mae"].mean().reset_index().iterrows()
                                if np.isfinite(r["pred_gt_hr_mae"])])
    chrom_summary = " | ".join([f"{r['split']} CHROM={r['chrom_gt_hr_mae']:.2f}"
                                for _, r in df_plot.groupby("split")["chrom_gt_hr_mae"].mean().reset_index().iterrows()
                                if np.isfinite(r["chrom_gt_hr_mae"])])

    custom_data = np.stack([df_plot["dataset"], df_plot["split"], df_plot["seq"],
                            df_plot["subject_id"], df_plot["n_windows"], df_plot["fs"]], axis=-1)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_ids, y=df_plot["pred_gt_hr_mae"], name="Model",
                         marker_color=colors_pred, customdata=custom_data,
                         hovertemplate="<b>%{customdata[2]}</b><br>Model MAE: %{y:.2f} BPM<extra></extra>"))
    fig.add_trace(go.Bar(x=x_ids, y=df_plot["chrom_gt_hr_mae"], name="CHROM",
                         marker_color=colors_chrom, customdata=custom_data,
                         marker_pattern_shape="-",
                         hovertemplate="<b>%{customdata[2]}</b><br>CHROM MAE: %{y:.2f} BPM<extra></extra>"))
    fig.update_layout(
        title=f"<b>{plot_title}</b><br><span style='font-size:11px'>{pred_summary}<br>{chrom_summary}</span>",
        xaxis_title="Video Sequence", yaxis=dict(title="HR MAE (BPM)", range=[0, 25]),
        barmode="group", template="plotly_white",
        width=max(1600, len(df_plot) * 50), height=600,
        xaxis=dict(tickmode="array", tickvals=x_ids, ticktext=tick_text, tickangle=-45),
        legend=dict(orientation="h", xanchor="center", x=0.5, yanchor="bottom", y=1.02),
        margin=dict(l=80, r=40, t=120, b=180), font=dict(size=11))
    fig.write_html(str(out_html), include_plotlyjs=True)


def make_scatter_comparison(df: pd.DataFrame, out_html: Path, plot_title: str):
    if len(df) == 0:
        return
    df_plot = df[df["pred_gt_hr_mae"].notna() & df["chrom_gt_hr_mae"].notna()].copy()
    if len(df_plot) == 0:
        return
    colors  = [get_split_color(split)[0] for split in df_plot["split"]]
    max_val = max(df_plot["chrom_gt_hr_mae"].max(), df_plot["pred_gt_hr_mae"].max())
    model_better = (df_plot["pred_gt_hr_mae"] < df_plot["chrom_gt_hr_mae"]).sum()
    total = len(df_plot)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot["chrom_gt_hr_mae"], y=df_plot["pred_gt_hr_mae"],
                             mode="markers", marker=dict(size=10, color=colors),
                             text=[f"{r['seq']}<br>{r['split']}" for _, r in df_plot.iterrows()],
                             hovertemplate="<b>%{text}</b><br>CHROM: %{x:.2f}<br>Model: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines",
                             line=dict(color="gray", dash="dash", width=2), name="y = x"))
    fig.update_layout(
        title=f"<b>{plot_title}</b><br><span style='font-size:11px'>Model wins: {model_better}/{total}</span>",
        xaxis_title="CHROM HR MAE (BPM)", yaxis_title="Model HR MAE (BPM)",
        template="plotly_white", width=700, height=700)
    fig.write_html(str(out_html), include_plotlyjs=True)


def make_correlation_comparison(df: pd.DataFrame, out_html: Path, plot_title: str):
    if len(df) == 0:
        return
    df_plot = df[df["pred_gt_pearson"].notna() & df["chrom_gt_pearson"].notna()].copy()
    if len(df_plot) == 0:
        return
    df_plot = df_plot.sort_values(["split","dataset","seq"]).reset_index(drop=True)
    x_ids   = [f"x_{i}" for i in range(len(df_plot))]
    colors  = [get_split_color(r["split"])[0] for _, r in df_plot.iterrows()]
    tick_text = [f"<span style='color:{c}'>{r['seq']}</span>"
                 for c, (_, r) in zip(colors, df_plot.iterrows())]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_ids, y=df_plot["pred_gt_pearson"], name="Model", marker_color=colors))
    fig.add_trace(go.Bar(x=x_ids, y=df_plot["chrom_gt_pearson"], name="CHROM",
                         marker_color=colors, marker_pattern_shape="/"))
    fig.update_layout(
        title=f"<b>{plot_title}</b>",
        xaxis_title="Video Sequence",
        yaxis=dict(title="Pearson Correlation with GT", range=[0, 1]),
        barmode="group", template="plotly_white",
        width=max(1600, len(df_plot) * 50), height=600,
        xaxis=dict(tickmode="array", tickvals=x_ids, ticktext=tick_text, tickangle=-45),
        legend=dict(orientation="h", xanchor="center", x=0.5, yanchor="bottom", y=1.02),
        margin=dict(l=80, r=40, t=100, b=180))
    fig.write_html(str(out_html), include_plotlyjs=True)


# ── Metric helpers for the extended summary (RMSE, SD, Acc) ───────────────────

def _rmse(errs: np.ndarray) -> float:
    errs = errs[np.isfinite(errs)]
    return float(np.sqrt(np.mean(errs ** 2))) if errs.size else np.nan


def _sd(errs: np.ndarray) -> float:
    errs = errs[np.isfinite(errs)]
    return float(np.std(errs)) if errs.size else np.nan


def _acc_within(errs: np.ndarray, thr: float = 5.0) -> float:
    errs = errs[np.isfinite(errs)]
    return float(100.0 * np.mean(errs <= thr)) if errs.size else np.nan


# ── Output directories, per protocol ──────────────────────────────────────────

def make_output_dirs(protocol_key: str) -> Dict[str, Path]:
    if protocol_key == "old" and RESOLUTION_OLD_BPM is not None:
        protocol_key = f"old_{RESOLUTION_OLD_BPM}BPM"
    base_dir         = SAVE_DIR / f"EVAL_PROTOCOL_{protocol_key}"
    bars_dir         = base_dir / "bar_plots"
    signals_dir      = base_dir / "signal_comparisons"
    scatter_dir      = base_dir / "scatter_plots"
    corr_dir         = base_dir / "correlation_plots"
    tables_dir       = base_dir / "tables"
    diagnostics_dir  = base_dir / "diagnostics"
    trajectories_dir = base_dir / "trajectories"

    for d in [base_dir, bars_dir, signals_dir, scatter_dir, corr_dir,
              tables_dir, diagnostics_dir, trajectories_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "base": base_dir, "bars": bars_dir, "signals": signals_dir,
        "scatter": scatter_dir, "corr": corr_dir, "tables": tables_dir,
        "diagnostics": diagnostics_dir, "trajectories": trajectories_dir,
    }


# ── Run a single protocol end-to-end ──────────────────────────────────────────

def run_protocol(protocol_key: str, model: nn.Module, manifest_df: pd.DataFrame):
    #cfg      = PROTOCOL_CONFIGS[protocol_key]
    cfg      = dict(PROTOCOL_CONFIGS[protocol_key])  # shallow copy — safe to modify
    if protocol_key == "old" and RESOLUTION_OLD_BPM is not None:
        cfg["nfft"] = int(round(30.0 * 60.0 / RESOLUTION_OLD_BPM))
        
    out_dirs = make_output_dirs(protocol_key)

    print(f"\n{'#'*74}")
    print(f"#  PROTOCOL: {cfg['display_name']}  (key='{protocol_key}')")
    print(f"#  window={cfg['window_s']}s stride={cfg['stride_s']}s  "
          f"nfft={cfg['nfft']}  band={cfg['bp_low_hz']}-{cfg['bp_high_hz']}Hz  "
          f"agg={cfg['aggregation']}")
    print(f"#  Output -> {out_dirs['base']}")
    print(f"{'#'*74}\n")

    rows, all_window_rows, all_psd_peak_rows, all_deep_rows = [], [], [], []

    for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df),
                       desc=f"[{protocol_key}] Evaluating", ncols=100):
        rel_or_abs_path = row["path"]
        npz_path = resolve_npz_path(rel_or_abs_path, MANIFEST_PATH)

        try:
            _, _, _, meta_tmp = load_subject_npz(npz_path)
            dataset_name = infer_dataset_name(npz_path, meta_tmp, row)

            result = evaluate_one_subject(npz_path=npz_path, model=model, cfg=cfg)

            seq_str    = str(row["seq"])
            seq_name   = seq_str if seq_str not in ("", "nan", "None") else result["seq"]
            split_name = str(row["split"])
            window_df  = result.pop("window_df", pd.DataFrame())

            if len(window_df) > 0:
                window_df["dataset"]       = dataset_name
                window_df["split"]         = split_name
                window_df["seq"]           = seq_name
                window_df["subject_id"]    = str(row["subject_id"])
                window_df["path"]          = str(rel_or_abs_path)
                window_df["resolved_path"] = str(npz_path)
                all_window_rows.append(window_df)

            result.update({
                "dataset": dataset_name, "split": split_name,
                "seq": seq_name, "subject_id": str(row["subject_id"]),
                "path": str(rel_or_abs_path), "resolved_path": str(npz_path),
            })
            rows.append(result)

            if result["n_windows"] > 0:
                sig_html = (out_dirs["signals"] /
                    f"{safe_name(dataset_name)}_{safe_name(split_name)}_{safe_name(seq_name)}.html")
                psd_html = (out_dirs["diagnostics"] /
                    f"{safe_name(dataset_name)}_{safe_name(split_name)}_{safe_name(seq_name)}_PSD.html")

                if cfg["aggregation"] == "video":
                    # video-level: single full-video plots, no slider
                    make_signal_comparison_video(
                        npz_path=npz_path, model=model, out_html=sig_html,
                        cfg=cfg, dataset_name=dataset_name, split_name=split_name)
                    make_psd_diagnostic_video(
                        npz_path=npz_path, model=model, out_html=psd_html,
                        cfg=cfg, dataset_name=dataset_name, split_name=split_name)
                else:
                    # window-level: slider-based window-by-window plots
                    make_signal_comparison_slider(
                        npz_path=npz_path, model=model, out_html=sig_html,
                        cfg=cfg, dataset_name=dataset_name, split_name=split_name)
                    make_psd_diagnostic_slider(
                        npz_path=npz_path, model=model, out_html=psd_html,
                        cfg=cfg, dataset_name=dataset_name, split_name=split_name)

                    # PSD peak collection and deep features: window-level only
                    all_psd_peak_rows.extend(
                        collect_psd_top_peaks_for_subject(
                            npz_path=npz_path, model=model, cfg=cfg,
                            dataset_name=dataset_name, split_name=split_name,
                            seq_name=seq_name, subject_id=str(row["subject_id"])))

                    all_deep_rows.extend(
                        collect_deep_features_for_subject(
                            npz_path=npz_path, model=model,
                            roi_index=ROI_INDEX, window_s=cfg["window_s"],
                            stride_s=cfg["stride_s"],
                            dataset_name=dataset_name, split_name=split_name,
                            seq_name=seq_name, subject_id=str(row["subject_id"])))

        except Exception as e:
            import traceback
            traceback.print_exc()
            rows.append({
                "dataset": "unknown", "split": str(row.get("split", "")),
                "seq": str(row.get("seq", "")), "subject_id": str(row.get("subject_id", "")),
                "path": str(rel_or_abs_path), "resolved_path": str(npz_path),
                "n_windows": 0, "pred_gt_hr_mae": np.nan, "chrom_gt_hr_mae": np.nan,
                "pred_gt_pearson": np.nan, "chrom_gt_pearson": np.nan,
                "fs": np.nan, "error": str(e),
            })

    # ── Save tables ───────────────────────────────────────────────────────────
    full_df = pd.DataFrame(rows)
    full_df.to_csv(out_dirs["tables"] / "ALL_EVAL_RESULTS.csv", index=False)

    if all_window_rows:
        all_window_df = pd.concat(all_window_rows, ignore_index=True)
        all_window_df.to_csv(out_dirs["tables"] / "ALL_WINDOW_ERRORS.csv", index=False)
        (all_window_df.sort_values("model_minus_chrom", ascending=False)
         .to_csv(out_dirs["tables"] / "BAD_WINDOWS_MODEL_WORSE_THAN_CHROM.csv", index=False))

    if all_psd_peak_rows:
        pd.DataFrame(all_psd_peak_rows).to_csv(
            out_dirs["tables"] / "PSD_TOP_PEAKS_SUMMARY.csv", index=False)

    if all_deep_rows:
        deep_df = pd.DataFrame(all_deep_rows)
        deep_df.to_csv(out_dirs["tables"] / "DEEP_FEATURES_SUMMARY.csv", index=False)
        write_deep_analysis(deep_df, out_dirs["tables"], out_dirs["diagnostics"])

    # ── Console + summary.txt ─────────────────────────────────────────────────
    print(f"\n{'='*70}\nSPLIT-WISE SUMMARY  [{cfg['display_name']}]\n{'='*70}\n")

    split_summary = (full_df.groupby("split", dropna=False)
                     .agg(N=("seq","count"),
                          model_mae=("pred_gt_hr_mae","mean"),
                          chrom_mae=("chrom_gt_hr_mae","mean"),
                          model_r=("pred_gt_pearson","mean"),
                          chrom_r=("chrom_gt_pearson","mean")).reset_index())
    ds_split_summary = (full_df.groupby(["dataset","split"], dropna=False)
                        .agg(N=("seq","count"),
                             model_mae=("pred_gt_hr_mae","mean"),
                             chrom_mae=("chrom_gt_hr_mae","mean"),
                             model_r=("pred_gt_pearson","mean"),
                             chrom_r=("chrom_gt_pearson","mean")).reset_index())

    # extended per-split metrics (RMSE, SD, Acc) from per-video errors
    ext_rows = []
    for split_val, g in full_df.groupby("split", dropna=False):
        m_err = g["pred_gt_hr_mae"].to_numpy(dtype=float)
        c_err = g["chrom_gt_hr_mae"].to_numpy(dtype=float)
        ext_rows.append({
            "split": split_val,
            "model_MAE":  np.nanmean(m_err), "model_RMSE": _rmse(m_err),
            "model_SD":   _sd(m_err),        "model_Acc5": _acc_within(m_err),
            "chrom_MAE":  np.nanmean(c_err), "chrom_RMSE": _rmse(c_err),
            "chrom_SD":   _sd(c_err),        "chrom_Acc5": _acc_within(c_err),
        })
    ext_df = pd.DataFrame(ext_rows)
    ext_df.to_csv(out_dirs["tables"] / "EXTENDED_METRICS.csv", index=False)

    for _, r in split_summary.iterrows():
        diff = r["chrom_mae"] - r["model_mae"]
        sign = "\u2193" if diff > 0 else "\u2191"
        print(f"Split: {r['split']}  N={int(r['N'])}")
        print(f"  Model: MAE={r['model_mae']:.3f} BPM | r={r['model_r']:.3f}")
        print(f"  CHROM: MAE={r['chrom_mae']:.3f} BPM | r={r['chrom_r']:.3f}")
        if np.isfinite(diff):
            print(f"  Gap: {sign} {abs(diff):.3f} BPM")
        print()

    txt_path = out_dirs["tables"] / "summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\nEVALUATION CONFIGURATION\n" + "="*80 + "\n\n")
        f.write(f"Protocol:      {cfg['display_name']}  (key='{protocol_key}')\n")
        f.write(f"Checkpoint:    {CKPT_PATH}\n")
        f.write(f"Manifest:      {MANIFEST_PATH}\n")
        f.write(f"Window:        {cfg['window_s']}s\n")
        f.write(f"Stride:        {cfg['stride_s']}s\n")
        f.write(f"Aggregation:   {cfg['aggregation']}\n")
        f.write(f"FFT nfft:      {cfg['nfft']}  (None = signal length)\n")
        f.write(f"Bandpass:      {cfg['bp_low_hz']}-{cfg['bp_high_hz']} Hz, order {cfg['bp_order']}\n")
        f.write(f"Note:          protocol-only evaluation (no dual-band harmonic check)\n")
        f.write(f"HR range:      {cfg['bpm_min']}-{cfg['bpm_max']} BPM (GT max {cfg['bpm_max_gt']})\n")
        f.write(f"Hann taper:    {cfg['use_hann']}\n")
        f.write(f"G_SNR thresh:  {G_SNR_THRESHOLD}\n\n")
        f.write("="*80 + "\nSPLIT-WISE SUMMARY\n" + "="*80 + "\n\n")
        for _, r in split_summary.iterrows():
            f.write(f"Split: {r['split']}\n  N={int(r['N'])}\n")
            f.write(f"  Model: HR MAE={r['model_mae']:.3f} BPM | Pearson={r['model_r']:.3f}\n")
            f.write(f"  CHROM: HR MAE={r['chrom_mae']:.3f} BPM | Pearson={r['chrom_r']:.3f}\n\n")
        f.write("="*80 + "\nEXTENDED METRICS (MAE / RMSE / SD / Acc@5BPM)\n" + "="*80 + "\n\n")
        for _, r in ext_df.iterrows():
            f.write(f"Split: {r['split']}\n")
            f.write(f"  Model: MAE={r['model_MAE']:.3f}  RMSE={r['model_RMSE']:.3f}  "
                    f"SD={r['model_SD']:.3f}  Acc@5={r['model_Acc5']:.1f}%\n")
            f.write(f"  CHROM: MAE={r['chrom_MAE']:.3f}  RMSE={r['chrom_RMSE']:.3f}  "
                    f"SD={r['chrom_SD']:.3f}  Acc@5={r['chrom_Acc5']:.1f}%\n\n")
        f.write("="*80 + "\nDATASET + SPLIT SUMMARY\n" + "="*80 + "\n\n")
        for _, r in ds_split_summary.iterrows():
            f.write(f"Dataset={r['dataset']} | Split={r['split']} | N={int(r['N'])}\n")
            f.write(f"  Model: HR MAE={r['model_mae']:.3f} | Pearson={r['model_r']:.3f}\n")
            f.write(f"  CHROM: HR MAE={r['chrom_mae']:.3f} | Pearson={r['chrom_r']:.3f}\n\n")

    # ── Per-dataset visualisations ────────────────────────────────────────────
    print("Generating visualisations...")
    for dataset_name, ddf in full_df.groupby("dataset"):
        tag = f"{dataset_name} [{cfg['display_name']}]"
        make_hr_mae_comparison_bar(ddf, out_dirs["bars"] / f"{safe_name(dataset_name)}_hr_mae.html",
                                   f"HR MAE Comparison \u2014 {tag}")
        make_scatter_comparison(ddf, out_dirs["scatter"] / f"{safe_name(dataset_name)}_scatter.html",
                                f"Model vs CHROM \u2014 {tag}")
        make_correlation_comparison(ddf, out_dirs["corr"] / f"{safe_name(dataset_name)}_correlation.html",
                                    f"Waveform Correlation \u2014 {tag}")

    print(f"\n{'='*70}")
    print(f"Protocol '{protocol_key}' complete.  Results in: {out_dirs['base']}")
    print(f"{'='*70}\n")

    # return the split summary so an "all" run can print a combined ablation table
    return split_summary.assign(protocol=cfg["display_name"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading model from: {CKPT_PATH}")
    model, ckpt = load_model(CKPT_PATH)
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    df = pd.read_csv(MANIFEST_PATH,
                     dtype={"split": str, "path": str, "seq": str, "subject_id": str})
    for col in ("split", "seq", "subject_id"):
        if col not in df.columns:
            df[col] = "unknown"
    print(f"\nEvaluating {len(df)} videos...")

    if PROTOCOL == "all":
        protocols_to_run = ["old", "prism", "toolbox"]
    else:
        if PROTOCOL not in PROTOCOL_CONFIGS:
            raise ValueError(f"Unknown PROTOCOL='{PROTOCOL}'. "
                             f"Choose from {list(PROTOCOL_CONFIGS) + ['all']}.")
        protocols_to_run = [PROTOCOL]

    ablation_frames = []
    for pkey in protocols_to_run:
        summ = run_protocol(pkey, model, df)
        ablation_frames.append(summ)

    # ── Combined ablation table when more than one protocol was run ───────────
    if len(ablation_frames) > 1:
        ablation = pd.concat(ablation_frames, ignore_index=True)
        ablation_dir = SAVE_DIR / "EVAL_PROTOCOL_ablation"
        ablation_dir.mkdir(parents=True, exist_ok=True)
        ablation.to_csv(ablation_dir / "PROTOCOL_ABLATION_SUMMARY.csv", index=False)

        print(f"\n{'='*74}\nPROTOCOL ABLATION SUMMARY (model MAE by split)\n{'='*74}\n")
        pivot = ablation.pivot_table(index="split", columns="protocol",
                                     values="model_mae", aggfunc="mean")
        print(pivot.round(3).to_string())
        print(f"\nSaved combined ablation table -> "
              f"{ablation_dir / 'PROTOCOL_ABLATION_SUMMARY.csv'}\n")


if __name__ == "__main__":
    main()
