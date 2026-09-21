"""
eval.py — rPPG evaluation script
─────────────────────────────────────
Changes from the original version:

  Fix 1  predict_window — fs was hardcoded as 30.0; now passed as a
         parameter so the bandpass filter uses the actual recording fps.

  Fix 2  make_psd_diagnostic_slider — vlines for gt_bpm, gt÷2, and gt×2
         are now added as per-window Scatter traces so they move with the
         slider.  Title now shows G-channel SNR and a failure-type tag.

  Fix 3  make_hr_trajectory_plot (new) — time-axis plot of GT / Model /
         CHROM HR per window, with a second panel showing G-channel SNR.
         Sub-harmonic failure windows are marked with red × symbols.

  Fix 4  collect_psd_top_peaks_for_subject — three new columns added:
         G_snr, model_failure_type, chrom_failure_type.

  Fix 5  All call sites of predict_window updated to pass fs=fs.
"""

from __future__ import annotations

from importlib.resources import path
import json
import re
from pathlib import Path
from typing import Dict
from unicodedata import name

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr
from tqdm import tqdm

#from model import SimpleRGBProjector
#from model import SimpleRGBProjector,TwoBranchAdaptive
#from model import AdaptiveDetrendProjector
#from model_temporal import build_model
from model_ZOO import build_model

from HIT_train import SAVE_DIR

from mega_eval_additions import (collect_deep_features_for_subject,
                                  write_deep_analysis)


# ── Paths & hyper-parameters ─────────────────────────────────────────────────

CKPT_PATH = SAVE_DIR / "last_model.pt"

MANIFEST_PATH = Path(
    "/media/data/rPPG/Code/GitHub/Project_rPPG/Dataset/3_ROI/"
    "PURE-x-UBFC-x-Tokoyo/manifest_split.csv"
)

ROI_INDEX = "avg"
WINDOW_S  = 8.0
STRIDE_S  = 1.0
BPM_MIN   = 40.0
BPM_MAX   = 180.0
BPM_MAX_GT = 150.0  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# G-channel SNR threshold that separates reliable from unreliable windows.
# Derived from the analysis: perfect-window median = 2.40, error-window
# median = 1.60–1.70.  Threshold sits between the two distributions.
G_SNR_THRESHOLD = 1.7


# ── Output directories ────────────────────────────────────────────────────────

def make_output_dirs() -> Dict[str, Path]:
    #base_dir         = SAVE_DIR / f"In_detailed_EVAL_WIN_S-{WINDOW_S}"
    base_dir         = SAVE_DIR / f"EVAL_WIN_S-{WINDOW_S}"
    bars_dir         = base_dir / "bar_plots"
    signals_dir      = base_dir / "signal_comparisons"
    scatter_dir      = base_dir / "scatter_plots"
    corr_dir         = base_dir / "correlation_plots"
    tables_dir       = base_dir / "tables"
    diagnostics_dir  = base_dir / "diagnostics"
    trajectories_dir = base_dir / "trajectories"          # Fix 3 — new

    for d in [base_dir, bars_dir, signals_dir, scatter_dir, corr_dir,
              tables_dir, diagnostics_dir, trajectories_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "base":         base_dir,
        "bars":         bars_dir,
        "signals":      signals_dir,
        "scatter":      scatter_dir,
        "corr":         corr_dir,
        "tables":       tables_dir,
        "diagnostics":  diagnostics_dir,
        "trajectories": trajectories_dir,               # Fix 3 — new
    }


# ── Utility helpers ───────────────────────────────────────────────────────────

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


def bandpass_filter(
    sig: np.ndarray,
    fs: float,
    low_hz: float = 0.67,
    high_hz: float = 4.0,
    order: int = 3,
) -> np.ndarray:
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


def chrom_signal(rgb: np.ndarray, fs: float) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"Expected rgb shape [T,3], got {rgb.shape}")
    eps  = 1e-8
    r    = rgb[:, 0];  g = rgb[:, 1];  b = rgb[:, 2]
    r_n  = r  / (np.mean(r)  + eps)
    g_n  = g  / (np.mean(g)  + eps)
    b_n  = b  / (np.mean(b)  + eps)
    x_c  = 3.0 * r_n - 2.0 * g_n
    y_c  = 1.5 * r_n + g_n - 1.5 * b_n
    alpha = np.std(x_c) / (np.std(y_c) + eps)
    s    = x_c - alpha * y_c
    return bandpass_filter(s, fs).astype(np.float64)



# def chrom_signal(rgb: np.ndarray, fs: float) -> np.ndarray:
#     rgb = np.asarray(rgb, dtype=np.float64)
#     if rgb.ndim != 2 or rgb.shape[1] != 3:
#         raise ValueError(f"Expected rgb shape [T,3], got {rgb.shape}")
#     eps  = 1e-8
#     r    = rgb[:, 0];  g = rgb[:, 1];  b = rgb[:, 2]
#     r_n  = r  / (np.mean(r)  + eps)
#     g_n  = g  / (np.mean(g)  + eps)
#     b_n  = b  / (np.mean(b)  + eps)

#     s    = 0.385* r_n - 0.574 * g_n + 0.15 * b_n
#     return bandpass_filter(s, fs).astype(np.float64)




def fft_peak_bpm_1d(
    sig: np.ndarray,
    fs: float,
    bpm_min: float = 40.0,
    bpm_max: float = 180.0,
) -> float:
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    if len(sig) < 4 or not np.isfinite(fs) or fs <= 0:
        return float("nan")
    sig   = sig - np.mean(sig)
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    spec  = np.abs(np.fft.rfft(sig)) ** 2
    fmin  = bpm_min / 60.0
    fmax  = bpm_max / 60.0
    mask  = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return float("nan")
    peak_f = freqs[mask][np.argmax(spec[mask])]
    return float(peak_f * 60.0)





def fft_peak_bpm_quadratic_1d(
    sig: np.ndarray,
    fs: float,
    bpm_min: float = 40.0,
    bpm_max: float = 180.0,
) -> float:
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)

    if len(sig) < 4 or not np.isfinite(fs) or fs <= 0:
        return float("nan")

    sig = sig - np.mean(sig)

    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    spec = np.abs(np.fft.rfft(sig)) ** 2

    mask = (freqs >= bpm_min / 60.0) & (freqs <= bpm_max / 60.0)

    if not np.any(mask):
        return float("nan")

    freqs_b = freqs[mask]
    spec_b = spec[mask]

    i = int(np.argmax(spec_b))

    if i == 0 or i == len(spec_b) - 1:
        return float(freqs_b[i] * 60.0)

    y0 = np.log(spec_b[i - 1] + 1e-12)
    y1 = np.log(spec_b[i] + 1e-12)
    y2 = np.log(spec_b[i + 1] + 1e-12)

    denom = y0 - 2.0 * y1 + y2

    if abs(denom) < 1e-12:
        return float(freqs_b[i] * 60.0)

    delta = 0.5 * (y0 - y2) / denom
    delta = np.clip(delta, -0.5, 0.5)

    df = freqs_b[1] - freqs_b[0]
    peak_f = freqs_b[i] + delta * df

    return float(peak_f * 60.0)



def fft_peak_bpm_1d_Hann(
    sig: np.ndarray,
    fs: float,
    bpm_min: float = 40.0,
    bpm_max: float = 180.0,
) -> float:
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    if len(sig) < 4 or not np.isfinite(fs) or fs <= 0:
        return float("nan")
    sig   = sig - np.mean(sig)
    win   = np.hanning(len(sig))          # Hann window
    sig   = sig * win                      # apply taper before FFT
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    spec  = np.abs(np.fft.rfft(sig)) ** 2
    fmin  = bpm_min / 60.0
    fmax  = bpm_max / 60.0
    mask  = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return float("nan")
    peak_f = freqs[mask][np.argmax(spec[mask])]
    return float(peak_f * 60.0)





HR_Function = fft_peak_bpm_1d

#HR_Function = fft_peak_bpm_1d_Hann







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
    #model = SimpleRGBProjector()
    #model = TwoBranchAdaptive()
    #model = AdaptiveDetrendProjector()
    name = "plain_k1"
    model = build_model(name).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    # w = model.proj.weight.data.squeeze()

    # print(f"R = {w[0].item():+.8f}")
    # print(f"G = {w[1].item():+.8f}")
    # print(f"B = {w[2].item():+.8f}")

    return model.to(DEVICE), ckpt


# ── Signal extraction ─────────────────────────────────────────────────────────

# Fix 1 — fs is now a required parameter; removed the hardcoded 30.0
def predict_window(model: nn.Module, rgb_window: np.ndarray,
                   fs: float = 30.0) -> np.ndarray:
    """
    Run the model on one window and return a bandpass-filtered BVP signal.
    fs must match the recording frame rate — not 30.0 for all datasets.
    """
    xt   = torch.from_numpy(rgb_window.astype(np.float32)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(xt).squeeze(0).detach().cpu().numpy()
    return bandpass_filter(
        pred.astype(np.float64), fs=fs, low_hz=0.67, high_hz=3.0, order=3
    ).astype(np.float64)


# ── Failure-type labelling ────────────────────────────────────────────────────

def _failure_type(pred_bpm: float, gt_bpm: float) -> str:
    """
    Classify a single window's prediction into one of six failure types.
    Used in both the PSD slider title and the PSD peaks CSV.
    """
    if not (np.isfinite(pred_bpm) and np.isfinite(gt_bpm)):
        return "nan"
    err   = abs(pred_bpm - gt_bpm)
    ratio = gt_bpm / pred_bpm if pred_bpm > 0 else 0.0
    if err < 4.0:
        return "correct"
    if 1.7 < ratio < 2.3:
        return "sub_harm_half"     # model predicts gt÷2
    if 2.5 < ratio < 3.5:
        return "sub_harm_third"    # model predicts gt÷3
    if 0.3 < ratio < 0.6:
        return "super_harm_2x"     # model predicts 2×gt
    if 0.6 < ratio < 0.8:
        return "super_harm_1p5x"
    if err <= 7.5:
        return "1bin"
    if err <= 15.0:
        return "2bin"
    if err <= 22.5:
        return "3bin"
    return "large_error"


def _failure_tag(failure_type: str) -> str:
    """Short display string for the PSD slider title."""
    return {
        "correct":          "✓ CORRECT",
        "sub_harm_half":    "⚠ SUB-HARMONIC  model = gt÷2",
        "sub_harm_third":   "⚠ SUB-HARMONIC  model = gt÷3",
        "super_harm_2x":    "⚠ SUPER-HARMONIC  model = 2×gt",
        "super_harm_1p5x":  "⚠ SUPER-HARMONIC  model = 1.5×gt",
        "1bin":             "↔ QUANTISATION  1 bin (7.5 BPM)",
        "2bin":             "↔ QUANTISATION  2 bins (15 BPM)",
        "3bin":             "↔ QUANTISATION  3 bins (22.5 BPM)",
        "large_error":      "✗ LARGE ERROR",
        "nan":              "?",
    }.get(failure_type, failure_type)


# ── Subject evaluation ────────────────────────────────────────────────────────

def evaluate_one_subject(
    npz_path: Path,
    model: nn.Module,
    roi_index,
    window_s: float,
    stride_s: float,
) -> Dict:
    X, Y, t, meta = load_subject_npz(npz_path)
    fs      = infer_fs(meta, t)
    X_roi   = select_roi_rgb_np(X, roi_index)
    win_len = int(round(window_s * fs))
    step_len= int(round(stride_s * fs))
    seq_name= str(meta.get("seq", npz_path.stem))

    if len(Y) < win_len or len(X_roi) < win_len:
        return {
            "seq": seq_name, "n_windows": 0,
            "pred_gt_hr_mae": np.nan, "chrom_gt_hr_mae": np.nan,
            "pred_gt_pearson": np.nan, "chrom_gt_pearson": np.nan,
            "fs": fs, "error": "sequence_too_short",
            "window_df": pd.DataFrame(),
        }

    starts = list(range(0, len(Y) - win_len + 1, step_len))
    pred_hr_errs, chrom_hr_errs = [], []
    pred_pearsons, chrom_pearsons = [], []
    window_rows = []

    for w_idx, s in enumerate(starts):
        e       = s + win_len
        rgb_w   = X_roi[s:e]
        gt_w    = Y[s:e]
        pred_w  = predict_window(model, rgb_w, fs=fs)          # Fix 1
        chrom_w = chrom_signal(rgb_w, fs)

        bpm_gt    = HR_Function(gt_w,    fs, BPM_MIN, BPM_MAX_GT)
        bpm_pred  = HR_Function(pred_w,  fs, BPM_MIN, BPM_MAX)
        bpm_chrom = HR_Function(chrom_w, fs, BPM_MIN, BPM_MAX)




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
            "model_minus_chrom": model_err - chrom_err if np.isfinite(model_err) and np.isfinite(chrom_err) else np.nan,
            "model_pearson": pred_corr, "chrom_pearson": chrom_corr,
            "model_better": (model_err < chrom_err) if np.isfinite(model_err) and np.isfinite(chrom_err) else False,
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


# ── PSD top-peak extraction ───────────────────────────────────────────────────

def extract_top_psd_peaks(sig: np.ndarray, fs: float, top_k: int = 3,
                          bpm_max: float = BPM_MAX) -> dict:
    sig       = np.asarray(sig, dtype=np.float64).reshape(-1) - np.mean(sig)
    freqs     = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    bpm       = freqs * 60.0
    psd       = np.abs(np.fft.rfft(sig)) ** 2
    mask      = (bpm >= BPM_MIN) & (bpm <= bpm_max)      # ← was BPM_MAX
    bpm, psd  = bpm[mask], psd[mask]
    psd_norm  = psd / (np.max(psd) + 1e-12)
    idxs      = np.argsort(psd_norm)[::-1][:top_k]
    peaks = {}
    for i, idx in enumerate(idxs, start=1):
        peaks[f"top{i}_bpm"]   = float(bpm[idx])
        peaks[f"top{i}_power"] = float(psd_norm[idx])
    return peaks


# ── PSD diagnostic slider (Fix 2) ────────────────────────────────────────────

def make_psd_diagnostic_slider(
    npz_path: Path,
    model: nn.Module,
    out_html: Path,
    roi_index,
    window_s: float,
    stride_s: float,
    dataset_name: str,
    split_name: str,
):
    """
    Interactive PSD plot with one slider step per window.

    Fix 2a — vertical marker lines for gt_bpm, gt÷2, and gt×2 are added as
              per-window Scatter traces so they update with the slider.
    Fix 2b — slider step title now includes G-channel SNR and a failure-type
              label (e.g. '⚠ SUB-HARMONIC  model = gt÷2').
    """
    X, Y, t, meta = load_subject_npz(npz_path)
    seq_name = str(meta.get("seq", npz_path.stem))
    fs       = infer_fs(meta, t)
    X_roi    = select_roi_rgb_np(X, roi_index)
    win_len  = int(round(window_s * fs))
    step_len = int(round(stride_s * fs))

    if len(X_roi) < win_len:
        return

    starts     = list(range(0, len(X_roi) - win_len + 1, step_len))
    n_windows  = len(starts)
    # Fix 2 — 9 traces per window: 6 signals + 3 vertical marker lines
    TRACES_PER_WIN = 9

    fig          = go.Figure()
    slider_steps = []
    title_text   = f"<b>PSD Diagnostic | {dataset_name} | {split_name} | {seq_name}</b>"

    def _normalize(ch: np.ndarray) -> np.ndarray:
        return ch / (np.mean(ch) + 1e-8)

    def _get_psd(sig: np.ndarray):
        sig   = np.asarray(sig, dtype=np.float64).reshape(-1) - np.mean(sig)
        freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
        bpm_f = freqs * 60.0
        psd   = np.abs(np.fft.rfft(sig)) ** 2
        mask  = (bpm_f >= BPM_MIN) & (bpm_f <= BPM_MAX)
        bpm_f = bpm_f[mask];  psd = psd[mask]
        psd   = psd / (np.max(psd) + 1e-12)
        return bpm_f, psd

    SIGNAL_COLORS = {
        "R_norm": "#FF0000",
        "G_norm": "#00CC33",
        "B_norm": "#1E00FF",
        "GT":     "#111111",
        "Model":  "#9D3DFC",
        "CHROM":  "#FFA500",
    }

    for w_idx, s in enumerate(starts):
        e       = s + win_len
        rgb_w   = X_roi[s:e]
        gt_w    = Y[s:e]
        pred_w  = predict_window(model, rgb_w, fs=fs)          # Fix 1
        chrom_w = chrom_signal(rgb_w, fs)

        bpm_gt    = HR_Function(gt_w,    fs, BPM_MIN, BPM_MAX_GT)
        bpm_pred  = HR_Function(pred_w,  fs, BPM_MIN, BPM_MAX)
        bpm_chrom = HR_Function(chrom_w, fs, BPM_MIN, BPM_MAX)

        # Fix 2b — G-channel SNR for this window
        g_peaks  = extract_top_psd_peaks(_normalize(rgb_w[:, 1]), fs, top_k=2)
        g_snr    = g_peaks["top1_power"] / (g_peaks.get("top2_power", 1.0) + 1e-8)
        snr_flag = "🟢" if g_snr >= G_SNR_THRESHOLD else "🔴"

        # Fix 2b — failure type for model prediction
        model_ft = _failure_type(bpm_pred,  bpm_gt)
        chrom_ft = _failure_type(bpm_chrom, bpm_gt)

        visible = (w_idx == 0)

        # ── 6 signal traces ───────────────────────────────────────────────
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
            peak_bpm = bpm_axis[np.argmax(psd_vals)]
            fig.add_trace(go.Scatter(
                x=bpm_axis, y=psd_vals,
                mode="lines+markers",
                name=f"{name} | peak={peak_bpm:.1f}",
                visible=visible,
                line=dict(color=SIGNAL_COLORS.get(name, "gray")),
            ))

        # ── Fix 2a — 3 dynamic vertical marker lines ──────────────────────
        vline_defs = [
            (bpm_gt,             f"GT={bpm_gt:.1f} BPM",       "blue",   "dash"),
            (bpm_gt / 2.0,       f"gt÷2={bpm_gt/2:.1f} BPM",  "red",    "dashdot"),
            (min(bpm_gt * 2.0, BPM_MAX),
                                  f"gt×2={bpm_gt*2:.1f} BPM", "purple", "dot"),
        ]
        for x_val, label, col, dash in vline_defs:
            if not np.isfinite(x_val):
                # Add a dummy invisible trace to keep the trace count constant
                fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                                         name=label, visible=visible,
                                         showlegend=False))
            else:
                fig.add_trace(go.Scatter(
                    x=[x_val, x_val], y=[0.0, 1.05],
                    mode="lines",
                    name=label,
                    visible=visible,
                    line=dict(color=col, dash=dash, width=1.5),
                    showlegend=True,
                ))

        # ── Slider step ───────────────────────────────────────────────────
        visibility = [False] * (TRACES_PER_WIN * n_windows)
        for offset in range(TRACES_PER_WIN):
            visibility[TRACES_PER_WIN * w_idx + offset] = True

        slider_steps.append(dict(
            method="update",
            args=[
                {"visible": visibility},
                {"title": (
                    f"{title_text}<br>"
                    f"<span style='font-size:12px'>"
                    f"Win {w_idx} | "
                    f"GT={bpm_gt:.1f}  Model={bpm_pred:.1f}  CHROM={bpm_chrom:.1f} BPM | "
                    f"{snr_flag} G_SNR={g_snr:.2f} | "
                    f"Model: {_failure_tag(model_ft)} | "
                    f"CHROM: {_failure_tag(chrom_ft)}"
                    f"</span>"
                )},
            ],
            label=f"Win {w_idx}",
        ))

    fig.update_layout(
        title=title_text,
        xaxis_title="Heart Rate (BPM)",
        yaxis_title="Normalised PSD Power",
        template="plotly_white",
        width=1500, height=750,
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Window: "},
            pad={"t": 50},
            steps=slider_steps,
        )],
        legend=dict(orientation="v", x=1.02, y=1.0),
        margin=dict(l=80, r=230, t=120, b=100),
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs=True)


# ── PSD top-peak collection (Fix 4) ──────────────────────────────────────────

def collect_psd_top_peaks_for_subject(
    npz_path: Path,
    model: nn.Module,
    roi_index,
    window_s: float,
    stride_s: float,
    dataset_name: str,
    split_name: str,
    seq_name: str,
    subject_id: str,
) -> list:
    """
    Fix 4 — three new columns added to each row:
      G_snr               — G top1 / G top2 power ratio (quality indicator)
      model_failure_type  — one of: correct / sub_harm_half / 2bin / etc.
      chrom_failure_type  — same taxonomy for CHROM
    """
    X, Y, t, meta = load_subject_npz(npz_path)
    fs       = infer_fs(meta, t)
    X_roi    = select_roi_rgb_np(X, roi_index)
    win_len  = int(round(window_s * fs))
    step_len = int(round(stride_s * fs))
    starts   = list(range(0, len(Y) - win_len + 1, step_len))
    rows     = []

    def _normalize(ch: np.ndarray) -> np.ndarray:
        return ch / (np.mean(ch) + 1e-8)

    for w_idx, s in enumerate(starts):
        e       = s + win_len
        rgb_w   = X_roi[s:e]
        gt_w    = Y[s:e]

        signals = {
            "R":     _normalize(rgb_w[:, 0]),
            "G":     _normalize(rgb_w[:, 1]),
            "B":     _normalize(rgb_w[:, 2]),
            "GT":    gt_w,
            "CHROM": chrom_signal(rgb_w, fs),
            "MODEL": predict_window(model, rgb_w, fs=fs),       # Fix 1
        }

        bpm_gt    = HR_Function(signals["GT"],    fs, BPM_MIN, BPM_MAX_GT)
        bpm_model = HR_Function(signals["MODEL"], fs, BPM_MIN, BPM_MAX)
        bpm_chrom = HR_Function(signals["CHROM"], fs, BPM_MIN, BPM_MAX)

        row = {
            "dataset": dataset_name, "split": split_name,
            "seq": seq_name, "subject_id": subject_id,
            "window_idx": w_idx, "start_idx": s, "end_idx": e,
            "fs": fs,
            "gt_bpm":    bpm_gt,
            "model_bpm": bpm_model,
            "chrom_bpm": bpm_chrom,
        }

        for name, sig in signals.items():
            limit = BPM_MAX_GT if name == "GT" else BPM_MAX
            peaks = extract_top_psd_peaks(sig, fs, top_k=3, bpm_max=limit)
            for k, v in peaks.items():
                row[f"{name}_{k}"] = v

        # Fix 4 — G-channel SNR
        g_top1 = row.get("G_top1_power", 1.0)
        g_top2 = row.get("G_top2_power", 1.0)
        row["G_snr"] = float(g_top1 / (g_top2 + 1e-8))

        # Fix 4 — failure type labels
        row["model_failure_type"] = _failure_type(bpm_model, bpm_gt)
        row["chrom_failure_type"] = _failure_type(bpm_chrom, bpm_gt)

        rows.append(row)

    return rows


# ── HR trajectory plot (Fix 3 — new function) ────────────────────────────────

def make_hr_trajectory_plot(
    npz_path: Path,
    model: nn.Module,
    out_html: Path,
    roi_index,
    window_s: float,
    stride_s: float,
    dataset_name: str,
    split_name: str,
):
    """
    Fix 3 — new diagnostic: temporal HR trajectory over all windows.

    Row 1 — GT / Model / CHROM HR per window on the time axis.
            Sub-harmonic failure windows are marked with red × symbols.
            Makes sporadic dips to 45 / 52.5 BPM visually obvious.

    Row 2 — G-channel SNR per window with the reliability threshold line.
            Periods below G_SNR_THRESHOLD are shaded in orange.

    The combination of the two panels immediately shows whether a failure
    window coincides with low G-channel signal quality.
    """
    X, Y, t, meta = load_subject_npz(npz_path)
    seq_name = str(meta.get("seq", npz_path.stem))
    fs       = infer_fs(meta, t)
    X_roi    = select_roi_rgb_np(X, roi_index)
    win_len  = int(round(window_s * fs))
    step_len = int(round(stride_s * fs))
    starts   = list(range(0, len(Y) - win_len + 1, step_len))

    gt_hrs, model_hrs, chrom_hrs, g_snrs, win_times = [], [], [], [], []
    model_failure_types, chrom_failure_types = [], []

    def _normalize(ch: np.ndarray) -> np.ndarray:
        return ch / (np.mean(ch) + 1e-8)

    for s in starts:
        e       = s + win_len
        rgb_w   = X_roi[s:e]
        gt_w    = Y[s:e]
        pred_w  = predict_window(model, rgb_w, fs=fs)           # Fix 1
        chrom_w = chrom_signal(rgb_w, fs)

        bpm_gt    = HR_Function(gt_w,    fs, BPM_MIN, BPM_MAX_GT)
        bpm_model = HR_Function(pred_w,  fs, BPM_MIN, BPM_MAX)
        bpm_chrom = HR_Function(chrom_w, fs, BPM_MIN, BPM_MAX)

        g_peaks = extract_top_psd_peaks(_normalize(rgb_w[:, 1]), fs, top_k=2)
        g_snr   = g_peaks["top1_power"] / (g_peaks.get("top2_power", 1.0) + 1e-8)

        gt_hrs.append(bpm_gt);   model_hrs.append(bpm_model)
        chrom_hrs.append(bpm_chrom); g_snrs.append(g_snr)
        win_times.append(float(t[s]))
        model_failure_types.append(_failure_type(bpm_model, bpm_gt))
        chrom_failure_types.append(_failure_type(bpm_chrom, bpm_gt))

    win_times   = np.array(win_times,   dtype=np.float64)
    gt_hrs      = np.array(gt_hrs,      dtype=np.float64)
    model_hrs   = np.array(model_hrs,   dtype=np.float64)
    chrom_hrs   = np.array(chrom_hrs,   dtype=np.float64)
    g_snrs      = np.array(g_snrs,      dtype=np.float64)

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("HR estimate per window", "G-channel SNR (quality indicator)"),
    )

    # ── Row 1: HR trajectory ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=win_times, y=gt_hrs, mode="lines",
        name="GT HR", line=dict(color="#2E86AB", width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=win_times, y=model_hrs, mode="lines",
        name="Model HR", line=dict(color="#F18F01", width=1.5),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=win_times, y=chrom_hrs, mode="lines",
        name="CHROM HR", line=dict(color="#A23B72", width=1.5, dash="dot"),
    ), row=1, col=1)

    # Sub-harmonic model failures — red × markers
    sh_mask_model = np.array([ft in ("sub_harm_half", "sub_harm_third")
                               for ft in model_failure_types])
    if sh_mask_model.any():
        fig.add_trace(go.Scatter(
            x=win_times[sh_mask_model], y=model_hrs[sh_mask_model],
            mode="markers",
            marker=dict(color="red", size=11, symbol="x-thin",
                        line=dict(width=2.5, color="red")),
            name=f"Model sub-harmonic ({sh_mask_model.sum()} wins)",
            hovertemplate=(
                "Win time: %{x:.1f}s<br>"
                "Model: %{y:.1f} BPM<br>"
                "Expected GT: <extra></extra>"
            ),
        ), row=1, col=1)

    # Super-harmonic model failures — orange △ markers
    sp_mask_model = np.array([ft in ("super_harm_2x", "super_harm_1p5x")
                               for ft in model_failure_types])
    if sp_mask_model.any():
        fig.add_trace(go.Scatter(
            x=win_times[sp_mask_model], y=model_hrs[sp_mask_model],
            mode="markers",
            marker=dict(color="darkorange", size=10, symbol="triangle-up"),
            name=f"Model super-harmonic ({sp_mask_model.sum()} wins)",
        ), row=1, col=1)

    # CHROM sub-harmonic failures — dark red ○ markers
    sh_mask_chrom = np.array([ft in ("sub_harm_half", "sub_harm_third")
                               for ft in chrom_failure_types])
    if sh_mask_chrom.any():
        fig.add_trace(go.Scatter(
            x=win_times[sh_mask_chrom], y=chrom_hrs[sh_mask_chrom],
            mode="markers",
            marker=dict(color="darkred", size=11, symbol="circle-open",
                        line=dict(width=2.0)),
            name=f"CHROM sub-harmonic ({sh_mask_chrom.sum()} wins)",
        ), row=1, col=1)

    # ── Row 2: G-channel SNR ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=win_times, y=g_snrs, mode="lines",
        name="G SNR", line=dict(color="#1D9E75", width=1.5),
        fill="tozeroy", fillcolor="rgba(29,158,117,0.12)",
    ), row=2, col=1)

    # Threshold line
    fig.add_hline(
        y=G_SNR_THRESHOLD, row=2, col=1,
        line_dash="dash", line_color="orange", line_width=1.5,
        annotation_text=f"Threshold = {G_SNR_THRESHOLD}",
        annotation_position="top right",
        annotation_font_size=11,
    )

    # Shade low-SNR regions (below threshold) in light orange
    low_snr_mask = g_snrs < G_SNR_THRESHOLD
    if low_snr_mask.any():
        fig.add_trace(go.Scatter(
            x=np.concatenate([win_times, win_times[::-1]]),
            y=np.concatenate([
                np.where(low_snr_mask, g_snrs, G_SNR_THRESHOLD),
                np.full(len(win_times), G_SNR_THRESHOLD)[::-1],
            ]),
            fill="toself",
            fillcolor="rgba(255,140,0,0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Low-SNR region",
            showlegend=True,
        ), row=2, col=1)

    # ── Layout ───────────────────────────────────────────────────────────────
    n_subharm  = int(sh_mask_model.sum())
    n_superharm= int(sp_mask_model.sum())
    n_low_snr  = int(low_snr_mask.sum())
    total      = len(starts)

    model_mae = float(np.nanmean(np.abs(model_hrs - gt_hrs)))
    chrom_mae = float(np.nanmean(np.abs(chrom_hrs - gt_hrs)))

    fig.update_layout(
        title=(
            f"<b>HR Trajectory | {dataset_name} | {split_name} | {seq_name}</b><br>"
            f"<span style='font-size:11px'>"
            f"Model MAE={model_mae:.2f} BPM | CHROM MAE={chrom_mae:.2f} BPM | "
            f"Sub-harmonic failures: model={n_subharm}, CHROM={int(sh_mask_chrom.sum())} | "
            f"Low G-SNR windows: {n_low_snr}/{total} ({100*n_low_snr/max(total,1):.1f}%)"
            f"</span>"
        ),
        template="plotly_white",
        width=1600, height=700,
        legend=dict(orientation="v", x=1.01, y=1.0),
        margin=dict(l=80, r=200, t=100, b=80),
        font=dict(size=11),
    )
    fig.update_yaxes(title_text="HR (BPM)",   row=1, col=1)
    fig.update_yaxes(title_text="G SNR",      row=2, col=1)
    fig.update_xaxes(title_text="Window start time (s)", row=2, col=1)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs=True)


# ── Remaining visualisation helpers (unchanged) ───────────────────────────────

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
    fig.add_trace(go.Bar(x=x_ids, y=df_plot["pred_gt_hr_mae"],  name="Model",
                          marker_color=colors_pred,  customdata=custom_data,
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
        margin=dict(l=80, r=40, t=120, b=180), font=dict(size=11),
    )
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
        template="plotly_white", width=700, height=700,
    )
    fig.write_html(str(out_html), include_plotlyjs=True)


def make_signal_comparison_slider(
    npz_path: Path,
    model: nn.Module,
    out_html: Path,
    roi_index,
    window_s: float,
    stride_s: float,
    dataset_name: str,
    split_name: str,
):
    X, Y, t, meta = load_subject_npz(npz_path)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    seq_name = str(meta.get("seq", npz_path.stem))
    fs       = infer_fs(meta, t)
    X_roi    = select_roi_rgb_np(X, roi_index)
    win_len  = int(round(window_s * fs))
    step_len = int(round(stride_s * fs))

    if len(X_roi) < win_len:
        return

    starts    = list(range(0, len(X_roi) - win_len + 1, step_len))
    n_windows = len(starts)
    fig       = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35],
                               subplot_titles=("BVP Waveforms", "Frequency Spectra"),
                               vertical_spacing=0.12)
    slider_steps = []
    title_text   = f"<b>{dataset_name} | {split_name} | {seq_name}</b>"

    for w_idx, s in enumerate(starts):
        e       = s + win_len
        rgb_w   = X_roi[s:e]
        gt_w    = Y[s:e]
        t_w     = t[s:e]
        pred_w  = predict_window(model, rgb_w, fs=fs)           # Fix 1
        chrom_w = chrom_signal(rgb_w, fs)

        pred_bpm  = HR_Function(pred_w,  fs, BPM_MIN, BPM_MAX)
        gt_bpm    = HR_Function(gt_w,    fs, BPM_MIN, BPM_MAX_GT)
        chrom_bpm = HR_Function(chrom_w, fs, BPM_MIN, BPM_MAX)

        pred_mae  = abs(pred_bpm  - gt_bpm) if np.isfinite(pred_bpm)  and np.isfinite(gt_bpm) else np.nan
        chrom_mae = abs(chrom_bpm - gt_bpm) if np.isfinite(chrom_bpm) and np.isfinite(gt_bpm) else np.nan
        pred_corr = compute_pearson(pred_w, gt_w)
        chrom_corr= compute_pearson(chrom_w, gt_w)
        visible   = (w_idx == 0)

        fig.add_trace(go.Scatter(x=t_w, y=zscore_np(gt_w),    mode="lines",
            name=f"GT | HR={gt_bpm:.1f} BPM",
            line=dict(color="#2E86AB", width=2), visible=visible, legendgroup="gt"),
            row=1, col=1)
        fig.add_trace(go.Scatter(x=t_w, y=zscore_np(pred_w),  mode="lines",
            name=f"Model | HR={pred_bpm:.1f} | MAE={pred_mae:.2f} | r={pred_corr:.3f}",
            line=dict(color="#F18F01", width=2), visible=visible, legendgroup="model"),
            row=1, col=1)
        fig.add_trace(go.Scatter(x=t_w, y=zscore_np(chrom_w), mode="lines",
            name=f"CHROM | HR={chrom_bpm:.1f} | MAE={chrom_mae:.2f} | r={chrom_corr:.3f}",
            line=dict(color="#A23B72", width=2, dash="dot"), visible=visible, legendgroup="chrom"),
            row=1, col=1)

        freqs    = np.fft.rfftfreq(len(gt_w), d=1.0 / fs)
        bpm_f    = freqs * 60.0
        mask     = (bpm_f >= BPM_MIN) & (bpm_f <= BPM_MAX)
        gt_fft   = np.abs(np.fft.rfft(gt_w    - np.mean(gt_w)))   ** 2
        pred_fft = np.abs(np.fft.rfft(pred_w  - np.mean(pred_w))) ** 2
        chrom_fft= np.abs(np.fft.rfft(chrom_w - np.mean(chrom_w)))** 2

        fig.add_trace(go.Scatter(x=bpm_f[mask], y=gt_fft[mask]/gt_fft[mask].max(),
            mode="lines", name="GT Spectrum", line=dict(color="#2E86AB", width=2),
            visible=visible, showlegend=False, legendgroup="gt"), row=2, col=1)
        fig.add_trace(go.Scatter(x=bpm_f[mask], y=pred_fft[mask]/pred_fft[mask].max(),
            mode="lines", name="Model Spectrum", line=dict(color="#F18F01", width=2),
            visible=visible, showlegend=False, legendgroup="model"), row=2, col=1)
        fig.add_trace(go.Scatter(x=bpm_f[mask], y=chrom_fft[mask]/chrom_fft[mask].max(),
            mode="lines", name="CHROM Spectrum", line=dict(color="#A23B72", width=2, dash="dot"),
            visible=visible, showlegend=False, legendgroup="chrom"), row=2, col=1)

        visibility = [False] * (6 * n_windows)
        for offset in range(6):
            visibility[6 * w_idx + offset] = True
        slider_steps.append(dict(
            method="update",
            args=[{"visible": visibility}, {"title": title_text}],
            label=f"Win {w_idx}",
        ))

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Normalised Amplitude", row=1, col=1)
    fig.update_xaxes(title_text="Heart Rate (BPM)", row=2, col=1)
    fig.update_yaxes(title_text="Normalised Power", row=2, col=1)
    fig.update_layout(
        title=title_text, template="plotly_white", width=1600, height=900,
        sliders=[dict(active=0, currentvalue={"prefix": "Window: "},
                      pad={"t": 50}, steps=slider_steps)],
        legend=dict(orientation="v", xanchor="left", x=1.01, y=1.0),
        margin=dict(l=80, r=180, t=100, b=100), font=dict(size=11),
    )
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
    fig.add_trace(go.Bar(x=x_ids, y=df_plot["pred_gt_pearson"],  name="Model", marker_color=colors))
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
        margin=dict(l=80, r=40, t=100, b=180),
    )
    fig.write_html(str(out_html), include_plotlyjs=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    out_dirs = make_output_dirs()

    print(f"Loading model from: {CKPT_PATH}")
    model, ckpt = load_model(CKPT_PATH)
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    df = pd.read_csv(MANIFEST_PATH,
                     dtype={"split": str, "path": str, "seq": str, "subject_id": str})
    for col in ("split", "seq", "subject_id"):
        if col not in df.columns:
            df[col] = "unknown"

    rows, all_window_rows, all_psd_peak_rows = [], [], []
    all_deep_rows = []
    print(f"\nEvaluating {len(df)} videos...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating", ncols=100):
        rel_or_abs_path = row["path"]
        npz_path = resolve_npz_path(rel_or_abs_path, MANIFEST_PATH)

        try:
            _, _, _, meta_tmp = load_subject_npz(npz_path)
            dataset_name = infer_dataset_name(npz_path, meta_tmp, row)

            result = evaluate_one_subject(
                npz_path=npz_path, model=model,
                roi_index=ROI_INDEX, window_s=WINDOW_S, stride_s=STRIDE_S,
            )

            seq_str   = str(row["seq"])
            seq_name  = seq_str if seq_str not in ("", "nan", "None") else result["seq"]
            split_name= str(row["split"])
            window_df = result.pop("window_df", pd.DataFrame())

            if len(window_df) > 0:
                window_df["dataset"]       = dataset_name
                window_df["split"]         = split_name
                window_df["seq"]           = seq_name
                window_df["subject_id"]    = str(row["subject_id"])
                window_df["path"]          = str(rel_or_abs_path)
                window_df["resolved_path"] = str(npz_path)
                window_csv = (out_dirs["tables"] /
                              f"{safe_name(dataset_name)}_{safe_name(split_name)}"
                              f"_{safe_name(seq_name)}_WINDOW_ERRORS.csv")
                #window_df.to_csv(window_csv, index=False)
                all_window_rows.append(window_df)

            result.update({
                "dataset": dataset_name, "split": split_name,
                "seq": seq_name, "subject_id": str(row["subject_id"]),
                "path": str(rel_or_abs_path), "resolved_path": str(npz_path),
            })
            rows.append(result)

            if result["n_windows"] > 0:
                # Signal comparison slider
                make_signal_comparison_slider(
                    npz_path=npz_path, model=model,
                    out_html=out_dirs["signals"] /
                        f"{safe_name(dataset_name)}_{safe_name(split_name)}_{safe_name(seq_name)}.html",
                    roi_index=ROI_INDEX, window_s=WINDOW_S, stride_s=STRIDE_S,
                    dataset_name=dataset_name, split_name=split_name,
                )

                # PSD diagnostic slider (Fix 2 applied)
                make_psd_diagnostic_slider(
                    npz_path=npz_path, model=model,
                    out_html=out_dirs["diagnostics"] /
                        f"{safe_name(dataset_name)}_{safe_name(split_name)}_{safe_name(seq_name)}_PSD.html",
                    roi_index=ROI_INDEX, window_s=WINDOW_S, stride_s=STRIDE_S,
                    dataset_name=dataset_name, split_name=split_name,
                )

                # Fix 3 — HR trajectory plot (new)
                # make_hr_trajectory_plot(
                #     npz_path=npz_path, model=model,
                #     out_html=out_dirs["trajectories"] /
                #         f"{safe_name(dataset_name)}_{safe_name(split_name)}_{safe_name(seq_name)}_TRAJ.html",
                #     roi_index=ROI_INDEX, window_s=WINDOW_S, stride_s=STRIDE_S,
                #     dataset_name=dataset_name, split_name=split_name,
                # )

                # PSD peak collection (Fix 4 applied)
                all_psd_peak_rows.extend(
                    collect_psd_top_peaks_for_subject(
                        npz_path=npz_path, model=model,
                        roi_index=ROI_INDEX, window_s=WINDOW_S, stride_s=STRIDE_S,
                        dataset_name=dataset_name, split_name=split_name,
                        seq_name=seq_name, subject_id=str(row["subject_id"]),
                    )
                )

                all_deep_rows.extend(
                    collect_deep_features_for_subject(
                        npz_path=npz_path, model=model,
                        roi_index=ROI_INDEX, window_s=WINDOW_S, stride_s=STRIDE_S,
                        dataset_name=dataset_name, split_name=split_name,
                        seq_name=seq_name, subject_id=str(row["subject_id"]),
                    )
                )

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
    full_df  = pd.DataFrame(rows)
    all_csv  = out_dirs["tables"] / "ALL_EVAL_RESULTS.csv"
    full_df.to_csv(all_csv, index=False)

    if all_window_rows:
        all_window_df = pd.concat(all_window_rows, ignore_index=True)
        all_window_df.to_csv(out_dirs["tables"] / "ALL_WINDOW_ERRORS.csv", index=False)
        (all_window_df
         .sort_values("model_minus_chrom", ascending=False)
         .to_csv(out_dirs["tables"] / "BAD_WINDOWS_MODEL_WORSE_THAN_CHROM.csv", index=False))

    if all_psd_peak_rows:
        psd_df = pd.DataFrame(all_psd_peak_rows)
        psd_df.to_csv(out_dirs["tables"] / "PSD_TOP_PEAKS_SUMMARY.csv", index=False)

    if all_deep_rows:
        print(f"Saving deep feature summaries for {len(all_deep_rows)} windows...")
        deep_df = pd.DataFrame(all_deep_rows)
        deep_df.to_csv(out_dirs["tables"] / "DEEP_FEATURES_SUMMARY.csv", index=False)
        write_deep_analysis(deep_df, out_dirs["tables"], out_dirs["diagnostics"])

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SPLIT-WISE SUMMARY")
    print(f"{'='*70}\n")

    split_summary = (
        full_df.groupby("split", dropna=False)
        .agg(N=("seq","count"),
             model_mae=("pred_gt_hr_mae","mean"),
             chrom_mae=("chrom_gt_hr_mae","mean"),
             model_r=("pred_gt_pearson","mean"),
             chrom_r=("chrom_gt_pearson","mean"))
        .reset_index()
    )
    ds_split_summary = (
        full_df.groupby(["dataset","split"], dropna=False)
        .agg(N=("seq","count"),
             model_mae=("pred_gt_hr_mae","mean"),
             chrom_mae=("chrom_gt_hr_mae","mean"),
             model_r=("pred_gt_pearson","mean"),
             chrom_r=("chrom_gt_pearson","mean"))
        .reset_index()
    )

    for _, r in split_summary.iterrows():
        diff = r["chrom_mae"] - r["model_mae"]
        sign = "↓" if diff > 0 else "↑"
        print(f"Split: {r['split']}  N={int(r['N'])}")
        print(f"  Model: MAE={r['model_mae']:.3f} BPM | r={r['model_r']:.3f}")
        print(f"  CHROM: MAE={r['chrom_mae']:.3f} BPM | r={r['chrom_r']:.3f}")
        if np.isfinite(diff):
            print(f"  Gap: {sign} {abs(diff):.3f} BPM")
        print()

    # ── Write summary.txt ─────────────────────────────────────────────────────
    txt_path = out_dirs["tables"] / "summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\nEVALUATION CONFIGURATION\n" + "="*80 + "\n\n")
        f.write(f"Checkpoint:    {CKPT_PATH}\n")
        f.write(f"Manifest:      {MANIFEST_PATH}\n")
        f.write(f"Window:        {WINDOW_S}s\n")
        f.write(f"Stride:        {STRIDE_S}s\n")
        f.write(f"HR Range (MODEL/CHROM): {BPM_MIN}-{BPM_MAX} BPM\n")
        f.write(f"HR Range (GT):          {BPM_MIN}-{BPM_MAX_GT} BPM\n")
        f.write(f"G_SNR threshold: {G_SNR_THRESHOLD}\n\n")
        f.write("="*80 + "\nSPLIT-WISE SUMMARY\n" + "="*80 + "\n\n")
        for _, r in split_summary.iterrows():
            f.write(f"Split: {r['split']}\n  N={int(r['N'])}\n")
            f.write(f"  Model: HR MAE={r['model_mae']:.3f} BPM | Pearson={r['model_r']:.3f}\n")
            f.write(f"  CHROM: HR MAE={r['chrom_mae']:.3f} BPM | Pearson={r['chrom_r']:.3f}\n\n")
        f.write("="*80 + "\nDATASET + SPLIT SUMMARY\n" + "="*80 + "\n\n")
        for _, r in ds_split_summary.iterrows():
            f.write(f"Dataset={r['dataset']} | Split={r['split']} | N={int(r['N'])}\n")
            f.write(f"  Model: HR MAE={r['model_mae']:.3f} | Pearson={r['model_r']:.3f}\n")
            f.write(f"  CHROM: HR MAE={r['chrom_mae']:.3f} | Pearson={r['chrom_r']:.3f}\n\n")

    # ── Per-dataset visualisations ────────────────────────────────────────────
    print("Generating visualisations...")
    for dataset_name, ddf in full_df.groupby("dataset"):
        make_hr_mae_comparison_bar(
            ddf,
            out_dirs["bars"] / f"{safe_name(dataset_name)}_hr_mae.html",
            f"HR MAE Comparison — {dataset_name}",
        )
        make_scatter_comparison(
            ddf,
            out_dirs["scatter"] / f"{safe_name(dataset_name)}_scatter.html",
            f"Model vs CHROM — {dataset_name}",
        )
        make_correlation_comparison(
            ddf,
            out_dirs["corr"] / f"{safe_name(dataset_name)}_correlation.html",
            f"Waveform Correlation — {dataset_name}",
        )

    print(f"\n{'='*70}")
    print(f"Evaluation complete.  Results in: {out_dirs['base']}")
    print(f"New output folder — trajectories: {out_dirs['trajectories']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

    