"""
deep_features.py — deep per-window feature extraction for rPPG error analysis
─────────────────────────────────────────────────────────────────────────────
This module ADDS to evaluate.py without changing any existing behaviour.
It computes, for every window, a rich feature record covering:

  • full top-5 spectral peaks (bpm + power) for R, G, B, GT, MODEL, CHROM
  • PHASE at each channel's peaks + CROSS-CHANNEL PHASE COHERENCE
    (the key untested signal — pulse is phase-coherent, noise is not)
  • per-channel power AT the GT freq, at the model-wrong freq,
    at GT/2 (sub-harmonic) and at 2*GT (harmonic)
  • spectral flatness / entropy per channel (signal-quality)
  • instantaneous-frequency stability via Hilbert (per channel)
  • channel-agreement and common-mode flags
  • recoverability flags (is true HR the model's top2/top3)

Drop this file next to evaluate.py and import collect_deep_features_for_subject.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

BPM_MIN = 40.0
BPM_MAX = 180.0
TOL_BPM = 4.0


# ── small helpers (self-contained; do not depend on evaluate.py) ──────────────

def _bandpass(sig, fs, low=0.67, high=3.0, order=3):
    sig = np.asarray(sig, float).reshape(-1)
    if len(sig) < order*3+1: return sig.copy()
    nyq = 0.5*fs; lo, hi = low/nyq, high/nyq
    if not (0 < lo < hi < 1): return sig.copy()
    b, a = butter(order, [lo, hi], btype="band")
    try: return filtfilt(b, a, sig)
    except Exception: return sig.copy()

def _norm(ch):
    return ch / (np.mean(ch) + 1e-8)

def _spectrum(sig, fs):
    """Return bpm axis, power, and complex spectrum within HR band."""
    sig = np.asarray(sig, float).reshape(-1)
    sig = sig - np.mean(sig)
    win = np.hanning(len(sig))
    comp = np.fft.rfft(sig*win)
    freqs = np.fft.rfftfreq(len(sig), d=1.0/fs)
    bpm = freqs*60.0
    mask = (bpm >= BPM_MIN) & (bpm <= BPM_MAX)
    return bpm[mask], (np.abs(comp)**2)[mask], comp[mask], freqs[mask]

def _top_peaks(bpm, power, k=5):
    """Top-k by power (local maxima preferred, fallback to argsort)."""
    idx = []
    for i in range(len(power)):
        lo = power[i-1] if i>0 else -1
        hi = power[i+1] if i<len(power)-1 else -1
        if power[i] >= lo and power[i] >= hi:
            idx.append(i)
    if not idx:
        idx = list(range(len(power)))
    idx = sorted(idx, key=lambda i: power[i], reverse=True)[:k]
    pmax = power.max() + 1e-12
    return [(float(bpm[i]), float(power[i]/pmax), int(i)) for i in idx]

def _power_at(bpm, power, target_bpm, tol=TOL_BPM):
    """Normalised power at the bin nearest target_bpm (0 if out of range)."""
    if not np.isfinite(target_bpm): return 0.0
    j = np.argmin(np.abs(bpm - target_bpm))
    if abs(bpm[j] - target_bpm) > tol*2: return 0.0
    return float(power[j] / (power.max() + 1e-12))

def _phase_at(bpm, comp, target_bpm):
    """Phase (radians) of the complex spectrum at the bin nearest target_bpm."""
    if not np.isfinite(target_bpm) or len(bpm)==0: return np.nan
    j = np.argmin(np.abs(bpm - target_bpm))
    return float(np.angle(comp[j]))

def _spectral_entropy(power):
    p = power / (power.sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p*np.log(p)) / (np.log(len(p)) + 1e-12)) if len(p)>1 else 0.0

def _spectral_flatness(power):
    p = power + 1e-12
    gm = np.exp(np.mean(np.log(p)))
    am = np.mean(p)
    return float(gm/am)

def _inst_freq_std(sig, fs):
    """Std of Hilbert instantaneous frequency in BPM — low = stable rhythm."""
    sig = np.asarray(sig, float)
    sig = _bandpass(sig - np.mean(sig), fs)
    if len(sig) < 8: return np.nan
    a = hilbert(sig)
    ph = np.unwrap(np.angle(a))
    inst = np.diff(ph) * fs / (2*np.pi) * 60.0  # BPM
    inst = inst[(inst>=BPM_MIN) & (inst<=BPM_MAX)]
    return float(np.std(inst)) if len(inst)>2 else np.nan

def _peak_bpm(bpm, power):
    return float(bpm[np.argmax(power)]) if len(power) else float("nan")


# ── the main per-window deep feature builder ──────────────────────────────────

def deep_window_features(rgb_w, gt_w, model_w, chrom_w, fs):
    """
    rgb_w  : [T,3] raw RGB window
    gt_w   : [T]   ground-truth BVP
    model_w: [T]   model output BVP (already bandpassed)
    chrom_w: [T]   CHROM output BVP
    Returns a flat dict of deep features for this window.
    """
    R, G, B = _norm(rgb_w[:,0]), _norm(rgb_w[:,1]), _norm(rgb_w[:,2])

    sigs = {"R":R, "G":G, "B":B, "GT":gt_w, "MODEL":model_w, "CHROM":chrom_w}
    spec = {}
    for name, s in sigs.items():
        bpm, pw, comp, fr = _spectrum(s, fs)
        spec[name] = (bpm, pw, comp)

    row = {}

    # peak frequencies
    bpm_gt    = _peak_bpm(*spec["GT"][:2])
    bpm_model = _peak_bpm(*spec["MODEL"][:2])
    bpm_chrom = _peak_bpm(*spec["CHROM"][:2])
    row["gt_bpm"], row["model_bpm"], row["chrom_bpm"] = bpm_gt, bpm_model, bpm_chrom

    # ── top-5 peaks + phase for every signal ──────────────────────────────────
    for name in sigs:
        bpm, pw, comp = spec[name]
        peaks = _top_peaks(bpm, pw, k=5)
        for r, (pb, pp, pi) in enumerate(peaks, 1):
            row[f"{name}_top{r}_bpm"]   = pb
            row[f"{name}_top{r}_power"] = pp
            row[f"{name}_top{r}_phase"] = float(np.angle(comp[pi]))
        for r in range(len(peaks)+1, 6):  # pad
            row[f"{name}_top{r}_bpm"]=np.nan; row[f"{name}_top{r}_power"]=np.nan; row[f"{name}_top{r}_phase"]=np.nan

    # ── per-channel power AT key frequencies (GT, wrong, GT/2, 2*GT) ──────────
    for name in ["R","G","B","MODEL","CHROM"]:
        bpm, pw, comp = spec[name]
        row[f"{name}_pow_at_GT"]    = _power_at(bpm, pw, bpm_gt)
        row[f"{name}_pow_at_wrong"] = _power_at(bpm, pw, bpm_model)
        row[f"{name}_pow_at_half"]  = _power_at(bpm, pw, bpm_gt/2.0)
        row[f"{name}_pow_at_2x"]    = _power_at(bpm, pw, bpm_gt*2.0)

    # ── CROSS-CHANNEL PHASE COHERENCE (the key new signal) ───────────────────
    # phase of R,G,B at a given freq; coherence = how aligned they are.
    def coherence_at(target):
        ph = [ _phase_at(spec[c][0], spec[c][2], target) for c in ["R","G","B"] ]
        ph = [p for p in ph if np.isfinite(p)]
        if len(ph) < 2: return np.nan
        # circular concentration: |mean(e^{i*phase})| in [0,1], 1=perfectly aligned
        v = np.mean([np.exp(1j*p) for p in ph])
        return float(np.abs(v))
    row["phase_coh_at_GT"]    = coherence_at(bpm_gt)
    row["phase_coh_at_wrong"] = coherence_at(bpm_model)
    row["phase_coh_at_half"]  = coherence_at(bpm_gt/2.0)
    row["phase_coh_at_2x"]    = coherence_at(bpm_gt*2.0)
    # pairwise phase differences at GT (R-G, G-B, R-B)
    pR = _phase_at(spec["R"][0], spec["R"][2], bpm_gt)
    pG = _phase_at(spec["G"][0], spec["G"][2], bpm_gt)
    pB = _phase_at(spec["B"][0], spec["B"][2], bpm_gt)
    def wrap(d): return float(np.arctan2(np.sin(d), np.cos(d)))
    row["dphase_RG_at_GT"] = wrap(pR-pG) if np.isfinite(pR) and np.isfinite(pG) else np.nan
    row["dphase_GB_at_GT"] = wrap(pG-pB) if np.isfinite(pG) and np.isfinite(pB) else np.nan
    row["dphase_RB_at_GT"] = wrap(pR-pB) if np.isfinite(pR) and np.isfinite(pB) else np.nan

    # ── spectral quality per channel ─────────────────────────────────────────
    for name in ["R","G","B","MODEL","CHROM"]:
        bpm, pw, comp = spec[name]
        row[f"{name}_entropy"]  = _spectral_entropy(pw)
        row[f"{name}_flatness"] = _spectral_flatness(pw)
        row[f"{name}_snr"]      = float(_top_peaks(bpm,pw,2)[0][1] /
                                        (_top_peaks(bpm,pw,2)[1][1] if len(_top_peaks(bpm,pw,2))>1 else 1.0))
        row[f"{name}_if_std"]   = _inst_freq_std(sigs[name], fs)

    # ── recoverability & structure flags (the proven hypothesis) ──────────────
    def near(a,b): return np.isfinite(a) and np.isfinite(b) and abs(a-b) < TOL_BPM
    m_t2 = row.get("MODEL_top2_bpm", np.nan)
    m_t3 = row.get("MODEL_top3_bpm", np.nan)
    row["gt_is_model_top2"] = int(near(m_t2, bpm_gt))
    row["gt_is_model_top3"] = int(near(m_t3, bpm_gt))
    row["gt_recoverable"]   = int(near(m_t2, bpm_gt) or near(m_t3, bpm_gt))
    row["model_top1_top2_ratio"] = float(row.get("MODEL_top1_power",1.0) /
                                          (row.get("MODEL_top2_power",1.0)+1e-9))

    # channel agreement / common-mode flags
    R1,G1,B1 = row["R_top1_bpm"], row["G_top1_bpm"], row["B_top1_bpm"]
    row["channels_agree_top1"] = int(near(R1,G1) and near(G1,B1))
    row["wrong_in_all_raw"]    = int(all(
        any(near(row[f"{c}_top{k}_bpm"], bpm_model) for k in [1,2,3,4,5]) for c in ["R","G","B"]))
    row["gt_in_all_raw"]       = int(all(
        any(near(row[f"{c}_top{k}_bpm"], bpm_gt) for k in [1,2,3,4,5]) for c in ["R","G","B"]))
    for c in ["R","G","B"]:
        row[f"{c}_has_GT"]    = int(any(near(row[f"{c}_top{k}_bpm"], bpm_gt) for k in [1,2,3,4,5]))
        row[f"{c}_has_wrong"] = int(any(near(row[f"{c}_top{k}_bpm"], bpm_model) for k in [1,2,3,4,5]))

    return row
