# RUN_validate_npz_time_alignment.py
# Example:
# python RUN_validate_npz_time_alignment.py --npz_dir /media/data/rPPG/Code/GitHub/Project_rPPG/Dataset/3_ROI/PURE_RAW
# python RUN_validate_npz_time_alignment.py --npz_dir /media/data/rPPG/Code/GitHub/Project_rPPG/Dataset/3_ROI/UBFC_RAW
# python RUN_validate_npz_time_alignment.py --npz_dir /media/data/rPPG/Code/GitHub/Project_rPPG/Dataset/3_ROI/BH_RAW
# python RUN_validate_npz_time_alignment.py --npz_dir /media/data/rPPG/Code/GitHub/Project_rPPG/Dataset/3_ROI/COHFACE_RAW
# python RUN_validate_npz_time_alignment.py --npz_dir /media/data/rPPG/Code/GitHub/Project_rPPG/Dataset/3_ROI/TokyoTech_RAW

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Small helpers
# =========================================================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_json_maybe(x):
    try:
        if isinstance(x, bytes):
            x = x.decode("utf-8")
        if isinstance(x, np.ndarray):
            if x.shape == ():
                x = x.item()
            else:
                x = x.tolist()
        if isinstance(x, str):
            return json.loads(x)
    except Exception:
        pass
    return {}


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def robust_zscore(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    good = np.isfinite(x)
    y = np.zeros_like(x, dtype=np.float64)
    if np.sum(good) < 3:
        return y
    xf = x[good]
    med = np.median(xf)
    mad = np.median(np.abs(xf - med)) + 1e-8
    y[good] = (xf - med) / (1.4826 * mad + 1e-8)
    return y


def moving_average(x, win):
    x = np.asarray(x, dtype=np.float64)
    if win <= 1:
        return x.copy()
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(win, dtype=np.float64) / float(win)
    y = np.convolve(xp, ker, mode="same")
    return y[pad:pad + len(x)]


def detrend_ma(x, fs, sec=1.0):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) < 5 or not np.isfinite(fs) or fs <= 0:
        return x.copy()
    win = max(3, int(round(sec * fs)))
    if win % 2 == 0:
        win += 1
    trend = moving_average(x, win)
    return x - trend


def normalize_std(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    good = np.isfinite(x)
    y = np.zeros_like(x)
    if np.sum(good) < 3:
        return y
    xf = x[good]
    s = np.std(xf)
    if s < 1e-8:
        return y
    y[good] = (xf - np.mean(xf)) / s
    return y


def simple_bandpass_like(x, fs):
    """
    Very simple pulse-emphasis filter without scipy:
    high-pass-ish by detrending, then mild smoothing.
    This is only for alignment visualization, not for final science.
    """
    x1 = detrend_ma(x, fs=fs, sec=1.0)
    x2 = moving_average(x1, max(3, int(round(0.15 * fs)) | 1))
    return normalize_std(x2)


def estimate_fs_from_t(t):
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if len(t) < 3:
        return np.nan
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return np.nan
    return 1.0 / np.median(dt)


def crosscorr_best_lag(x, y, fs, max_lag_s=2.0):
    """
    Returns:
        best_lag_sec
        peak_corr
        zero_lag_corr
        lags_sec
        corr_curve
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]

    if len(x) < 10 or not np.isfinite(fs) or fs <= 0:
        return np.nan, np.nan, np.nan, np.array([]), np.array([])

    x = normalize_std(x)
    y = normalize_std(y)

    max_lag = int(round(max_lag_s * fs))
    lags = np.arange(-max_lag, max_lag + 1, dtype=int)
    corr = np.zeros(len(lags), dtype=np.float64)

    for i, lag in enumerate(lags):
        if lag < 0:
            xa = x[-lag:]
            ya = y[:len(xa)]
        elif lag > 0:
            xa = x[:-lag]
            ya = y[lag:]
        else:
            xa = x
            ya = y

        if len(xa) < 5:
            corr[i] = np.nan
            continue

        sx = np.std(xa)
        sy = np.std(ya)
        if sx < 1e-8 or sy < 1e-8:
            corr[i] = np.nan
            continue

        corr[i] = np.mean(normalize_std(xa) * normalize_std(ya))

    if np.all(~np.isfinite(corr)):
        return np.nan, np.nan, np.nan, lags / fs, corr

    idx = np.nanargmax(np.abs(corr))
    best_lag_sec = lags[idx] / fs
    peak_corr = corr[idx]

    zero_idx = np.where(lags == 0)[0]
    zero_lag_corr = corr[zero_idx[0]] if len(zero_idx) else np.nan

    return best_lag_sec, peak_corr, zero_lag_corr, lags / fs, corr


# =========================================================
# RGB pulse proxy
# =========================================================
def build_rgb_proxy_from_X(X):
    """
    X shape: [T, C] where C = 3*K
    We build a simple visual pulse proxy from ROI-averaged green.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < 3:
        return np.zeros(X.shape[0], dtype=np.float64), "invalid_X"

    C = X.shape[1]
    K = C // 3
    if K < 1:
        g = X[:, 1]
        return g, "single_green"

    greens = []
    for k in range(K):
        g = X[:, 3 * k + 1]
        greens.append(g)
    greens = np.stack(greens, axis=1)  # [T, K]

    proxy = np.nanmean(greens, axis=1)
    return proxy, "mean_green_over_rois"


# =========================================================
# One-file analysis
# =========================================================
def analyze_one_npz(npz_path: Path, out_plot_dir: Path, overlay_sec: float = 20.0):
    data = np.load(npz_path, allow_pickle=True)

    X = np.asarray(data["X"], dtype=np.float64)
    Y = np.asarray(data["Y"], dtype=np.float64).reshape(-1)
    t = np.asarray(data["t"], dtype=np.float64).reshape(-1)

    meta = load_json_maybe(data["meta"]) if "meta" in data else {}
    qc_in = load_json_maybe(data["qc"]) if "qc" in data else {}

    seq_name = npz_path.stem
    dataset = meta.get("dataset", "UNKNOWN")

    # ---------------------------
    # Basic consistency checks
    # ---------------------------
    T_x = int(X.shape[0]) if X.ndim >= 1 else 0
    T_y = int(len(Y))
    T_t = int(len(t))

    lengths_match = (T_x == T_y == T_t)

    t_monotonic = bool(np.all(np.diff(t) > 0)) if len(t) >= 2 else False

    if len(t) >= 3:
        dt = np.diff(t)
        dt_pos = dt[np.isfinite(dt) & (dt > 0)]
        dt_median = float(np.median(dt_pos)) if len(dt_pos) else np.nan
        dt_std = float(np.std(dt_pos)) if len(dt_pos) else np.nan
        fs_est = float(1.0 / dt_median) if np.isfinite(dt_median) and dt_median > 0 else np.nan
    else:
        dt_median = np.nan
        dt_std = np.nan
        fs_est = np.nan

    video_duration = float(t[-1] - t[0]) if len(t) >= 2 else np.nan
    gt_y_std = float(np.std(Y[np.isfinite(Y)])) if np.any(np.isfinite(Y)) else np.nan

    # ---------------------------
    # Proxy build
    # ---------------------------
    rgb_proxy_raw, proxy_name = build_rgb_proxy_from_X(X)

    # pulse-like versions for alignment check
    Yf = simple_bandpass_like(Y, fs_est) if np.isfinite(fs_est) else normalize_std(Y)
    Pf = simple_bandpass_like(rgb_proxy_raw, fs_est) if np.isfinite(fs_est) else normalize_std(rgb_proxy_raw)

    # ---------------------------
    # Lag estimation
    # ---------------------------
    best_lag_sec, peak_corr, zero_lag_corr, lag_axis, corr_curve = crosscorr_best_lag(
        Pf, Yf, fs=fs_est if np.isfinite(fs_est) else 30.0, max_lag_s=2.0
    )

    # simple pass/fail-like hint
    if not lengths_match or not t_monotonic:
        align_status = "FAIL_basic_time"
    elif not np.isfinite(best_lag_sec):
        align_status = "WARN_no_lag_estimate"
    elif abs(best_lag_sec) <= 0.10:
        align_status = "GOOD_near_zero_lag"
    elif abs(best_lag_sec) <= 0.30:
        align_status = "OK_small_lag"
    else:
        align_status = "WARN_large_lag"

    # ---------------------------
    # Plot
    # ---------------------------
    ensure_dir(out_plot_dir)
    plot_path = out_plot_dir / f"{seq_name}_alignment_check.png"

    fig = plt.figure(figsize=(14, 10))

    # 1) first window overlay
    ax1 = plt.subplot(3, 1, 1)
    if len(t) > 0:
        mask = t <= (t[0] + overlay_sec)
        tt = t[mask]
        yy = normalize_std(Y[mask])
        pp = normalize_std(rgb_proxy_raw[mask])
        ax1.plot(tt, yy, label="GT (normalized)", linewidth=1.5)
        ax1.plot(tt, pp, label=f"RGB proxy raw ({proxy_name})", linewidth=1.0, alpha=0.85)
    ax1.set_title(f"{dataset} | {seq_name} | First {overlay_sec:.0f}s visual check")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Normalized amplitude")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2) filtered full overlay
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(t[:len(Yf)], Yf[:len(t)], label="GT filtered", linewidth=1.2)
    ax2.plot(t[:len(Pf)], Pf[:len(t)], label="RGB proxy filtered", linewidth=1.0, alpha=0.85)
    ax2.set_title(
        f"Filtered full-sequence overlay | best_lag={best_lag_sec:.4f}s | "
        f"peak_corr={peak_corr:.4f} | zero_lag_corr={zero_lag_corr:.4f}"
    )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Filtered / standardized")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3) lag curve
    ax3 = plt.subplot(3, 1, 3)
    if len(lag_axis) > 0:
        ax3.plot(lag_axis, corr_curve, linewidth=1.5)
        ax3.axvline(0.0, linestyle="--", linewidth=1.0)
        if np.isfinite(best_lag_sec):
            ax3.axvline(best_lag_sec, linestyle="--", linewidth=1.0)
    ax3.set_title("Cross-correlation around zero lag")
    ax3.set_xlabel("Lag (s)  |  positive means GT is best matched after shifting")
    ax3.set_ylabel("Correlation")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------
    # Return row
    # ---------------------------
    row = {
        "file": npz_path.name,
        "dataset": dataset,
        "seq": seq_name,
        "status": align_status,
        "lengths_match": bool(lengths_match),
        "T_x": T_x,
        "T_y": T_y,
        "T_t": T_t,
        "t_monotonic": bool(t_monotonic),
        "dt_median_s": dt_median,
        "dt_std_s": dt_std,
        "fs_est": fs_est,
        "video_duration_s": video_duration,
        "gt_y_std": gt_y_std,
        "proxy_name": proxy_name,
        "best_lag_sec": best_lag_sec,
        "peak_corr": peak_corr,
        "zero_lag_corr": zero_lag_corr,
        "plot_path": str(plot_path),
    }

    # also carry selected meta/qc values if present
    for k in ["subj", "task", "state", "frag", "fps", "fs_gt", "sync_offset_s", "align_mode"]:
        if k in meta:
            row[f"meta_{k}"] = meta[k]

    for k in ["duration_ratio_gt_over_video", "fps_est_median", "face_fail_frac_est", "y_jump_frac"]:
        if k in qc_in:
            row[f"qc_{k}"] = qc_in[k]

    return row


# =========================================================
# HTML report
# =========================================================
def write_html_report(df: pd.DataFrame, out_html: Path, title: str):
    rows = []

    rows.append("<html><head><meta charset='utf-8'>")
    rows.append(f"<title>{title}</title>")
    rows.append("""
    <style>
        body { font-family: Arial, sans-serif; margin: 24px; }
        h1, h2 { margin-bottom: 8px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 24px; }
        th, td { border: 1px solid #cccccc; padding: 6px 8px; font-size: 13px; }
        th { background: #f2f2f2; }
        .good { background: #e8f7e8; }
        .warn { background: #fff6dd; }
        .fail { background: #fdeaea; }
        img { max-width: 1100px; width: 100%; border: 1px solid #ddd; margin-bottom: 28px; }
        .small { color: #555; font-size: 13px; }
    </style>
    """)
    rows.append("</head><body>")
    rows.append(f"<h1>{title}</h1>")
    rows.append("<p class='small'>This report checks internal NPZ time consistency and likely GT/RGB co-timing. It does not reconstruct hidden original timestamps outside the NPZ.</p>")

    # summary table
    rows.append("<h2>Summary</h2>")
    rows.append("<table>")
    cols = [
        "file", "dataset", "status", "lengths_match", "t_monotonic",
        "fs_est", "video_duration_s", "best_lag_sec", "peak_corr", "zero_lag_corr"
    ]
    rows.append("<tr>" + "".join([f"<th>{c}</th>" for c in cols]) + "</tr>")

    for _, r in df.iterrows():
        status = str(r.get("status", ""))
        cls = "good" if status.startswith("GOOD") else ("warn" if status.startswith("OK") or status.startswith("WARN") else "fail")
        rows.append("<tr class='%s'>" % cls)
        for c in cols:
            v = r.get(c, "")
            rows.append(f"<td>{v}</td>")
        rows.append("</tr>")
    rows.append("</table>")

    # plots
    rows.append("<h2>Plots</h2>")
    for _, r in df.iterrows():
        rows.append(f"<h3>{r.get('file','')}</h3>")
        rows.append(f"<p class='small'>status={r.get('status','')} | best_lag_sec={r.get('best_lag_sec', np.nan)} | peak_corr={r.get('peak_corr', np.nan)}</p>")
        plot_path = r.get("plot_path", "")
        if plot_path and os.path.exists(plot_path):
            rel = os.path.relpath(plot_path, out_html.parent)
            rows.append(f"<img src='{rel}' alt='{r.get('file','')}'>")

    rows.append("</body></html>")

    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", type=str, default="/media/data/rPPG/Code/GitHub/Project_rPPG/Dataset/3_ROI/UBFCPhys_RAW", help="Folder containing .npz files")
    parser.add_argument("--out_dir", type=str, default=None, help="Output report folder")
    parser.add_argument("--overlay_sec", type=float, default=20.0, help="Seconds for top overlay panel")
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir)
    if not npz_dir.exists():
        raise FileNotFoundError(f"npz_dir not found: {npz_dir}")

    if args.out_dir is None:
        out_dir = npz_dir / "_alignment_validation"
    else:
        out_dir = Path(args.out_dir)

    plot_dir = out_dir / "plots"
    ensure_dir(out_dir)
    ensure_dir(plot_dir)

    npz_files = sorted(npz_dir.glob("*.npz"))
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No .npz files found in: {npz_dir}")

    rows = []
    for npz_path in npz_files:
        try:
            row = analyze_one_npz(
                npz_path=npz_path,
                out_plot_dir=plot_dir,
                overlay_sec=args.overlay_sec
            )
            rows.append(row)
            print(f"[OK] {npz_path.name} | {row['status']} | lag={row['best_lag_sec']:.4f}s")
        except Exception as e:
            rows.append({
                "file": npz_path.name,
                "dataset": "UNKNOWN",
                "seq": npz_path.stem,
                "status": "FAIL_analysis_exception",
                "error": str(e),
            })
            print(f"[FAIL] {npz_path.name}: {e}")

    df = pd.DataFrame(rows)

    csv_path = out_dir / "alignment_report.csv"
    html_path = out_dir / "alignment_report.html"

    df.to_csv(csv_path, index=False)
    write_html_report(df, html_path, title=f"NPZ Time Alignment Validation: {npz_dir.name}")

    # quick summary
    n_total = len(df)
    n_good = int(df["status"].astype(str).str.startswith("GOOD").sum()) if "status" in df.columns else 0
    n_ok = int(df["status"].astype(str).str.startswith("OK").sum()) if "status" in df.columns else 0
    n_warn = int(df["status"].astype(str).str.startswith("WARN").sum()) if "status" in df.columns else 0
    n_fail = int(df["status"].astype(str).str.startswith("FAIL").sum()) if "status" in df.columns else 0

    print("\n==============================")
    print(f"Done: {npz_dir}")
    print(f"Total files : {n_total}")
    print(f"GOOD        : {n_good}")
    print(f"OK          : {n_ok}")
    print(f"WARN        : {n_warn}")
    print(f"FAIL        : {n_fail}")
    print(f"CSV report  : {csv_path}")
    print(f"HTML report : {html_path}")
    print("==============================")


if __name__ == "__main__":
    main()