"""
mega_eval_additions.py — additions to your evaluate.py
─────────────────────────────────────────────────────────────────────────────
ADD-ONLY. Does not modify or remove any existing function.

Integration (3 small edits to your evaluate.py):

  1) at the top, add:
         from deep_features import deep_window_features
         from mega_eval_additions import (collect_deep_features_for_subject,
                                           write_deep_analysis)

  2) inside main(), create a list before the loop:
         all_deep_rows = []

     and inside the `if result["n_windows"] > 0:` block, after the existing
     collect_psd_top_peaks_for_subject(...) call, add:

         all_deep_rows.extend(
             collect_deep_features_for_subject(
                 npz_path=npz_path, model=model,
                 roi_index=ROI_INDEX, window_s=WINDOW_S, stride_s=STRIDE_S,
                 dataset_name=dataset_name, split_name=split_name,
                 seq_name=seq_name, subject_id=str(row["subject_id"]),
             )
         )

  3) after the existing PSD_TOP_PEAKS_SUMMARY.csv save block, add:

         if all_deep_rows:
             deep_df = pd.DataFrame(all_deep_rows)
             deep_csv = out_dirs["tables"] / "DEEP_FEATURES_SUMMARY.csv"
             deep_df.to_csv(deep_csv, index=False)
             write_deep_analysis(deep_df, out_dirs["tables"], out_dirs["diagnostics"])

Everything else in your file stays exactly as-is.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

# these imports mirror what evaluate.py already exposes
from deep_features import deep_window_features


def _failure_type(pred_bpm, gt_bpm):
    if not (np.isfinite(pred_bpm) and np.isfinite(gt_bpm)): return "nan"
    err = abs(pred_bpm - gt_bpm); ratio = gt_bpm/pred_bpm if pred_bpm>0 else 0
    if err < 4.0: return "correct"
    if 1.7 < ratio < 2.3: return "sub_harm_half"
    if 2.5 < ratio < 3.5: return "sub_harm_third"
    if 0.3 < ratio < 0.6: return "super_harm_2x"
    if 0.6 < ratio < 0.8: return "super_harm_1p5x"
    if err <= 7.5: return "1bin"
    if err <= 15.0: return "2bin"
    if err <= 22.5: return "3bin"
    return "large_error"


def collect_deep_features_for_subject(
    npz_path, model, roi_index, window_s, stride_s,
    dataset_name, split_name, seq_name, subject_id,
):
    """Mirror of collect_psd_top_peaks_for_subject but emits the 153-feature deep record."""
    # import here to reuse evaluate.py's exact signal pipeline
    from eval import (load_subject_npz, select_roi_rgb_np, infer_fs,
                          predict_window, chrom_signal)

    X, Y, t, meta = load_subject_npz(npz_path)
    fs = infer_fs(meta, t)
    X_roi = select_roi_rgb_np(X, roi_index)
    win_len = int(round(window_s*fs)); step = int(round(stride_s*fs))
    starts = list(range(0, len(Y)-win_len+1, step))
    rows = []
    for w_idx, s in enumerate(starts):
        e = s+win_len
        rgb_w = X_roi[s:e]; gt_w = Y[s:e]
        model_w = predict_window(model, rgb_w, fs=fs)
        chrom_w = chrom_signal(rgb_w, fs)

        feat = deep_window_features(rgb_w, gt_w, model_w, chrom_w, fs)
        feat.update({
            "dataset": dataset_name, "split": split_name,
            "seq": seq_name, "subject_id": subject_id,
            "window_idx": w_idx, "fs": fs,
        })
        feat["model_failure_type"] = _failure_type(feat["model_bpm"], feat["gt_bpm"])
        feat["chrom_failure_type"] = _failure_type(feat["chrom_bpm"], feat["gt_bpm"])
        rows.append(feat)
    return rows


def write_deep_analysis(deep_df: pd.DataFrame, tables_dir: Path, diag_dir: Path):
    """
    Produce a focused analysis CSV + an HTML visualization that tests the
    key hypotheses (phase coherence separates correct vs error; recoverability).
    """
    df = deep_df.copy()
    mft = df["model_failure_type"]

    groups = {
        "correct": mft=="correct",
        "1bin":    mft=="1bin",
        "2bin":    mft=="2bin",
        "sub":     mft.isin(["sub_harm_half","sub_harm_third"]),
    }

    metrics = [
        "phase_coh_at_GT","phase_coh_at_wrong","phase_coh_at_half","phase_coh_at_2x",
        "model_top1_top2_ratio","gt_recoverable",
        "G_snr","R_snr","B_snr",
        "G_entropy","G_flatness","G_if_std",
        "G_pow_at_GT","G_pow_at_wrong","R_pow_at_wrong","B_pow_at_wrong",
        "wrong_in_all_raw","gt_in_all_raw","channels_agree_top1",
    ]
    summary = []
    for gname, gmask in groups.items():
        sub = df[gmask]
        rec = {"group": gname, "N": int(len(sub))}
        for m in metrics:
            if m in sub.columns:
                rec[m] = float(np.nanmedian(sub[m].values))
        summary.append(rec)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(tables_dir / "DEEP_ANALYSIS_BY_ERRORTYPE.csv", index=False)

    # ── HTML viz ──────────────────────────────────────────────────────────────
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        key_metrics = ["phase_coh_at_GT","phase_coh_at_wrong",
                       "model_top1_top2_ratio","gt_recoverable",
                       "G_snr","G_if_std"]
        titles = ["Phase coherence @ GT","Phase coherence @ wrong peak",
                  "MODEL top1/top2 power","True HR recoverable (frac)",
                  "G SNR","G inst-freq std (stability)"]
        fig = make_subplots(rows=2, cols=3, subplot_titles=titles)
        order = ["correct","1bin","2bin","sub"]
        colors = {"correct":"#1D9E75","1bin":"#6BAF92","2bin":"#EF9F27","sub":"#E24B4A"}
        for i, m in enumerate(key_metrics):
            r, c = i//3+1, i%3+1
            xs, ys, cs = [], [], []
            for g in order:
                sub = df[groups[g]]
                if m in sub.columns and len(sub):
                    xs.append(g); ys.append(float(np.nanmedian(sub[m].values))); cs.append(colors[g])
            fig.add_trace(go.Bar(x=xs, y=ys, marker_color=cs, showlegend=False), row=r, col=c)
        fig.update_layout(title="<b>Deep feature medians by error type — does any metric separate correct from error?</b>",
                          template="plotly_white", width=1400, height=720)
        fig.write_html(str(diag_dir / "DEEP_FEATURE_SEPARATION.html"), include_plotlyjs=True)
    except Exception as ex:
        print("viz skipped:", ex)

    print(f"[deep] wrote DEEP_FEATURES_SUMMARY.csv ({len(df)} rows, {df.shape[1]} cols)")
    print(f"[deep] wrote DEEP_ANALYSIS_BY_ERRORTYPE.csv")
    print(f"[deep] wrote diagnostics/DEEP_FEATURE_SEPARATION.html")
