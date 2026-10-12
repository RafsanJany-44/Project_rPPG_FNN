#!/usr/bin/env python3
"""
ROOT_DIR = must be the same root as used in the RESULTS_ROOT in lunch_train.py, e.g. /media/data/rPPG/Code/GitHub/Project_rPPG_Result/Result_Lab_1_Concentration_Margin_Test_PM1_with_Seed_123_456_789/
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — edit this path
# ═══════════════════════════════════════════════════════════════
ROOT_DIR = "/media/data/rPPG/Code/GitHub/Project_rPPG_Result/Result_Lab_1_Concentration_Margin_Test_PM1_with_Seed_123_456_789/"

# Protocol folder name → protocol key mapping
PROTOCOL_MAP = {
    "EVAL_PROTOCOL_old_1.0BPM": "old",
    "EVAL_PROTOCOL_prism":      "prism",
    "EVAL_PROTOCOL_toolbox":    "toolbox",
}

PROTOCOL_LABELS = {
    "old":     "Old@1BPM",
    "prism":   "PRISM",
    "toolbox": "Toolbox",
}

DATASETS = ["PURE", "TokyoTech", "UBFC"]

# ═══════════════════════════════════════════════════════════════
# STEP 1: Discover models and protocols
# ═══════════════════════════════════════════════════════════════
def discover_structure(root):
    """Walk ROOT and find all model/protocol/table combinations."""
    structure = {}  # model_name -> {proto_key -> tables_path}
    root = Path(root)

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        # Skip if it looks like a file or non-model folder
        model_name = model_dir.name
        if model_name.startswith(".") or model_name.startswith("__"):
            continue

        # Check for EVAL_PROTOCOL_* subfolders
        proto_found = {}
        for proto_dir in sorted(model_dir.iterdir()):
            if not proto_dir.is_dir():
                continue
            proto_folder = proto_dir.name
            if proto_folder in PROTOCOL_MAP:
                tables_dir = proto_dir / "tables"
                if tables_dir.is_dir():
                    proto_found[PROTOCOL_MAP[proto_folder]] = tables_dir

        if proto_found:
            structure[model_name] = proto_found

    return structure


# ═══════════════════════════════════════════════════════════════
# STEP 2: Error classification — CRITICAL ACCURACY
# ═══════════════════════════════════════════════════════════════
def classify_error_bin(abs_err):
    """Classify absolute error into bin categories."""
    if abs_err <= 1.0:
        return "1-bin (<=1)"
    elif abs_err <= 2.0:
        return "2-bin (1-2)"
    elif abs_err <= 3.0:
        return "3-bin (2-3)"
    elif abs_err <= 5.0:
        return "4-5 BPM"
    else:
        return ">5 BPM"


def classify_harmonic(gt_bpm, model_bpm, abs_err):
    """
    Classify harmonic error type for windows with >5 BPM error.
    Returns harmonic type string.

    Sub-harmonic:    predicted near GT/2 (half-frequency lock)
    Super-harm 2x:   predicted near GT*2 (double-frequency lock)
    Ratio 3:2:       predicted near GT*1.5
    Ratio 2:3:       predicted near GT*0.667
    """
    if abs_err <= 5.0:
        return "none"

    # Guard: NaN or invalid BPM
    if not np.isfinite(gt_bpm) or not np.isfinite(model_bpm):
        return "LARGE-UNKNOWN"
    if gt_bpm <= 0 or model_bpm <= 0:
        return "LARGE-UNKNOWN"

    # Check sub-harmonic: pred ≈ GT/2
    if abs(model_bpm - gt_bpm / 2.0) < 5.0:
        return "SUB-HARM"

    # Check super-harmonic 2×: pred ≈ GT*2
    if abs(model_bpm - gt_bpm * 2.0) < 5.0:
        return "SUP-HARM-2x"

    # Check 3:2 ratio: pred ≈ GT*1.5
    if abs(model_bpm - gt_bpm * 1.5) < 3.0:
        return "RATIO-3:2"

    # Check 2:3 ratio: pred ≈ GT*0.667
    if abs(model_bpm - gt_bpm * 0.667) < 3.0:
        return "RATIO-2:3"

    # Non-harmonic large error
    return "LARGE-NON-HARM"


def classify_direction(gt_bpm, model_bpm, abs_err):
    """Classify error direction for >5 BPM errors.
    Uses actual BPM values, not model_err sign, to avoid
    sign convention ambiguity."""
    if abs_err <= 5.0:
        return "OK"
    if model_bpm > gt_bpm:
        return "HIGH"   # model predicts above GT
    else:
        return "LOW"    # model predicts below GT


# ═══════════════════════════════════════════════════════════════
# STEP 3: Read and process all window error CSVs
# ═══════════════════════════════════════════════════════════════
def read_window_errors(tables_path, model_name, proto_key):
    """Read ALL_WINDOW_ERRORS.csv, add computed columns."""
    csv_path = tables_path / "ALL_WINDOW_ERRORS.csv"
    if not csv_path.exists():
        print(f"  [WARN] Missing: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    # Filter out 'unknown' dataset rows (known eval bug)
    df = df[df["dataset"] != "unknown"].copy()

    # Clean seq column — some CSVs have leading quote
    if "seq" in df.columns:
        df["seq"] = df["seq"].astype(str).str.lstrip("'")

    # Add model and protocol identifiers
    df["model"] = model_name
    df["protocol"] = proto_key
    df["protocol_label"] = PROTOCOL_LABELS.get(proto_key, proto_key)

    # ── Compute derived columns ──

    # Absolute errors
    df["abs_model_err"] = df["model_err"].abs()
    df["abs_chrom_err"] = df["chrom_err"].abs()

    # Error bin classification
    df["error_bin"] = df["abs_model_err"].apply(classify_error_bin)

    # Harmonic classification
    df["harmonic_type"] = df.apply(
        lambda r: classify_harmonic(r["gt_bpm"], r["model_bpm"], r["abs_model_err"]),
        axis=1
    )

    # Error direction (uses BPM values directly, not model_err sign)
    df["error_direction"] = df.apply(
        lambda r: classify_direction(r["gt_bpm"], r["model_bpm"], r["abs_model_err"]),
        axis=1
    )

    # Predicted / GT ratio
    df["pred_gt_ratio"] = np.where(
        df["gt_bpm"] > 0,
        (df["model_bpm"] / df["gt_bpm"]).round(4),
        np.nan
    )

    # Model beats CHROM flag
    df["is_model_better"] = df["abs_model_err"] < df["abs_chrom_err"]

    # ── Cross-validation: merge eval code's failure type from PSD file ──
    psd_path = tables_path / "PSD_TOP_PEAKS_SUMMARY.csv"
    if psd_path.exists():
        psd = pd.read_csv(psd_path)
        psd = psd[psd["dataset"] != "unknown"]
        if "model_failure_type" in psd.columns:
            # Clean seq column in PSD to match window errors
            if "seq" in psd.columns:
                psd["seq"] = psd["seq"].astype(str).str.lstrip("'")
            # Merge key MUST include seq to avoid Cartesian product
            # (same subject has multiple recordings, each with window_idx 0,1,2...)
            merge_cols = ["dataset", "split", "subject_id", "window_idx", "seq"]
            if all(c in psd.columns and c in df.columns for c in merge_cols):
                psd_slim = psd[merge_cols + ["model_failure_type"]].copy()
                psd_slim = psd_slim.rename(columns={"model_failure_type": "eval_failure_type"})
                before_len = len(df)
                df = df.merge(psd_slim, on=merge_cols, how="left")
                after_len = len(df)
                if after_len != before_len:
                    print(f"    [WARN] PSD merge changed row count: {before_len} -> {after_len}")
    if "eval_failure_type" not in df.columns:
        df["eval_failure_type"] = np.nan

    # ── Verify sign convention (printed during build, not stored) ──
    sample = df.head(5)
    for _, r in sample.iterrows():
        computed_diff = r["gt_bpm"] - r["model_bpm"]
        stored_err = r["model_err"]
        sign_match = abs(computed_diff - stored_err) < 0.01
        if not sign_match:
            alt_match = abs(computed_diff + stored_err) < 0.01
            if alt_match:
                print(f"    [INFO] model_err sign = pred-gt (positive = model predicts HIGH)")
            else:
                print(f"    [WARN] model_err sign convention unclear: "
                      f"gt={r['gt_bpm']:.1f} pred={r['model_bpm']:.1f} "
                      f"stored_err={stored_err:.3f} computed_diff={computed_diff:.3f}")
            break

    return df


def read_eval_results(tables_path, model_name, proto_key):
    """Read ALL_EVAL_RESULTS.csv (recording-level)."""
    csv_path = tables_path / "ALL_EVAL_RESULTS.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df = df[df["dataset"] != "unknown"].copy()
    if "seq" in df.columns:
        df["seq"] = df["seq"].astype(str).str.lstrip("'")
    df["model"] = model_name
    df["protocol"] = proto_key
    df["protocol_label"] = PROTOCOL_LABELS.get(proto_key, proto_key)
    return df


def read_summary_txt(tables_path):
    """Parse summary.txt for per-dataset MAE values."""
    txt_path = tables_path / "summary.txt"
    if not txt_path.exists():
        return {}
    text = txt_path.read_text()
    results = {}
    # Parse dataset+split blocks
    pattern = r"Dataset=(\w+)\s*\|\s*Split=(\w+)\s*\|\s*N=(\d+)\s*\n\s*Model:\s*HR MAE=([\d.]+)"
    for m in re.finditer(pattern, text):
        ds, split, n, mae = m.group(1), m.group(2), int(m.group(3)), float(m.group(4))
        results[(ds, split)] = {"mae": mae, "n": n}
    # Parse overall split MAE
    pattern2 = r"Split:\s*(\w+)\s*\n\s*N=(\d+)\s*\n\s*Model:\s*HR MAE=([\d.]+)"
    for m in re.finditer(pattern2, text):
        split, n, mae = m.group(1), int(m.group(2)), float(m.group(3))
        results[("ALL", split)] = {"mae": mae, "n": n}
    return results


# ═══════════════════════════════════════════════════════════════
# STEP 4: Build MEGA CSV
# ═══════════════════════════════════════════════════════════════
def build_mega_csv(structure, root):
    """Combine all window errors into one DataFrame."""
    all_frames = []

    for model_name, protos in structure.items():
        for proto_key, tables_path in protos.items():
            print(f"  Reading: {model_name} / {proto_key}")
            df = read_window_errors(tables_path, model_name, proto_key)
            if df is not None:
                all_frames.append(df)

    if not all_frames:
        print("[ERROR] No data found!")
        return pd.DataFrame()

    mega = pd.concat(all_frames, ignore_index=True)

    # Reorder columns for LLM readability
    priority_cols = [
        "model", "protocol", "protocol_label", "split", "dataset",
        "subject_id", "seq", "window_idx",
        "gt_bpm", "model_bpm", "chrom_bpm",
        "model_err", "chrom_err", "abs_model_err", "abs_chrom_err",
        "error_bin", "harmonic_type", "error_direction", "pred_gt_ratio",
        "is_model_better",
        "model_pearson", "chrom_pearson",
        "fs", "start_time", "end_time",
    ]
    # Keep only columns that exist
    existing = [c for c in priority_cols if c in mega.columns]
    remaining = [c for c in mega.columns if c not in existing]
    mega = mega[existing + remaining]

    return mega


# ═══════════════════════════════════════════════════════════════
# STEP 5: Compute summary tables for HTML
# ═══════════════════════════════════════════════════════════════
def compute_mae_tables(mega, structure):
    """Compute window-level and recording-level MAE tables."""
    results = {"window": {}, "recording": {}}
    models = sorted(mega["model"].unique())
    protos = ["old", "prism", "toolbox"]

    # ── Window-level MAE ──
    for split in ["train", "val"]:
        rows = []
        for model_name in models:
            row = {"model": model_name}
            for ds in DATASETS + ["ALL"]:
                for proto in protos:
                    sub = mega[(mega["model"] == model_name) &
                               (mega["protocol"] == proto) &
                               (mega["split"] == split)]
                    if ds != "ALL":
                        sub = sub[sub["dataset"] == ds]
                    if len(sub) > 0:
                        row[f"{ds}_{proto}"] = round(float(sub["abs_model_err"].mean()), 3)
                    else:
                        row[f"{ds}_{proto}"] = None
            rows.append(row)

        # Add CHROM row — use first available model's CHROM values
        # (CHROM output is identical across models for the same window)
        chrom_row = {"model": "CHROM (baseline)"}
        first_model = models[0] if models else None
        for ds in DATASETS + ["ALL"]:
            for proto in protos:
                if first_model is None:
                    chrom_row[f"{ds}_{proto}"] = None
                    continue
                sub = mega[(mega["model"] == first_model) &
                           (mega["protocol"] == proto) &
                           (mega["split"] == split)]
                if ds != "ALL":
                    sub = sub[sub["dataset"] == ds]
                if len(sub) > 0:
                    chrom_row[f"{ds}_{proto}"] = round(float(sub["abs_chrom_err"].mean()), 3)
                else:
                    chrom_row[f"{ds}_{proto}"] = None
        rows.append(chrom_row)
        results["window"][split] = rows

    # ── Recording-level MAE (from summary.txt) ──
    for split in ["train", "val"]:
        rows = []
        for model_name in models:
            row = {"model": model_name}
            protos_data = structure.get(model_name, {})
            for proto_key, tables_path in protos_data.items():
                summary = read_summary_txt(tables_path)
                for ds in DATASETS:
                    val = summary.get((ds, split))
                    if val:
                        row[f"{ds}_{proto_key}"] = val["mae"]
                # Overall
                val_all = summary.get(("ALL", split))
                if val_all:
                    row[f"ALL_{proto_key}"] = val_all["mae"]
            rows.append(row)
        results["recording"][split] = rows

    return results


def compute_error_tables(mega):
    """Compute error type count tables."""
    results = {}
    models = sorted(mega["model"].unique())

    for proto in ["old", "prism"]:
        for split in ["train", "val"]:
            rows = []
            for model_name in models:
                sub = mega[(mega["model"] == model_name) &
                           (mega["protocol"] == proto) &
                           (mega["split"] == split)]
                n = len(sub)
                if n == 0:
                    continue

                ae = sub["abs_model_err"]
                b1 = int((ae <= 1).sum())
                b2 = int(((ae > 1) & (ae <= 2)).sum())
                b3 = int(((ae > 2) & (ae <= 3)).sum())
                b45 = int(((ae > 3) & (ae <= 5)).sum())
                gt5 = int((ae > 5).sum())

                sub_h = int((sub["harmonic_type"] == "SUB-HARM").sum())
                sup_h = int((sub["harmonic_type"] == "SUP-HARM-2x").sum())
                r32 = int((sub["harmonic_type"] == "RATIO-3:2").sum())
                large_nh = int((sub["harmonic_type"] == "LARGE-NON-HARM").sum())

                max_err = float(ae.max())

                # Find recording(s) with max error
                max_idx = ae.idxmax()
                max_row = sub.loc[max_idx]
                max_rec = f"{max_row['dataset']}|{max_row['subject_id']}|{max_row.get('seq','?')}"

                # Find ALL recordings that have >5 BPM errors
                big_err_recs = sub[ae > 5]
                if len(big_err_recs) > 0:
                    unique_recs = big_err_recs.apply(
                        lambda r: f"{r['dataset']}|{r['subject_id']}", axis=1
                    ).unique()
                    max_err_recordings = "; ".join(sorted(set(unique_recs)))
                else:
                    max_err_recordings = "—"

                rows.append({
                    "model": model_name,
                    "n": n,
                    "b1": b1, "b1_pct": round(100*b1/n, 1),
                    "b2": b2, "b2_pct": round(100*b2/n, 1),
                    "b3": b3, "b3_pct": round(100*b3/n, 1),
                    "b45": b45, "b45_pct": round(100*b45/n, 1),
                    "gt5": gt5, "gt5_pct": round(100*gt5/n, 1),
                    "sub_h": sub_h,
                    "sup_h": sup_h,
                    "r32": r32,
                    "large_nh": large_nh,
                    "max_err": round(max_err, 1),
                    "max_err_rec": max_rec,
                    "all_gt5_recordings": max_err_recordings,
                })
            results[f"{proto}_{split}"] = rows

    return results


# ═══════════════════════════════════════════════════════════════
# STEP 6: Generate HTML Dashboard
# ═══════════════════════════════════════════════════════════════
def generate_html(mae_tables, error_tables, output_path):
    """Generate interactive HTML dashboard."""

    def mae_to_html_table(rows, split_label):
        """Convert MAE rows to HTML table string."""
        protos = ["old", "prism", "toolbox"]
        proto_labels = {"old": "Old@1BPM", "prism": "PRISM", "toolbox": "Toolbox"}
        ds_list = DATASETS + ["ALL"]
        ds_labels = {"PURE": "PURE", "TokyoTech": "TokyoTech", "UBFC": "UBFC", "ALL": "Overall"}

        html = f'<h3>{split_label}</h3>\n'
        html += '<table class="mae-table">\n'
        # Header row 1
        html += '<thead><tr><th rowspan="2">Model + Setup</th>'
        for ds in ds_list:
            html += f'<th colspan="3">{ds_labels[ds]}</th>'
        html += '</tr>\n<tr>'
        for ds in ds_list:
            for p in protos:
                html += f'<th>{proto_labels[p]}</th>'
        html += '</tr></thead>\n<tbody>\n'

        for row in rows:
            is_chrom = "CHROM" in row["model"]
            cls = ' class="chrom-row"' if is_chrom else ''
            html += f'<tr{cls}><td class="model-name">{row["model"]}</td>'
            for ds in ds_list:
                for p in protos:
                    val = row.get(f"{ds}_{p}")
                    if val is not None:
                        # Color code: green < 0.5, orange 0.5-2, red > 2
                        if val > 5:
                            cls_td = ' class="err-extreme"'
                        elif val > 2:
                            cls_td = ' class="err-high"'
                        elif val < 0.5:
                            cls_td = ' class="err-low"'
                        else:
                            cls_td = ''
                        html += f'<td{cls_td}>{val:.3f}</td>'
                    else:
                        html += '<td class="empty">—</td>'
            html += '</tr>\n'
        html += '</tbody></table>\n'
        return html

    def error_to_html_table(rows, title):
        """Convert error rows to HTML table string."""
        html = f'<h3>{title}</h3>\n'
        if not rows:
            html += '<p><em>No data available for this protocol/split.</em></p>'
            return html

        html += '<table class="error-table">\n'
        html += '<thead><tr>'
        html += '<th>Model</th><th>N</th>'
        html += '<th>≤1 BPM</th><th>1-2 BPM</th><th>2-3 BPM</th><th>3-5 BPM</th><th>&gt;5 BPM</th>'
        html += '<th>SubH</th><th>SupH</th><th>3:2</th><th>Large Err</th>'
        html += '<th>Max Err</th><th>Max Error Recording(s)</th>'
        html += '</tr></thead>\n<tbody>\n'

        for row in rows:
            html += f'<tr><td class="model-name">{row["model"]}</td>'
            html += f'<td>{row["n"]}</td>'
            for key, pct_key in [("b1","b1_pct"),("b2","b2_pct"),("b3","b3_pct"),("b45","b45_pct"),("gt5","gt5_pct")]:
                html += f'<td>{row[key]} ({row[pct_key]}%)</td>'
            html += f'<td class="{"harm" if row["sub_h"]>0 else ""}">{row["sub_h"]}</td>'
            html += f'<td class="{"harm" if row["sup_h"]>0 else ""}">{row["sup_h"]}</td>'
            html += f'<td class="{"harm" if row["r32"]>0 else ""}">{row["r32"]}</td>'
            html += f'<td>{row["large_nh"]}</td>'
            max_cls = "err-extreme" if row["max_err"] > 20 else ("err-high" if row["max_err"] > 10 else "")
            html += f'<td class="{max_cls}">{row["max_err"]} BPM</td>'
            html += f'<td class="rec-list">{row["all_gt5_recordings"]}</td>'
            html += '</tr>\n'
        html += '</tbody></table>\n'
        return html

    # ── Build all table HTML sections ──
    sections = {}

    # MAE tables
    for level in ["window", "recording"]:
        level_label = "Window-Level" if level == "window" else "Recording-Level (summary.txt)"
        for split in ["val", "train"]:
            split_label = "Validation" if split == "val" else "Training"
            key = f"mae_{level}_{split}"
            rows = mae_tables.get(level, {}).get(split, [])
            sections[key] = mae_to_html_table(rows, f"{split_label} MAE — {level_label}")

    # Error tables
    for proto in ["old", "prism"]:
        proto_label = PROTOCOL_LABELS.get(proto, proto)
        for split in ["train", "val"]:
            split_label = "Training" if split == "train" else "Validation"
            key = f"error_{proto}_{split}"
            rows = error_tables.get(f"{proto}_{split}", [])
            sections[key] = error_to_html_table(rows, f"{split_label} — Error Type Counts ({proto_label})")

    # ── Assemble full HTML ──
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MIMIC_CHROM Evaluation Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', Calibri, Arial, sans-serif; background: #f5f5f5; color: #333; padding: 20px; }}
h1 {{ text-align: center; margin-bottom: 5px; color: #2B579A; }}
.subtitle {{ text-align: center; color: #666; margin-bottom: 20px; font-size: 14px; }}
.controls {{ display: flex; gap: 15px; align-items: center; margin-bottom: 20px;
             padding: 15px; background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.controls label {{ font-weight: 600; font-size: 14px; }}
.controls select {{ padding: 8px 12px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; min-width: 200px; }}
.copy-btn {{ padding: 6px 14px; background: #2B579A; color: #fff; border: none; border-radius: 4px;
             cursor: pointer; font-size: 12px; margin-bottom: 8px; }}
.copy-btn:hover {{ background: #1a3d6e; }}
.copy-btn:active {{ background: #0d2540; }}
.copied {{ background: #27ae60 !important; }}
.section {{ display: none; }}
.section.active {{ display: block; }}
table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; background: #fff;
         box-shadow: 0 1px 3px rgba(0,0,0,0.08); font-size: 12px; }}
th {{ background: #2B579A; color: #fff; padding: 8px 6px; text-align: center;
      position: sticky; top: 0; z-index: 10; white-space: nowrap; }}
td {{ padding: 6px; text-align: center; border: 1px solid #e0e0e0; }}
.model-name {{ text-align: left; font-weight: 600; white-space: nowrap; min-width: 180px; }}
.chrom-row {{ background: #FFF3E0; }}
.empty {{ color: #ccc; }}
.err-low {{ background: #e8f5e9; color: #2e7d32; font-weight: 600; }}
.err-high {{ background: #fff3e0; color: #e65100; font-weight: 600; }}
.err-extreme {{ background: #ffebee; color: #c62828; font-weight: 700; }}
.harm {{ background: #ffcdd2; color: #b71c1c; font-weight: 700; }}
.rec-list {{ text-align: left; font-size: 11px; max-width: 300px; word-wrap: break-word; }}
tbody tr:hover {{ background: #e3f2fd; }}
h3 {{ margin: 15px 0 8px 0; color: #2B579A; }}
</style>
</head>
<body>
<h1>MIMIC_CHROM — Evaluation Dashboard</h1>
<p class="subtitle">Auto-generated from: {ROOT_DIR}</p>

<div class="controls">
  <label>View:</label>
  <select id="primary" onchange="updateView()">
    <option value="mae">MAE Tables</option>
    <option value="error">Error Analysis</option>
  </select>

  <label>Detail:</label>
  <select id="secondary" onchange="updateView()">
  </select>
</div>

<!-- MAE Window Val + Train -->
<div id="sec_mae_window" class="section">
  <button class="copy-btn" onclick="copyTable(this)">Copy Tables</button>
  {sections["mae_window_val"]}
  {sections["mae_window_train"]}
</div>

<!-- MAE Recording Val + Train -->
<div id="sec_mae_recording" class="section">
  <button class="copy-btn" onclick="copyTable(this)">Copy Tables</button>
  {sections["mae_recording_val"]}
  {sections["mae_recording_train"]}
</div>

<!-- Error Old -->
<div id="sec_error_old" class="section">
  <button class="copy-btn" onclick="copyTable(this)">Copy Tables</button>
  {sections["error_old_train"]}
  {sections["error_old_val"]}
</div>

<!-- Error PRISM -->
<div id="sec_error_prism" class="section">
  <button class="copy-btn" onclick="copyTable(this)">Copy Tables</button>
  {sections["error_prism_train"]}
  {sections["error_prism_val"]}
</div>

<script>
const secondaryOptions = {{
  mae: [
    {{value: "mae_window",    label: "Window-Level MAE"}},
    {{value: "mae_recording", label: "Recording-Level MAE (summary.txt)"}},
  ],
  error: [
    {{value: "error_old",   label: "Old@1BPM Protocol Errors"}},
    {{value: "error_prism", label: "PRISM Protocol Errors"}},
  ],
}};

function updateView() {{
  const primary = document.getElementById("primary").value;
  const secSelect = document.getElementById("secondary");

  // Update secondary options
  const opts = secondaryOptions[primary];
  const currentSec = secSelect.value;
  secSelect.innerHTML = "";
  opts.forEach(o => {{
    const opt = document.createElement("option");
    opt.value = o.value;
    opt.textContent = o.label;
    secSelect.appendChild(opt);
  }});

  // Keep selection if still valid
  if (opts.some(o => o.value === currentSec)) {{
    secSelect.value = currentSec;
  }}

  // Show/hide sections
  document.querySelectorAll(".section").forEach(s => s.classList.remove("active"));
  const activeId = "sec_" + secSelect.value;
  const el = document.getElementById(activeId);
  if (el) el.classList.add("active");
}}

function copyTable(btn) {{
  const section = btn.parentElement;
  const tables = section.querySelectorAll("table");
  let text = "";
  tables.forEach((table, idx) => {{
    // Get preceding h3
    const h3 = table.previousElementSibling;
    if (h3 && h3.tagName === "H3") text += h3.textContent + "\\n";

    const rows = table.querySelectorAll("tr");
    rows.forEach(row => {{
      const cells = row.querySelectorAll("th, td");
      const vals = Array.from(cells).map(c => c.textContent.trim());
      text += vals.join("\\t") + "\\n";
    }});
    text += "\\n";
  }});

  navigator.clipboard.writeText(text).then(() => {{
    btn.textContent = "Copied!";
    btn.classList.add("copied");
    setTimeout(() => {{
      btn.textContent = "Copy Tables";
      btn.classList.remove("copied");
    }}, 2000);
  }});
}}

// Initialize
updateView();
</script>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"  Dashboard written: {output_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main(root_dir=None):
    root = Path(root_dir if root_dir else ROOT_DIR)
    if not root.is_dir():
        print(f"[ERROR] ROOT not found: {ROOT_DIR}")
        return

    print(f"Root: {ROOT_DIR}")
    print(f"{'='*60}")

    # Step 1: Discover
    print("\n[1/5] Discovering models and protocols...")
    structure = discover_structure(root)
    for model_name, protos in structure.items():
        print(f"  {model_name}: {list(protos.keys())}")
    print(f"  Total: {len(structure)} models")

    # Step 2: Build MEGA CSV
    print("\n[2/5] Building MEGA CSV (all windows, all models)...")
    mega = build_mega_csv(structure, root)
    if mega.empty:
        print("[ERROR] No data loaded.")
        return

    mega_path = root / "MEGA_ALL_WINDOWS.csv"
    mega.to_csv(mega_path, index=False)
    print(f"  Saved: {mega_path}")
    print(f"  Rows: {len(mega):,}")
    print(f"  Models: {mega['model'].nunique()}")
    print(f"  Columns: {len(mega.columns)}")

    # Verification checksums
    print("\n  ── Verification ──")
    for model_name in sorted(mega["model"].unique()):
        for proto in ["old", "prism", "toolbox"]:
            sub = mega[(mega["model"] == model_name) & (mega["protocol"] == proto)]
            if len(sub) == 0:
                continue
            tr = sub[sub["split"] == "train"]
            vl = sub[sub["split"] == "val"]
            tr_mae = tr["abs_model_err"].mean()
            vl_mae = vl["abs_model_err"].mean()
            print(f"  {model_name:<40} {proto:<8} train={len(tr):>5} val={len(vl):>5} "
                  f"tr_MAE={tr_mae:.3f} vl_MAE={vl_mae:.3f}")

    # Cross-validate harmonic classification against eval code
    print("\n  ── Harmonic Cross-Validation ──")
    if "eval_failure_type" in mega.columns:
        has_eval = mega["eval_failure_type"].notna()
        if has_eval.sum() > 0:
            # Compare: when eval says "sub_harmonic", does this script agree?
            eval_sub = mega[mega["eval_failure_type"] == "sub_harmonic"]
            script_sub = mega[mega["harmonic_type"] == "SUB-HARM"]
            eval_sup = mega[mega["eval_failure_type"] == "super_harmonic_2x"]
            script_sup = mega[mega["harmonic_type"] == "SUP-HARM-2x"]

            print(f"  Eval code sub-harmonics:   {len(eval_sub)}")
            print(f"  Script sub-harmonics:      {len(script_sub)}")
            print(f"  Eval code super-harmonics: {len(eval_sup)}")
            print(f"  Script super-harmonics:    {len(script_sup)}")

            # Show disagreements
            if len(eval_sub) > 0 or len(script_sub) > 0:
                merged = mega[has_eval & ((mega["eval_failure_type"].str.contains("harm", na=False)) |
                              (mega["harmonic_type"].str.contains("HARM", na=False)))]
                if len(merged) > 0:
                    print(f"\n  Harmonic windows (combined):")
                    for _, r in merged.head(10).iterrows():
                        print(f"    {r['model']:<30} {r['dataset']:<10} s{r['subject_id']} w{r['window_idx']} "
                              f"GT={r['gt_bpm']:.1f} Pred={r['model_bpm']:.1f} "
                              f"eval={r['eval_failure_type']} script={r['harmonic_type']}")
        else:
            print("  No eval_failure_type data available for cross-validation")
    else:
        print("  No PSD files found — skipping cross-validation")

    # Step 3: Compute summary tables
    print("\n[3/5] Computing MAE summary tables...")
    mae_tables = compute_mae_tables(mega, structure)

    # Step 4: Compute error tables
    print("\n[4/5] Computing error analysis tables...")
    error_tables = compute_error_tables(mega)

    # Step 5: Generate HTML
    print("\n[5/5] Generating HTML dashboard...")
    html_path = root / "dashboard.html"
    generate_html(mae_tables, error_tables, str(html_path))

    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  MEGA CSV:  {mega_path}")
    print(f"  Dashboard: {html_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
