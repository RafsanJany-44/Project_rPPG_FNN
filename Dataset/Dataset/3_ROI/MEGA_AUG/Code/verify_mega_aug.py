"""
verify_mega_aug.py — end-to-end QA for the MEGA_AUG output
──────────────────────────────────────────────────────────────
Checks:
  1. Manifest integrity   — all expected columns, no NaN in critical fields
  2. File existence        — every path in the manifest points to a real .npz
  3. NPZ loadability       — sampled files load and contain X, Y, t, meta with matching shapes
  4. Split sanity          — no 'test' rows remain; aug rows exist only under 'train'
  5. Subject-level leakage — NO subject appears in both train and val (critical check)
  6. Dataset labels        — all 6 labels present, UBFCPhys distinct from UBFC
  7. Split ratio           — ~80/20 subject-level split per dataset
  8. HR shift validation   — sampled aug files: measured HR ≈ original HR × factor
  9. Waveform correlation  — resampled BVP correlates with original (shape preserved)
 10. Ablation manifest     — zero _x* rows, count matches, no test splits, no leakage

Run from PROJECT_ROOT or adjust the paths below.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/media/data/rPPG/Code/GitHub/Project_rPPG")
MEGA_DIR     = PROJECT_ROOT / "Dataset" / "3_ROI" / "MEGA_AUG"
AUG_MANIFEST  = MEGA_DIR / "manifest_mega_aug.csv"
ORIG_MANIFEST = MEGA_DIR / "manifest_mega_orig.csv"

BPM_MIN, BPM_MAX = 40.0, 180.0
WINDOW_S = 8.0
HR_TOLERANCE = 0.15       # allowed relative error between expected and measured HR shift
N_SAMPLE_VERIFY = 30      # number of random aug files to spot-check for HR shift
EXPECTED_TRAIN_RATIO = 0.80
# ──────────────────────────────────────────────────────────────────────────────


# ── SUBJECT EXTRACTION (must match make_mega_aug.py) ──────────────────────────

def _extract_subject_bh(sid): return sid.split("_")[0]
def _extract_subject_cohface(sid): parts = sid.rsplit("_", 1); return parts[0] if len(parts) > 1 else sid
def _extract_subject_pure(sid): return sid
def _extract_subject_tokyotech(sid): parts = sid.split("_"); return "_".join(parts[:2]) if len(parts) >= 2 else sid
def _extract_subject_ubfc(sid): return sid
def _extract_subject_ubfcphys(sid): m = re.match(r"(s\d+)", sid); return m.group(1) if m else sid

SUBJECT_EXTRACTORS = {
    "BH": _extract_subject_bh,
    "COHFACE": _extract_subject_cohface,
    "PURE": _extract_subject_pure,
    "TokyoTech": _extract_subject_tokyotech,
    "UBFC": _extract_subject_ubfc,
    "UBFCPhys": _extract_subject_ubfcphys,
}

def natural_sort_key(s: str):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


# ── LOGGING ───────────────────────────────────────────────────────────────────

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

errors = []
warnings = []

def log_pass(msg): print(f"  {PASS} {msg}")
def log_fail(msg): print(f"  {FAIL} {msg}"); errors.append(msg)
def log_warn(msg): print(f"  {WARN} {msg}"); warnings.append(msg)
def log_info(msg): print(f"  {INFO} {msg}")


# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_meta(m):
    if isinstance(m, np.ndarray): m = m.item()
    if isinstance(m, (bytes, bytearray)): m = m.decode("utf-8", "ignore")
    if isinstance(m, str):
        try: return json.loads(m)
        except Exception: return {}
    return m if isinstance(m, dict) else {}

def infer_fs(meta, t):
    fs = float(meta.get("fps", np.nan))
    if np.isfinite(fs) and fs > 1: return fs
    dt = np.diff(t); dt = dt[dt > 0]
    return float(1.0 / np.median(dt)) if dt.size else 30.0

def fft_hr(sig, fs):
    sig = np.asarray(sig, float) - np.mean(sig)
    if len(sig) < 16: return np.nan
    power = np.abs(np.fft.rfft(sig)) ** 2
    freqs = np.fft.rfftfreq(len(sig), d=1/fs) * 60
    mask = (freqs >= BPM_MIN) & (freqs <= BPM_MAX)
    if not mask.any(): return np.nan
    return float(freqs[mask][np.argmax(power[mask])])

def resolve_path(rel_path):
    p = PROJECT_ROOT / rel_path
    if p.exists(): return p
    p2 = Path(rel_path)
    if p2.is_absolute() and p2.exists(): return p2
    return p


def get_subject_splits(df):
    """
    Returns per-dataset dict of {subject: set_of_splits}.
    Used to detect leakage.
    """
    result = {}
    for ds in df["dataset"].unique():
        extractor = SUBJECT_EXTRACTORS.get(ds)
        if extractor is None:
            continue
        sub_df = df[df["dataset"] == ds]
        subj_splits = {}
        for _, row in sub_df.iterrows():
            true_subj = extractor(row["subject_id"])
            split = row["split"]
            subj_splits.setdefault(true_subj, set()).add(split)
        result[ds] = subj_splits
    return result


# ── CHECK 1: Manifest integrity ──────────────────────────────────────────────

def check_manifest_integrity():
    print("\n── CHECK 1: Manifest integrity ──")
    for label, path in [("aug", AUG_MANIFEST), ("orig", ORIG_MANIFEST)]:
        if not path.exists():
            log_fail(f"{label} manifest not found: {path}")
            continue
        log_pass(f"{label} manifest exists")
        df = pd.read_csv(path, dtype=str)
        required = ["dataset", "seq", "subject_id", "path", "T", "C", "split"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            log_fail(f"{label} manifest missing columns: {missing}")
        else:
            log_pass(f"{label} manifest has all required columns")
        for col in ["dataset", "seq", "path", "split"]:
            if col in df.columns:
                n_nan = df[col].isna().sum()
                if n_nan > 0:
                    log_fail(f"{label}: {n_nan} NaN in '{col}'")
                else:
                    log_pass(f"{label}: no NaN in '{col}'")


# ── CHECK 2: File existence ──────────────────────────────────────────────────

def check_file_existence():
    print("\n── CHECK 2: File existence ──")
    df = pd.read_csv(AUG_MANIFEST, dtype=str)
    missing = []
    for _, row in df.iterrows():
        p = resolve_path(row["path"])
        if not p.exists():
            missing.append(row["path"])
    if missing:
        log_fail(f"{len(missing)} / {len(df)} files missing")
        for m in missing[:10]:
            print(f"           {m}")
        if len(missing) > 10:
            print(f"           ... and {len(missing)-10} more")
    else:
        log_pass(f"All {len(df)} files exist on disk")


# ── CHECK 3: NPZ loadability + shape ─────────────────────────────────────────

def check_npz_loadability():
    print("\n── CHECK 3: NPZ loadability & shape (random sample) ──")
    df = pd.read_csv(AUG_MANIFEST, dtype=str)
    sample = df.sample(n=min(50, len(df)), random_state=42)
    load_fails, key_fails, shape_fails = [], [], []

    for _, row in sample.iterrows():
        p = resolve_path(row["path"])
        if not p.exists(): continue
        try:
            with np.load(p, allow_pickle=True) as z:
                keys = set(z.keys())
                if not {"X", "Y", "t", "meta"}.issubset(keys):
                    key_fails.append(row["path"]); continue
                X, Y, t = z["X"], z["Y"], z["t"]
                T_csv = int(row["T"]) if row["T"] and row["T"] != "" else None
                if X.shape[0] != Y.shape[0] or X.shape[0] != t.shape[0]:
                    shape_fails.append((row["path"], X.shape, Y.shape, t.shape))
                elif T_csv is not None and X.shape[0] != T_csv:
                    shape_fails.append((row["path"], f"X[0]={X.shape[0]} vs T={T_csv}"))
        except Exception as e:
            load_fails.append((row["path"], str(e)))

    if load_fails: log_fail(f"{len(load_fails)} files failed to load")
    else: log_pass(f"All {len(sample)} sampled files load successfully")
    if key_fails: log_fail(f"{len(key_fails)} files missing required keys")
    else: log_pass("All sampled files have required keys")
    if shape_fails: log_fail(f"{len(shape_fails)} files have shape mismatches")
    else: log_pass("All sampled files have consistent shapes")


# ── CHECK 4: Split sanity ────────────────────────────────────────────────────

def check_split_sanity():
    print("\n── CHECK 4: Split sanity ──")
    df = pd.read_csv(AUG_MANIFEST, dtype=str)

    test_rows = df[df["split"].str.lower() == "test"]
    if len(test_rows) > 0:
        log_fail(f"{len(test_rows)} rows still have split='test'")
    else:
        log_pass("No 'test' split rows remain")

    aug_mask = df["seq"].str.contains("_x", na=False)
    aug_val = df[aug_mask & (df["split"].str.lower() != "train")]
    if len(aug_val) > 0:
        log_fail(f"{len(aug_val)} augmented rows found outside 'train'")
    else:
        log_pass("All augmented rows are in 'train' only")

    log_info("Split distribution:")
    for split in sorted(df["split"].unique()):
        n = (df["split"] == split).sum()
        n_aug_s = df[df["split"] == split]["seq"].str.contains("_x", na=False).sum()
        print(f"           {split:6s}: {n:5d} rows  (aug: {n_aug_s})")


# ── CHECK 5: Subject-level leakage (CRITICAL) ────────────────────────────────

def check_subject_leakage():
    print("\n── CHECK 5: Subject-level data leakage (CRITICAL) ──")
    df = pd.read_csv(AUG_MANIFEST, dtype=str)
    ds_subj_splits = get_subject_splits(df)

    any_leak = False
    for ds in sorted(ds_subj_splits.keys()):
        subj_splits = ds_subj_splits[ds]
        leaked = {s: splits for s, splits in subj_splits.items() if len(splits) > 1}
        if leaked:
            any_leak = True
            log_fail(f"{ds}: {len(leaked)} subject(s) in BOTH train and val → DATA LEAKAGE")
            for s, splits in sorted(leaked.items(), key=lambda x: natural_sort_key(x[0])):
                print(f"           subject '{s}' in splits: {splits}")
        else:
            train_subjs = sorted([s for s, sp in subj_splits.items() if "train" in sp], key=natural_sort_key)
            val_subjs   = sorted([s for s, sp in subj_splits.items() if "val" in sp], key=natural_sort_key)
            log_pass(f"{ds:12s}: no leakage — train={len(train_subjs)} subj, val={len(val_subjs)} subj")
            print(f"           train: {train_subjs}")
            print(f"           val  : {val_subjs}")

    if not any_leak:
        log_pass("ALL datasets have clean subject-level separation")


# ── CHECK 6: Dataset labels ──────────────────────────────────────────────────

def check_dataset_labels():
    print("\n── CHECK 6: Dataset labels ──")
    df = pd.read_csv(AUG_MANIFEST, dtype=str)
    labels = sorted(df["dataset"].unique())
    log_info(f"Labels found: {labels}")

    expected = {"BH", "COHFACE", "PURE", "TokyoTech", "UBFC", "UBFCPhys"}
    found = set(labels)
    if expected == found:
        log_pass("All 6 expected labels present, UBFCPhys distinct from UBFC")
    else:
        if expected - found: log_fail(f"Missing labels: {expected - found}")
        if found - expected: log_warn(f"Unexpected labels: {found - expected}")

    log_info("Per-dataset row counts:")
    for ds in labels:
        sub = df[df["dataset"] == ds]
        n_orig_ds = (~sub["seq"].str.contains("_x", na=False)).sum()
        n_aug_ds  = sub["seq"].str.contains("_x", na=False).sum()
        n_tr = (sub["split"] == "train").sum()
        n_va = (sub["split"] == "val").sum()
        print(f"           {ds:12s}  orig={n_orig_ds:4d}  aug={n_aug_ds:5d}  train={n_tr:5d}  val={n_va:4d}")


# ── CHECK 7: Split ratio ─────────────────────────────────────────────────────

def check_split_ratio():
    print("\n── CHECK 7: Subject-level split ratio (~80/20) ──")
    df = pd.read_csv(AUG_MANIFEST, dtype=str)
    # Use only original rows (no aug) for ratio check
    df_orig_only = df[~df["seq"].str.contains("_x", na=False)]
    ds_subj_splits = get_subject_splits(df_orig_only)

    for ds in sorted(ds_subj_splits.keys()):
        subj_splits = ds_subj_splits[ds]
        n_train = sum(1 for sp in subj_splits.values() if "train" in sp)
        n_val   = sum(1 for sp in subj_splits.values() if "val" in sp)
        n_total = n_train + n_val
        ratio = n_train / n_total if n_total > 0 else 0
        status = "ok" if 0.70 <= ratio <= 0.90 else "off"
        if status == "ok":
            log_pass(f"{ds:12s}: {n_train}/{n_total} subjects in train ({ratio:.0%})")
        else:
            log_warn(f"{ds:12s}: {n_train}/{n_total} subjects in train ({ratio:.0%}) — expected ~80%")


# ── CHECK 8: HR shift validation ─────────────────────────────────────────────

def check_hr_shift():
    print(f"\n── CHECK 8: HR shift validation (sampling {N_SAMPLE_VERIFY} aug files) ──")
    df = pd.read_csv(AUG_MANIFEST, dtype=str)
    aug_rows = df[df["seq"].str.contains("_x", na=False)].copy()
    if len(aug_rows) == 0:
        log_warn("No augmented rows to verify"); return

    sample = aug_rows.sample(n=min(N_SAMPLE_VERIFY, len(aug_rows)), random_state=99)
    passed, failed, skipped = 0, 0, 0

    for _, row in sample.iterrows():
        seq = row["seq"]
        try: factor = float(seq.split("_x")[-1])
        except ValueError: skipped += 1; continue

        base_seq = seq.rsplit("_x", 1)[0]
        orig_row = df[(df["dataset"] == row["dataset"]) & (df["seq"] == base_seq)]
        if len(orig_row) == 0: skipped += 1; continue
        orig_row = orig_row.iloc[0]

        orig_path = resolve_path(orig_row["path"])
        aug_path  = resolve_path(row["path"])
        if not orig_path.exists() or not aug_path.exists(): skipped += 1; continue

        try:
            with np.load(orig_path, allow_pickle=True) as z:
                Y_orig, t_orig, meta_orig = z["Y"].astype(np.float32), z["t"].astype(np.float64), load_meta(z["meta"])
            with np.load(aug_path, allow_pickle=True) as z:
                Y_aug, t_aug, meta_aug = z["Y"].astype(np.float32), z["t"].astype(np.float64), load_meta(z["meta"])
        except Exception: skipped += 1; continue

        hr_orig = fft_hr(Y_orig, infer_fs(meta_orig, t_orig))
        hr_aug  = fft_hr(Y_aug,  infer_fs(meta_aug,  t_aug))
        if not np.isfinite(hr_orig) or not np.isfinite(hr_aug): skipped += 1; continue

        expected_hr = hr_orig * factor
        rel_error = abs(hr_aug - expected_hr) / (expected_hr + 1e-8)
        if rel_error <= HR_TOLERANCE:
            passed += 1
        else:
            failed += 1
            if failed <= 5:
                print(f"           seq={seq}  orig_HR={hr_orig:.1f}  ×{factor}  expected={expected_hr:.1f}  got={hr_aug:.1f}  err={rel_error:.1%}")

    total = passed + failed
    if total == 0: log_warn("Could not verify any HR shifts")
    elif failed == 0: log_pass(f"HR shift correct for all {passed} checked (tol={HR_TOLERANCE:.0%})")
    else: log_warn(f"HR shift: {passed}/{total} passed, {failed} outside tolerance")
    if skipped: log_info(f"{skipped} skipped (missing original or unreadable)")


# ── CHECK 9: Waveform correlation ────────────────────────────────────────────

def check_waveform_correlation():
    print(f"\n── CHECK 9: Waveform shape preservation (correlation check) ──")
    df = pd.read_csv(AUG_MANIFEST, dtype=str)
    aug_rows = df[df["seq"].str.contains("_x", na=False)]
    if len(aug_rows) == 0: log_warn("No augmented rows"); return

    sample = aug_rows.sample(n=min(15, len(aug_rows)), random_state=77)
    correlations = []

    for _, row in sample.iterrows():
        seq = row["seq"]
        base_seq = seq.rsplit("_x", 1)[0]
        try: factor = float(seq.split("_x")[-1])
        except ValueError: continue
        orig_row = df[(df["dataset"] == row["dataset"]) & (df["seq"] == base_seq)]
        if len(orig_row) == 0: continue

        orig_path = resolve_path(orig_row.iloc[0]["path"])
        aug_path  = resolve_path(row["path"])
        if not orig_path.exists() or not aug_path.exists(): continue

        try:
            Y_orig = np.load(orig_path, allow_pickle=True)["Y"].astype(np.float64)
            Y_aug  = np.load(aug_path,  allow_pickle=True)["Y"].astype(np.float64)
        except Exception: continue

        if len(Y_aug) < 16 or len(Y_orig) < 16: continue
        src = np.linspace(0, len(Y_orig) - 1, len(Y_aug))
        Y_orig_resampled = np.interp(src, np.arange(len(Y_orig)), Y_orig)
        corr = np.corrcoef(Y_orig_resampled, Y_aug)[0, 1]
        correlations.append(corr)

    if not correlations: log_warn("Could not compute any correlations"); return
    mean_corr, min_corr = np.mean(correlations), np.min(correlations)
    if min_corr > 0.95:
        log_pass(f"Waveform correlation: mean={mean_corr:.4f}  min={min_corr:.4f}  (n={len(correlations)})")
    elif min_corr > 0.80:
        log_warn(f"Waveform correlation: mean={mean_corr:.4f}  min={min_corr:.4f}")
    else:
        log_fail(f"Waveform correlation: mean={mean_corr:.4f}  min={min_corr:.4f} — shape not preserved")


# ── CHECK 10: Ablation manifest ──────────────────────────────────────────────

def check_ablation_manifest():
    print("\n── CHECK 10: Ablation manifest consistency ──")
    if not ORIG_MANIFEST.exists():
        log_fail(f"Ablation manifest not found"); return

    df_aug  = pd.read_csv(AUG_MANIFEST, dtype=str)
    df_orig = pd.read_csv(ORIG_MANIFEST, dtype=str)

    aug_in_orig = df_orig["seq"].str.contains("_x", na=False).sum()
    if aug_in_orig > 0:
        log_fail(f"Ablation manifest contains {aug_in_orig} augmented rows")
    else:
        log_pass("Ablation manifest has zero augmented rows")

    n_orig_in_aug = (~df_aug["seq"].str.contains("_x", na=False)).sum()
    if len(df_orig) == n_orig_in_aug:
        log_pass(f"Row count ({len(df_orig)}) matches originals in aug manifest")
    else:
        log_fail(f"Ablation rows={len(df_orig)} vs originals in aug={n_orig_in_aug}")

    test_in_orig = (df_orig["split"].str.lower() == "test").sum()
    if test_in_orig > 0:
        log_fail(f"Ablation has {test_in_orig} 'test' rows")
    else:
        log_pass("Ablation: no 'test' rows")

    # Subject leakage in ablation manifest too
    ds_subj_splits = get_subject_splits(df_orig)
    any_leak = False
    for ds, subj_splits in ds_subj_splits.items():
        leaked = {s for s, sp in subj_splits.items() if len(sp) > 1}
        if leaked:
            any_leak = True
            log_fail(f"Ablation — {ds}: {len(leaked)} subject(s) in both splits")
    if not any_leak:
        log_pass("Ablation: no subject-level leakage")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MEGA AUG — Verification Report")
    print("=" * 60)

    check_manifest_integrity()
    check_file_existence()
    check_npz_loadability()
    check_split_sanity()
    check_subject_leakage()
    check_dataset_labels()
    check_split_ratio()
    check_hr_shift()
    check_waveform_correlation()
    check_ablation_manifest()

    print(f"\n{'=' * 60}")
    if errors:
        print(f"{FAIL} {len(errors)} error(s) found:")
        for e in errors:
            print(f"       • {e}")
    else:
        print(f"{PASS} All checks passed.")
    if warnings:
        print(f"{WARN} {len(warnings)} warning(s):")
        for w in warnings:
            print(f"       • {w}")
    print("=" * 60)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
