"""
launch_train.py
─────────────────────────────────────────────────────────────────────────────
Overnight sweep: train every model once, then evaluate each trained checkpoint
under multiple measurement PROTOCOLS, unattended.

What it does, per model, one after another:

    build_model(name)                     from model_ZOO_multi_branch.py
           |
    TRAIN  via run_one_epoch(...)         the SAME engine HIT_train.py uses
           |   (saves last/best checkpoints into RESULTS_ROOT/<name>/)
           v
    EVAL   via eval_protocols.run_protocol(...)   once per selected protocol
           |   (model + output folder injected; each protocol writes into
           |    RESULTS_ROOT/<name>/EVAL_PROTOCOL_<key>/)
           v
    record one row per (model, protocol) in MEGA_SUMMARY.csv
           |
    at the end: MEGA_PROTOCOL_ABLATION.csv  (model x protocol cross-tab)

Design notes
  - The model trains ONCE. Protocols are only different ways of measuring the
    same frozen checkpoint, so retraining per protocol would be wasted compute.
  - Nothing inside eval_protocols.py is rewritten. This driver imports its
    reusable pieces (run_protocol, PROTOCOL_CONFIGS, build_model) and overrides
    only the module-level output paths and load_model per iteration.
  - eval_protocols.py does `from HIT_train import SAVE_DIR` at import time; a
    stub is injected into sys.modules so the import always succeeds. The real
    per-model save dir is assigned to the module afterwards.
  - Each model runs inside try/except: one failure is logged and skipped, the
    sweep continues (safe to leave running overnight).
  - Resumable: a (model, protocol) whose eval CSV already exists is skipped
    unless FORCE_RERUN is set.

Controlling protocols for the sweep
  - PROTOCOLS_TO_RUN  (below) selects WHICH protocols each model is evaluated
    under. Use ["old", "prism", "toolbox"] for the full ablation, or a subset
    such as ["toolbox"] for a single publishable table.
  - The protocol PARAMETERS themselves live in eval_protocols.PROTOCOL_CONFIGS
    and stay the single source of truth. Edit a cutoff or FFT size there once,
    and every sweep run picks it up.
  - PROTOCOL_OVERRIDES (below) is an optional hook to patch specific fields for
    the overnight run only, without touching eval_protocols.py. Leave it empty
    to use the defaults defined in eval_protocols.py.
"""

from __future__ import annotations
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # required for CuBLAS determinism on CUDA ≥ 10.2

import sys
import math
import random
import types
import importlib
import traceback
import copy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ── Reused training pieces (identical imports to HIT_train.py) ────────────────
from ds_npz_windows import NPZWindowDataset, NPZRandomWindowDataset, WindowSpec
from training_engine import run_one_epoch
from helper import format_metrics

# ── Model zoo ─────────────────────────────────────────────────────────────────
from model_ZOO import build_model, SWEEP_ORDER


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG — edit here
# ═════════════════════════════════════════════════════════════════════════════

# Where every model's folder is created.
RESULTS_ROOT = Path(
    "/media/data/rPPG/Code/GitHub/Project_rPPG_Result/"
    "Result_Lab_1_SE_Concentration_loss_deltaMargin-bpm3"
)

# Name of the multi-protocol evaluation module (the new eval script).
EVAL_MODULE = "eval_protocols"

# Which models to sweep.
MODELS_TO_RUN = list(SWEEP_ORDER)

# ── PROTOCOL CONTROL ─────────────────────────────────────────────────────────
# Which protocols each trained model is evaluated under.
#   "old"      Old Approach            (8s / 240-pt FFT / 7.5 BPM bins)
#   "prism"    PRISM Protocol          (10s / 16384-pt FFT / dual-band)
#   "toolbox"  rPPG-Toolbox Protocol   (full-video / 0.75-2.5 Hz / per-video)
PROTOCOLS_TO_RUN = ["old", "prism", "toolbox"]

# Optional per-run overrides applied on top of eval_protocols.PROTOCOL_CONFIGS.
# Leave each dict empty to use the values defined in eval_protocols.py.
# Example — force a finer FFT for the old protocol during the sweep only:
#   PROTOCOL_OVERRIDES = {"old": {"nfft": 4096}}
PROTOCOL_OVERRIDES: dict[str, dict] = {
    # "old":     {},
    # "prism":   {},
    # "toolbox": {},
}
# ─────────────────────────────────────────────────────────────────────────────

# Training manifest (the AUGmented split used by HIT_train.py).
TRAIN_MANIFEST = ("/media/data/rPPG/Code/GitHub/Project_rPPG/Dataset/3_ROI/"
                  "PURE-x-UBFC-x-Tokoyo/manifest_split_AUG_BALANCED.csv")

# The manifest used for EVALUATION (may differ from the training manifest).
EVAL_MANIFEST = ("/media/data/rPPG/Code/GitHub/Project_rPPG/Dataset/3_ROI/"
                 "PURE-x-UBFC-x-Tokoyo/manifest_split_BALANCED.csv")

# Training hyper-parameters (mirrors HIT_train.py).
WIN_S                  = 8.0
STRIDE_S               = 1
ROI_INDEX              = "avg"
BATCH_SIZE             = 64
LR                     = 3e-4
NUM_WORKERS            = 4
SEED                   = 123
TRAIN_WINDOWS_PER_SEQ  = 80
VAL_MAX_WINDOWS_PER_SEQ= 80

EPOCHS    = 300
PATIENCE  = 30
MIN_DELTA = 0.0

# LIGHT_EVAL skips the heavy per-subject HTML sliders (signal + PSD) to keep the
# overnight run fast. All CSV tables and the bar/scatter summaries are still
# produced. Set False to also generate every per-subject diagnostic HTML.
LIGHT_EVAL = False

# FORCE_RERUN re-runs (model, protocol) pairs even if their eval output exists.
FORCE_RERUN = False

# SMOKE_TEST validates the full pipeline quickly before the real run:
# 2 epochs, two models, one protocol, light eval.
SMOKE_TEST = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═════════════════════════════════════════════════════════════════════════════
# Logging
# ═════════════════════════════════════════════════════════════════════════════

_LOG_FILE = None

def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    if _LOG_FILE is not None:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")



# ── Option A: Full deterministic seeding (bitwise-identical across runs) ──
# ~10–20% slower training, but guarantees exact reproducibility.
# Requires: os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" at file top.
# Use for: ablation comparisons, debugging, verifying stability.
def seed_everything(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


# ── Option B: Fast seeding (non-deterministic GPU ops allowed) ──
# Full speed, but identical seeds may produce slightly different results.
# The os.environ CUBLAS line at file top can stay — has no effect without Option A.
# Use for: normal training, sweeps where speed matters.
# def seed_everything(seed: int = 123) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# ═════════════════════════════════════════════════════════════════════════════
# Training — reuses run_one_epoch, mirrors HIT_train.py's loop
# ═════════════════════════════════════════════════════════════════════════════

def train_one_model(name: str, save_dir: Path) -> dict:
    """
    Train a single model and save last/best checkpoints into save_dir.
    Returns a small metrics summary used for the aggregate table.
    Skips training if last_model.pt already exists and FORCE_RERUN is False.
    """
    last_path = save_dir / "last_model.pt"
    if last_path.exists() and not FORCE_RERUN:
        log(f"  checkpoint already exists — skipping training for '{name}'")
        ck = torch.load(last_path, map_location="cpu")
        vm = ck.get("val_metrics", {})
        return {
            "epochs_run": ck.get("epoch", -1),
            "best_val_loss": float("nan"),
            "best_val_hr_mae": float(vm.get("hr_mae", float("nan"))),
            "params": sum(p.numel() for p in build_model(name).parameters()),
            "reach_samples": -1,
        }

    seed_everything(SEED)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(name).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    reach = model.reach_samples() if hasattr(model, "reach_samples") else -1
    log(f"  built '{name}': params={n_params}, reach={reach} samples "
        f"({reach/30.0:.2f}s @30fps)")

    window   = WindowSpec(win_s=WIN_S, stride_s=STRIDE_S)
    ds_train = NPZRandomWindowDataset(
        manifest_csv=TRAIN_MANIFEST, split="train", window=window,
        windows_per_seq_per_epoch=TRAIN_WINDOWS_PER_SEQ, seed=SEED,
    )
    ds_val = NPZWindowDataset(
        manifest_csv=TRAIN_MANIFEST, split="val", window=window,
        max_windows_per_seq=VAL_MAX_WINDOWS_PER_SEQ,
    )
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
    dl_val   = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)

    best_loss_path = save_dir / "best_loss_model.pt"
    best_hr_path   = save_dir / "best_hr_model.pt"

    best_val_loss   = float("inf")
    best_val_hr_mae = float("inf")
    epochs_no_improve = 0
    loss_rows = []
    epochs_run = 0

    for epoch in range(1, EPOCHS + 1):
        epochs_run = epoch
        if hasattr(ds_train, "set_epoch"):
            ds_train.set_epoch(epoch)

        train_metrics = run_one_epoch(
            model, dl_train, optimizer=optimizer,
            ROI_INDEX=ROI_INDEX, DEVICE=DEVICE, epoch=epoch)
        with torch.no_grad():
            val_metrics = run_one_epoch(
                model, dl_val, optimizer=None,
                ROI_INDEX=ROI_INDEX, DEVICE=DEVICE, epoch=epoch)

        scheduler.step(val_metrics["loss_total"])
        current_lr = optimizer.param_groups[0]["lr"]

        ckpt = {
            "epoch": epoch,
            "model_name": name,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": {
                "TRAIN_MANIFEST": TRAIN_MANIFEST, "ROI_INDEX": ROI_INDEX,
                "WIN_S": WIN_S, "STRIDE_S": STRIDE_S,
                "BATCH_SIZE": BATCH_SIZE, "LR": LR,
            },
        }
        torch.save(ckpt, last_path)

        val_loss = val_metrics["loss_total"]
        val_hr   = val_metrics["hr_mae"]

        if val_loss < (best_val_loss - MIN_DELTA):
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(ckpt, best_loss_path)
        else:
            epochs_no_improve += 1

        if (not math.isnan(val_hr)) and val_hr < best_val_hr_mae:
            best_val_hr_mae = val_hr
            torch.save(ckpt, best_hr_path)

        loss_rows.append({
            "epoch": epoch, "lr": current_lr,
            "train_loss": train_metrics["loss_total"],
            "val_loss": val_loss, "val_hr_mae": val_hr,
        })

        if epoch % 10 == 0 or epoch == 1:
            log(f"    epoch {epoch:3d} | "
                f"train_loss={train_metrics['loss_total']:.4f} "
                f"val_loss={val_loss:.4f} val_hr_mae={val_hr:.3f} "
                f"lr={current_lr:.1e}")

        if epochs_no_improve >= PATIENCE:
            log(f"    early stop at epoch {epoch} "
                f"(no val-loss improvement for {PATIENCE})")
            break

    pd.DataFrame(loss_rows).to_csv(save_dir / "loss_curve.csv", index=False)
    log(f"  trained '{name}': epochs_run={epochs_run} "
        f"best_val_loss={best_val_loss:.4f} best_val_hr_mae={best_val_hr_mae:.3f}")

    return {
        "epochs_run": epochs_run,
        "best_val_loss": best_val_loss,
        "best_val_hr_mae": best_val_hr_mae,
        "params": n_params,
        "reach_samples": reach,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation — reuses eval_protocols.run_protocol, model + paths injected
# ═════════════════════════════════════════════════════════════════════════════

def _get_eval_module():
    """
    Import eval_protocols safely. It does `from HIT_train import SAVE_DIR` at
    import time; inject a stub so the import never fails.
    """
    if "HIT_train" not in sys.modules:
        stub = types.ModuleType("HIT_train")
        stub.SAVE_DIR = RESULTS_ROOT           # placeholder, overridden per model
        sys.modules["HIT_train"] = stub
    return importlib.import_module(EVAL_MODULE)


def _noop(*_args, **_kwargs):
    """Placeholder to skip heavy per-subject HTML plots under LIGHT_EVAL."""
    return None


def _apply_protocol_overrides(E) -> dict:
    """
    Return a deep copy of eval_protocols.PROTOCOL_CONFIGS with PROTOCOL_OVERRIDES
    merged in. The module's own dict is left untouched; the merged copy is what
    the sweep evaluates with.
    """
    merged = copy.deepcopy(E.PROTOCOL_CONFIGS)
    for pkey, patch in PROTOCOL_OVERRIDES.items():
        if pkey in merged and patch:
            merged[pkey].update(patch)
            log(f"  protocol override applied to '{pkey}': {patch}")
    return merged


def _eval_csv_path(save_dir: Path, protocol_key: str) -> Path:
    return save_dir / f"EVAL_PROTOCOL_{protocol_key}" / "tables" / "ALL_EVAL_RESULTS.csv"


def run_protocols_for_model(name: str, save_dir: Path) -> list[dict]:
    """
    Evaluate one trained model under every protocol in PROTOCOLS_TO_RUN.
    Returns one result dict per protocol for the aggregate table.
    """
    E = _get_eval_module()

    # Point the eval module at this model's folder, checkpoint, and manifest.
    E.SAVE_DIR      = save_dir
    E.CKPT_PATH     = save_dir / "last_model.pt"
    E.MANIFEST_PATH = Path(EVAL_MANIFEST)
    E.ROI_INDEX     = ROI_INDEX

    # Merge any per-run protocol overrides on top of the module defaults, and
    # install the merged configs back onto the module so run_protocol sees them.
    E.PROTOCOL_CONFIGS = _apply_protocol_overrides(E)

    # Under LIGHT_EVAL, skip the heavy per-subject HTML sliders (CSVs kept).
    if LIGHT_EVAL:
        E.make_signal_comparison_slider = _noop
        E.make_psd_diagnostic_slider    = _noop

    # Build THIS architecture and load its trained weights.
    model = build_model(name)
    sd = torch.load(save_dir / "last_model.pt", map_location="cpu")
    model.load_state_dict(sd["model_state"])
    model.eval()
    model = model.to(E.DEVICE)

    # Load the evaluation manifest once and reuse across protocols.
    eval_df = pd.read_csv(E.MANIFEST_PATH,
                          dtype={"split": str, "path": str,
                                 "seq": str, "subject_id": str})
    for col in ("split", "seq", "subject_id"):
        if col not in eval_df.columns:
            eval_df[col] = "unknown"

    results = []
    for pkey in PROTOCOLS_TO_RUN:
        if pkey not in E.PROTOCOL_CONFIGS:
            log(f"  WARNING: unknown protocol '{pkey}' — skipping")
            continue

        if (not FORCE_RERUN) and _eval_csv_path(save_dir, pkey).exists():
            log(f"  protocol '{pkey}' already done for '{name}' — skipping")
        else:
            log(f"  evaluating '{name}' under protocol '{pkey}' ...")
            try:
                E.run_protocol(pkey, model, eval_df)
            except Exception as exc:
                log(f"  ERROR evaluating '{name}' under '{pkey}': {exc}")
                log(traceback.format_exc())
                results.append({
                    "model": name, "protocol": pkey,
                    "status": f"EVAL_FAILED: {exc}",
                    "eval_val_model_mae": np.nan, "eval_val_chrom_mae": np.nan,
                    "eval_test_model_mae": np.nan, "eval_test_chrom_mae": np.nan,
                })
                continue

        # Read back the split-wise summary this protocol wrote.
        row = {"model": name, "protocol": pkey, "status": "ok",
               "eval_val_model_mae": np.nan, "eval_val_chrom_mae": np.nan,
               "eval_test_model_mae": np.nan, "eval_test_chrom_mae": np.nan}
        csv_path = _eval_csv_path(save_dir, pkey)
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for split_key, tag in (("val", "val"), ("test", "test")):
                sub = df[df["split"].astype(str).str.lower() == split_key]
                if len(sub) > 0:
                    row[f"eval_{tag}_model_mae"] = float(sub["pred_gt_hr_mae"].mean())
                    row[f"eval_{tag}_chrom_mae"] = float(sub["chrom_gt_hr_mae"].mean())
        else:
            log(f"  WARNING: eval CSV not found at {csv_path}")
            row["status"] = "no_csv"
        results.append(row)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Orchestration
# ═════════════════════════════════════════════════════════════════════════════

def _all_protocols_done(save_dir: Path) -> bool:
    return all(_eval_csv_path(save_dir, pkey).exists() for pkey in PROTOCOLS_TO_RUN)


def _write_ablation_table(summary_rows: list[dict]) -> None:
    """Write the model x protocol cross-tab of val model MAE."""
    if not summary_rows:
        return
    df = pd.DataFrame(summary_rows)
    if not {"model", "protocol", "eval_val_model_mae"}.issubset(df.columns):
        return
    try:
        pivot = df.pivot_table(index="model", columns="protocol",
                               values="eval_val_model_mae", aggfunc="mean")
        pivot.to_csv(RESULTS_ROOT / "MEGA_PROTOCOL_ABLATION.csv")
        log("\nPROTOCOL ABLATION (val model MAE, model x protocol):")
        log("\n" + pivot.round(3).to_string())
    except Exception as exc:
        log(f"  could not build ablation pivot: {exc}")


def main() -> None:
    global _LOG_FILE, MODELS_TO_RUN, EPOCHS, LIGHT_EVAL, PROTOCOLS_TO_RUN

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = RESULTS_ROOT / "mega_run_protocols.log"

    if SMOKE_TEST:
        MODELS_TO_RUN = ["plain_k1", "tcn_small"]
        EPOCHS = 2
        LIGHT_EVAL = True
        PROTOCOLS_TO_RUN = ["toolbox"]
        log("SMOKE_TEST enabled: 2 epochs, models=['plain_k1','tcn_small'], "
            "protocols=['toolbox'], light eval. Set SMOKE_TEST=False for the "
            "overnight run.")

    log("=" * 74)
    log(f"MEGA RUN start | device={DEVICE} | models={len(MODELS_TO_RUN)} | "
        f"protocols={PROTOCOLS_TO_RUN} | epochs<= {EPOCHS} | root={RESULTS_ROOT}")
    log(f"models: {MODELS_TO_RUN}")
    log("=" * 74)

    summary_rows = []

    for i, name in enumerate(MODELS_TO_RUN, start=1):
        save_dir = RESULTS_ROOT / name
        log("-" * 74)
        log(f"[{i}/{len(MODELS_TO_RUN)}] MODEL: {name}")

        if (not FORCE_RERUN) and _all_protocols_done(save_dir) \
                and (save_dir / "last_model.pt").exists():
            log(f"  all protocols already done for '{name}' — skipping. "
                f"Set FORCE_RERUN=True to redo.")
            continue

        try:
            train_info = train_one_model(name, save_dir)

            proto_results = run_protocols_for_model(name, save_dir)
            for pr in proto_results:
                pr.update({
                    "params": train_info.get("params"),
                    "epochs_run": train_info.get("epochs_run"),
                    "reach_samples": train_info.get("reach_samples"),
                    "best_val_hr_mae": train_info.get("best_val_hr_mae"),
                })
                summary_rows.append(pr)
                mae = pr.get("eval_val_model_mae", float("nan"))
                log(f"  DONE '{name}' [{pr['protocol']}]: "
                    f"val_model_mae={mae:.3f}"
                    if np.isfinite(mae) else
                    f"  DONE '{name}' [{pr['protocol']}]: val_model_mae=nan")

        except Exception as exc:
            log(f"  ERROR on '{name}': {exc}")
            log(traceback.format_exc())
            summary_rows.append({"model": name, "protocol": "-",
                                 "status": f"FAILED: {exc}"})

        # Persist the aggregate after every model so partial results survive.
        pd.DataFrame(summary_rows).to_csv(RESULTS_ROOT / "MEGA_SUMMARY.csv", index=False)
        _write_ablation_table(summary_rows)

    log("=" * 74)
    log(f"MEGA RUN complete. Summary -> {RESULTS_ROOT / 'MEGA_SUMMARY.csv'}")
    log(f"Ablation table   -> {RESULTS_ROOT / 'MEGA_PROTOCOL_ABLATION.csv'}")
    log("=" * 74)
    log("Generating dashboard...")
    from generate_dashboard import main as generate_dashboard
    generate_dashboard(root_dir=str(RESULTS_ROOT))


if __name__ == "__main__":
    main()
