# main/RUN_1_make_manifest.py
# python RUN_1_make_manifest.py

import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def seq_to_subject_id(seq_name: str) -> str:
    # Our PURE seq names look like "01-01", "10-06", etc.
    # Our subject_id is the part before "-" (e.g. "01", "10")
    if "-" in seq_name:
        return seq_name.split("-")[0]
    return seq_name




def main():
    dataset_name = "3_ROI/BH_RAW" # 3_ROI/UBFC_RAW, "UBFC-Phys_RAW", "PURE_RAW"
    NPZ_DIR = PROJECT_ROOT / "Dataset" / dataset_name


    OUT_CSV = NPZ_DIR / "manifest.csv"

    rows = []
    for npz_path in tqdm(sorted(NPZ_DIR.glob("*.npz"))):
        seq = npz_path.stem
        with np.load(npz_path, allow_pickle=True) as z:
            X = z["X"]
            Y = z["Y"]
            t = z["t"]
            meta = json.loads(str(z["meta"]))
            qc = json.loads(str(z["qc"]))

        rows.append({
            "dataset": dataset_name.split("/")[1].split("_")[0],
            "seq": f"'{seq}",  # prevents Excel from converting to date
            "subject_id": seq_to_subject_id(seq),
            "path":  npz_path.relative_to(PROJECT_ROOT).as_posix(),
            "T": int(X.shape[0]),
            "C": int(X.shape[1]),
            "t_start": float(t[0]) if len(t) else np.nan,
            "t_end": float(t[-1]) if len(t) else np.nan,
            "roi_names": ",".join(meta.get("roi_names", [])),
            "passed": qc.get("passed", None),
            "fps_est": qc.get("fs", qc.get("fps_est_median", None)),
            "duration_ratio": qc.get("duration_ratio_gt_over_video", None),
        })


    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[DONE] our manifest saved: {OUT_CSV}")
    print(df.head())


if __name__ == "__main__":
    main()


