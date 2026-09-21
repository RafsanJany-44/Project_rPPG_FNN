# main/RUN_2_split_manifest_subjectwise.py
# python RUN_2_split_manifest_subjectwise.py

from pathlib import Path
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():

    dataset_name = "3_ROI/BH_RAW" # 3_ROI/UBFC_RAW, "UBFC-Phys_RAW", "PURE_RAW"
    MANIFEST = PROJECT_ROOT / "Dataset" / dataset_name/ "manifest.csv"

    
    OUT_DIR  = MANIFEST.parent

    df = pd.read_csv(MANIFEST, dtype={"subject_id": str, "seq": str, "path": str})
    df["subject_id"] = df["subject_id"].str.zfill(2)

    subjects = sorted(df["subject_id"].unique().tolist())

    rng = np.random.RandomState(42)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(0.80 * n)
    n_val   = int(0.15 * n)

    train_sub = set(subjects[:n_train])
    val_sub   = set(subjects[n_train:n_train + n_val])
    test_sub  = set(subjects[n_train + n_val:])

    def split_name(sid):
        if sid in train_sub: return "train"
        if sid in val_sub: return "val"
        return "test"

    df["split"] = df["subject_id"].apply(split_name)

    out_path = OUT_DIR / "manifest_split.csv"
    df.to_csv(out_path, index=False)

    print("[DONE] our split saved:", out_path)
    print(df["split"].value_counts())


if __name__ == "__main__":
    main()
