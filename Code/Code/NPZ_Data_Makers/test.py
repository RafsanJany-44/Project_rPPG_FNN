import numpy as np
from pathlib import Path

# Check a single BH npz file
file_path = Path("/media/data/rPPG/Code/GitHub/Project_rPPG/Dataset/3_ROI/BH_RAW/0_0.npz")

with np.load(file_path, allow_pickle=True) as data:
    t = data['t']
    print(f"File: {file_path.name}")
    print(f"Time array length: {len(t)}")
    if len(t) > 1:
        print(f"t[0]: {t[0]}, t[-1]: {t[-1]}")
        print(f"Calculated Duration: {t[-1] - t[0]}s")
    else:
        print("Error: Time array is empty or too short.")