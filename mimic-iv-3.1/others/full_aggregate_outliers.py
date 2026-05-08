import os
import numpy as np
import pandas as pd


INPUT_CSV = "/home/beloslava/Downloads/mimic-iv-3.1/interim/patient_feature_matrix_top5.csv"
OUT_CSV   = "/home/beloslava/Downloads/mimic-iv-3.1/interim/full_aggregated.csv"

# Percentiles for outlier detection
LOW_Q  = 0.01
HIGH_Q = 0.99

# Load data
df = pd.read_csv(INPUT_CSV)

# Ensure subject_id stays first
if "subject_id" in df.columns:
    cols = ["subject_id"] + [c for c in df.columns if c != "subject_id"]
    df = df[cols]

# Detect numeric columns
numeric_cols = []
for col in df.columns:
    if col == "subject_id":
        continue
    coerced = pd.to_numeric(df[col], errors="coerce")
    if coerced.notna().sum() > 0:
        df[col] = coerced
        numeric_cols.append(col)

# Replace inf with NaN
df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

# Compute outlier thresholds
q_low  = df[numeric_cols].quantile(LOW_Q)
q_high = df[numeric_cols].quantile(HIGH_Q)


for col in numeric_cols:
    lo, hi = q_low[col], q_high[col]

    # Convert column to object so we can store strings
    df[col] = df[col].astype("object")

    # 1️⃣ Missing → "nan"
    df.loc[df[col].isna(), col] = "nan"

    # 2️⃣ Outliers → "dropped"
    if not pd.isna(lo) and not pd.isna(hi):
        mask_outlier = (
            (df[col] != "nan") &
            ((pd.to_numeric(df[col], errors="coerce") < lo) |
             (pd.to_numeric(df[col], errors="coerce") > hi))
        )
        df.loc[mask_outlier, col] = "dropped"


os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=False)

print("Saved:", OUT_CSV)
print("Shape:", df.shape)
print("Features processed:", len(numeric_cols))
