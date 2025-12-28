import pandas as pd

MIMIC_ROOT = "/home/beloslava/Desktop/CareQueue/mimic-iv-3.1/"

X_PATH = f"{MIMIC_ROOT}/interim/full_aggregated.csv"          # your patient matrix (strings: "nan"/"dropped")
PAT_PATH = f"{MIMIC_ROOT}/hosp/patients.csv.gz"
OUT_PATH = f"{MIMIC_ROOT}/interim/full_aggregated_with_reward.csv"

# Load patient matrix
X = pd.read_csv(X_PATH, dtype=str, keep_default_na=False, na_filter=False)

# Load patients table (need subject_id + dod)
patients = pd.read_csv(PAT_PATH, compression="gzip", usecols=["subject_id", "dod"])

# Reward: 1 if dod exists, else 0
patients["reward_death_1y"] = patients["dod"].notna().astype(int)

# Merge reward into X
X["subject_id"] = X["subject_id"].astype(int)
out = X.merge(patients[["subject_id", "reward_death_1y"]], on="subject_id", how="left")

# Any missing reward (shouldn't happen) -> treat as 0
out["reward_death_1y"] = out["reward_death_1y"].fillna(0).astype(int)

out.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH, "shape:", out.shape)
