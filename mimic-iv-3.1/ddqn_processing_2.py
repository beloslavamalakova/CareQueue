"""
Efficient mimic-iv-3.1 DDQN processing for HPC
Builds transition table (s, a, r, s') incrementally to avoid memory issues
Author: Anusha Asthana
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path("/home/20243009/mimic-iv-3.1")
ICU_DIR = BASE_DIR / "icu"
HOSP_DIR = BASE_DIR / "hosp"
OUT_PATH = Path("sepsis_ddqn_transitions.parquet")

# Files
ICUSTAYS_FILE = ICU_DIR / "icustays.csv.gz"
CHARTEVENTS_FILE = ICU_DIR / "chartevents.csv.gz"
PROCEDURE_FILE = ICU_DIR / "procedureevents.csv.gz"
ADMISSIONS_FILE = HOSP_DIR / "admissions.csv.gz"
DIAGNOSES_FILE = HOSP_DIR / "diagnoses_icd.csv.gz"

# State vector itemids and labels
STATE_V_ITEMIDS = [220045, 223762, 220277, 220050, 220051, 220052]
STATE_COLS = ["HR", "TEMP", "SPO2", "SBP", "DBP", "MBP"]

# Actions
VENTILATION_IDS = {225794}
INVASIVE_LINES_IDS = {224263, 224268, 225752}
U_CATHETER_IDS = {229351}
IN_EX_TUBATION_IDS = {227194}
DIALYSIS_IDS = {225802}

BIN_SIZE = 4  # 4-hour bins

# ---------------- Load small reference tables ----------------
print("Loading ICU stays, procedures, admissions, diagnoses...")
icustays = pd.read_csv(ICUSTAYS_FILE, parse_dates=["intime", "outtime"])
icustays = icustays.dropna(subset=["intime", "outtime"])
procedureevents = pd.read_csv(PROCEDURE_FILE, parse_dates=["starttime"])
admissions = pd.read_csv(ADMISSIONS_FILE)
diagnoses = pd.read_csv(DIAGNOSES_FILE, low_memory=False)

# Select sepsis cohort
SEPSIS_PREFIXES = ("A40", "A41", "R65")
sepsis_diagnoses = diagnoses[(diagnoses.icd_version == 10) &
                             diagnoses.icd_code.str.startswith(SEPSIS_PREFIXES)]
sepsis_hadm_ids = set(sepsis_diagnoses.hadm_id.unique())
icustays = icustays[icustays.hadm_id.isin(sepsis_hadm_ids)]

# Admissions lookup
admissions_idx = admissions.set_index("hadm_id")["hospital_expire_flag"]

# ---------------- Helpers ----------------
def make_bins(intime, outtime):
    return pd.date_range(start=intime, end=outtime, freq=f"{BIN_SIZE}h")

def get_state_vector(ce, start, end):
    window = ce[(ce.charttime >= start) & (ce.charttime < end)]
    state = []
    for iid in STATE_V_ITEMIDS:
        vals = window.loc[window.itemid == iid, "valuenum"]
        state.append(vals.mean() if not vals.empty else np.nan)
    return np.array(state, dtype=np.float32)

def get_action(pe, start, end):
    window = pe[(pe.starttime >= start) & (pe.starttime < end)]
    if window.empty:
        return 0
    if window.itemid.isin(VENTILATION_IDS).any():
        return 1
    if window.itemid.isin(INVASIVE_LINES_IDS).any():
        return 2
    if window.itemid.isin(U_CATHETER_IDS).any():
        return 3
    if window.itemid.isin(IN_EX_TUBATION_IDS).any():
        return 4
    if window.itemid.isin(DIALYSIS_IDS).any():
        return 5
    return 0

# ---------------- Process ICU stays incrementally ----------------
def build_transitions_for_stay(stay, ce_chunk):
    stay_id = stay.stay_id
    hadm_id = stay.hadm_id
    died = admissions_idx.get(hadm_id, 0) == 1

    ce = ce_chunk[ce_chunk.stay_id == stay_id]
    pe = procedureevents[procedureevents.stay_id == stay_id]

    bins = make_bins(stay.intime, stay.outtime)
    if len(bins) < 2:
        return []

    states, actions = [], []
    for i in range(len(bins) - 1):
        s = get_state_vector(ce, bins[i], bins[i + 1])
        a = get_action(pe, bins[i], bins[i + 1])
        states.append(s)
        actions.append(a)

    transitions = []
    for t in range(len(states) - 1):
        transitions.append({"state": states[t], "action": actions[t],
                            "reward": 0, "next_state": states[t + 1], "done": 0})
    transitions.append({"state": states[-1], "action": actions[-1],
                        "reward": -1 if died else 1, "next_state": np.zeros_like(states[-1]),
                        "done": 1})
    return transitions

# ---------------- Main loop ----------------
if __name__ == "__main__":
    print("Processing ICU stays incrementally...")

    # Pre-filter chartevents to only relevant itemids using chunks
    chunksize = 5_000_000  # 5M rows at a time
    parquet_rows = []

    for chunk in pd.read_csv(CHARTEVENTS_FILE, chunksize=chunksize, parse_dates=["charttime"], low_memory=False):
        chunk = chunk[chunk.itemid.isin(STATE_V_ITEMIDS)]
        for i, (_, stay) in enumerate(icustays.iterrows()):
            transitions = build_transitions_for_stay(stay, chunk)
            for tr in transitions:
                row = {"action": tr["action"], "reward": tr["reward"], "done": tr["done"]}
                for j, col in enumerate(STATE_COLS):
                    row[f"s_{col}"] = tr["state"][j]
                    row[f"s_next_{col}"] = tr["next_state"][j]
                parquet_rows.append(row)
            if i % 100 == 0:
                print(f"Processed {i}/{len(icustays)} ICU stays in current chunk...")

    # Save incrementally built dataset
    transitions_df = pd.DataFrame(parquet_rows)
    transitions_df.to_parquet(OUT_PATH)
    print(f"Saved dataset to {OUT_PATH}")
