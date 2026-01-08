'''
Docstring for mimic-iv-3.1.iql_processing
Idea is adapted from Anusha's ddqn_processing to make the dataset in a transition table format, 
specifically in the (s, a, r, s') format. The main change is in the actions defintion, I am looking to 
define actions as vectors, and broaden the possible actions.
Note that the rewards remain the +1 or -1, based on whether the patient has been 
discharged or has passed away, respectively. 

Changes:
1) Action is now a vector
In the adapted code, the action vector is built from three blocks:
(i) Procedures
(ii) Drugs
(iii) Continuous medications (encoded with start/stop/dose-change)
For each continuous class (e.g., norepinephrine), I encode a discrete value:
•	+1 = started
•	-1 = stopped
•	+2 = dose increased
•	-2 = dose decreased
•	0 = no change
Dose changes are detected by comparing current rate vs previous rate, with a threshold set at 10%.

2) 10-minute grouped “action windows”
Procedures that occur within like a 10 minute period then we group them, and then we have the next 4hrs of patients state

3) State is more informative than just mean
Change
On top of 4hr mean, I expanded each variable into compact summary stats of mean, min, max, std, last, trend
So each vital yields 6 features; with 6 vitals you get 36 features.
This is still small, but captures instability and direction, not just average.

Author: Ayush Jain

'''

from __future__ import annotations

import os
import math
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ----------------------------- Paths & I/O -----------------------------
BASE_DIR = Path("/home/20243009/mimic-iv-3.1")
ICU_DIR = BASE_DIR / "icu"
HOSP_DIR = BASE_DIR / "hosp"

ICUSTAYS_FILE = ICU_DIR / "icustays.csv.gz"
CHARTEVENTS_FILE = ICU_DIR / "chartevents.csv.gz"
PROCEDURE_FILE = ICU_DIR / "procedureevents.csv.gz"
INPUTEVENTS_FILE = ICU_DIR / "inputevents.csv.gz"
INGREDIENTEVENTS_FILE = ICU_DIR / "ingredientevents.csv.gz"  # optional; if missing, code will skip
ADMISSIONS_FILE = HOSP_DIR / "admissions.csv.gz"
DIAGNOSES_FILE = HOSP_DIR / "diagnoses_icd.csv.gz"

# Output dataset of transitions
OUT_PATH = Path("sepsis_iql_actionvec_transitions.parquet")

# Temporary storage for filtered chartevents (parquet parts)
TMP_STATE_DIR = Path("tmp_state_chartevents_parts")
TMP_STATE_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------- Time settings -----------------------------
ACTION_GROUP_MINUTES = 10     # group actions within 10 minutes
NEXT_STATE_HOURS = 4          # collapse next 4 hours into a single state vector


# ----------------------------- Cohort -----------------------------
# Example: sepsis cohort via ICD10 prefixes (same idea as your teammate’s code)
SEPSIS_PREFIXES = ("A40", "A41", "R65")


# ----------------------------- State definition -----------------------------
# Pick the charted signals you want in state (extend as needed)
STATE_ITEMIDS = [
    220045,  # HR
    223762,  # Temp
    220277,  # SpO2
    220050,  # SBP
    220051,  # DBP
    220052,  # MBP
]
STATE_NAMES = ["HR", "TEMP", "SPO2", "SBP", "DBP", "MBP"]

# Collapse each variable over a window into summary stats (compact & informative)
STATE_SUMMARIES = ["mean", "min", "max", "std", "last", "trend"]  # trend = last - first


# ----------------------------- Action vector definition -----------------------------
# (i) Procedures (binary)
PROCEDURE_ACTIONS: Dict[str, set] = {
    "proc_ventilation": {225794},
    "proc_invasive_lines": {224263, 224268, 225752},
    "proc_urinary_catheter": {229351},
    "proc_intub_extub": {227194},
    "proc_dialysis": {225802},
}

# (ii) Drugs administration
# TODO: Replace with the actual ICU medication ITEMIDs you want to include.
DRUG_ACTIONS: Dict[str, set] = {
    # Examples ONLY (placeholders). Replace with your curated sets.
    # "drug_antibiotic": {...},
    # "drug_vasopressor_bolus": {...},
    # "drug_sedative_bolus": {...},
}

# (iii) Continuous meds
# Encode per continuous med category:
#   +1 start, -1 stop, +2 dose increased, -2 dose decreased, 0 no change
# TODO: Replace with correct itemids for continuous infusions you care about.
CONT_MED_ACTIONS: Dict[str, set] = {
    # Examples ONLY (placeholders). Replace with your curated sets.
    # "cont_norepinephrine": {...},
    # "cont_epinephrine": {...},
    # "cont_vasopressin": {...},
}

CONT_RATE_COL_CANDIDATES = ["rate", "rateuom", "originalrate"]  # rate column differs by pipeline; we’ll search safely
DOSE_CHANGE_EPS = 0.10  # 10% relative change threshold to count as dose increase/decrease


# ----------------------------- Helpers -----------------------------
def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, **kwargs)


def pick_rate_column(df: pd.DataFrame) -> Optional[str]:
    # Try common columns in MIMIC-ish pipelines; fall back to None
    for c in ["rate", "originalrate", "patientweight", "infusionrate"]:
        if c in df.columns:
            return c
    return None


def cluster_times(times: np.ndarray, max_gap_minutes: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Given sorted timestamps, cluster them so that consecutive events within max_gap_minutes
    belong to the same cluster.

    Returns list of (cluster_start, cluster_end) where cluster_end is the last event time
    in that cluster (not +gap).
    """
    if len(times) == 0:
        return []

    max_gap = pd.Timedelta(minutes=max_gap_minutes)
    clusters: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    start = times[0]
    last = times[0]

    for t in times[1:]:
        if (t - last) <= max_gap:
            last = t
        else:
            clusters.append((start, last))
            start = t
            last = t

    clusters.append((start, last))
    return clusters


def summarize_state_window(ce: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, float]:
    """
    Collapse chart events in [start, end) into summary features for each STATE_ITEMID.
    Returns a dict of feature_name -> value (floats, NaN allowed).
    """
    out: Dict[str, float] = {}
    window = ce[(ce["charttime"] >= start) & (ce["charttime"] < end)]

    for itemid, name in zip(STATE_ITEMIDS, STATE_NAMES):
        vals = window.loc[window["itemid"] == itemid, ["charttime", "valuenum"]].dropna()
        if vals.empty:
            for s in STATE_SUMMARIES:
                out[f"{name}_{s}"] = np.nan
            continue

        v = vals["valuenum"].astype(float)
        out[f"{name}_mean"] = float(v.mean())
        out[f"{name}_min"] = float(v.min())
        out[f"{name}_max"] = float(v.max())
        out[f"{name}_std"] = float(v.std(ddof=0)) if len(v) > 1 else 0.0

        # last + trend use charttime ordering
        vals_sorted = vals.sort_values("charttime")
        first_v = float(vals_sorted["valuenum"].iloc[0])
        last_v = float(vals_sorted["valuenum"].iloc[-1])
        out[f"{name}_last"] = last_v
        out[f"{name}_trend"] = last_v - first_v

    return out


def init_action_vector_schema() -> List[str]:
    cols = []
    cols += list(PROCEDURE_ACTIONS.keys())
    cols += list(DRUG_ACTIONS.keys())
    cols += list(CONT_MED_ACTIONS.keys())
    return cols


ACTION_COLS = init_action_vector_schema()


def empty_action_vector() -> Dict[str, int]:
    # Procedures/drugs: 0/1 ; Continuous meds: integer code but default 0
    return {c: 0 for c in ACTION_COLS}


def compute_procedure_actions(pe: pd.DataFrame, w_start: pd.Timestamp, w_end: pd.Timestamp) -> Dict[str, int]:
    out = {}
    w = pe[(pe["starttime"] >= w_start) & (pe["starttime"] <= w_end)]
    for k, ids in PROCEDURE_ACTIONS.items():
        out[k] = int((~w.empty) and w["itemid"].isin(ids).any())
    return out


def compute_drug_actions(ie: pd.DataFrame, w_start: pd.Timestamp, w_end: pd.Timestamp) -> Dict[str, int]:
    """
    For bolus/administrations, simplest encoding is binary presence of any matching itemid.
    You can change to counts/sums if you prefer.
    """
    out = {}
    if ie is None or ie.empty or len(DRUG_ACTIONS) == 0:
        return {k: 0 for k in DRUG_ACTIONS.keys()}

    w = ie[(ie["starttime"] >= w_start) & (ie["starttime"] <= w_end)]
    for k, ids in DRUG_ACTIONS.items():
        out[k] = int((not w.empty) and w["itemid"].isin(ids).any())
    return out


def compute_cont_med_actions(
    ie: pd.DataFrame,
    w_start: pd.Timestamp,
    w_end: pd.Timestamp,
    prev_rates: Dict[str, float],
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    For continuous meds:
      +1 start, -1 stop, +2 dose increased, -2 dose decreased, 0 no change
    Uses the last observed rate in the window (if present) and compares with prev_rates.
    """
    out = {k: 0 for k in CONT_MED_ACTIONS.keys()}
    if ie is None or ie.empty or len(CONT_MED_ACTIONS) == 0:
        return out, prev_rates

    rate_col = pick_rate_column(ie)
    if rate_col is None:
        # If you don’t have a usable rate column, you can’t do dose change.
        # You may still detect start/stop via presence/absence, but we keep safe defaults.
        return out, prev_rates

    w = ie[(ie["starttime"] >= w_start) & (ie["starttime"] <= w_end)].copy()
    if w.empty:
        # no new info; no action changes detected
        return out, prev_rates

    # Normalize rate to numeric where possible
    w[rate_col] = pd.to_numeric(w[rate_col], errors="coerce")

    new_prev_rates = dict(prev_rates)

    for med_name, ids in CONT_MED_ACTIONS.items():
        ww = w[w["itemid"].isin(ids)].sort_values("starttime")
        if ww.empty:
            continue

        # We take the last rate in this window as "current"
        curr_rate = ww[rate_col].dropna()
        if curr_rate.empty:
            continue
        curr = float(curr_rate.iloc[-1])

        prev = float(new_prev_rates.get(med_name, 0.0))

        # Simple start/stop + relative change
        if prev <= 0 and curr > 0:
            out[med_name] = +1
        elif prev > 0 and curr <= 0:
            out[med_name] = -1
        elif prev > 0 and curr > 0:
            if curr > prev * (1.0 + DOSE_CHANGE_EPS):
                out[med_name] = +2
            elif curr < prev * (1.0 - DOSE_CHANGE_EPS):
                out[med_name] = -2
            else:
                out[med_name] = 0

        new_prev_rates[med_name] = curr

    return out, new_prev_rates


def build_action_vector_for_cluster(
    stay_id: int,
    pe_stay: pd.DataFrame,
    ie_stay: Optional[pd.DataFrame],
    cluster_start: pd.Timestamp,
    cluster_end: pd.Timestamp,
    prev_rates: Dict[str, float],
) -> Tuple[Dict[str, int], Dict[str, float]]:
    a = empty_action_vector()

    # Procedures
    a.update(compute_procedure_actions(pe_stay, cluster_start, cluster_end))

    # Drugs (bolus/admin)
    if ie_stay is not None:
        a.update(compute_drug_actions(ie_stay, cluster_start, cluster_end))

    # Continuous meds (start/stop/dose change)
    if ie_stay is not None:
        cont, prev_rates = compute_cont_med_actions(ie_stay, cluster_start, cluster_end, prev_rates)
        a.update(cont)

    return a, prev_rates


# ----------------------------- Main pipeline -----------------------------
def main():
    print("Loading cohort tables...")
    icustays = safe_read_csv(ICUSTAYS_FILE, parse_dates=["intime", "outtime"])
    icustays = icustays.dropna(subset=["intime", "outtime"])

    admissions = safe_read_csv(ADMISSIONS_FILE)
    diagnoses = safe_read_csv(DIAGNOSES_FILE, low_memory=False)

    # Cohort selection: sepsis by ICD10 prefixes
    sepsis_dx = diagnoses[
        (diagnoses["icd_version"] == 10)
        & diagnoses["icd_code"].astype(str).str.startswith(SEPSIS_PREFIXES)
    ]
    sepsis_hadm_ids = set(sepsis_dx["hadm_id"].unique())
    icustays = icustays[icustays["hadm_id"].isin(sepsis_hadm_ids)].copy()

    # Outcome lookup: hospital_expire_flag
    admissions_idx = admissions.set_index("hadm_id")["hospital_expire_flag"]

    # If cohort is empty, stop early
    if icustays.empty:
        raise RuntimeError("Sepsis cohort is empty after filtering. Check ICD filtering and files.")

    stay_ids = set(icustays["stay_id"].astype(int).unique())
    print(f"Cohort stays: {len(stay_ids)}")

    print("Loading procedureevents and medication events (inputevents/ingredientevents) ...")
    procedureevents = safe_read_csv(PROCEDURE_FILE, parse_dates=["starttime"])
    procedureevents = procedureevents[procedureevents["stay_id"].isin(stay_ids)].copy()

    inputevents = safe_read_csv(INPUTEVENTS_FILE, parse_dates=["starttime", "endtime"], low_memory=False)
    inputevents = inputevents[inputevents["stay_id"].isin(stay_ids)].copy()

    # ingredientevents is optional
    ingredientevents = None
    if INGREDIENTEVENTS_FILE.exists():
        ingredientevents = safe_read_csv(INGREDIENTEVENTS_FILE, parse_dates=["starttime", "endtime"], low_memory=False)
        ingredientevents = ingredientevents[ingredientevents["stay_id"].isin(stay_ids)].copy()
        # Combine into one meds dataframe for simplicity
        meds_events = pd.concat([inputevents, ingredientevents], ignore_index=True)
    else:
        meds_events = inputevents

    # Ensure required columns exist (fail fast, clear errors)
    for df_name, df, cols in [
        ("procedureevents", procedureevents, ["stay_id", "starttime", "itemid"]),
        ("meds_events", meds_events, ["stay_id", "starttime", "itemid"]),
    ]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{df_name} missing required columns: {missing}")

    # ---------------- Pass 1: Extract relevant chartevents rows to Parquet parts ----------------
    print("Pass 1: Extracting relevant chartevents to Parquet parts (filtered by itemid + cohort stay_id)...")
    # Clean tmp dir if it already has old parts (optional)
    # Comment out if you want to resume.
    for f in TMP_STATE_DIR.glob("part_*.parquet"):
        f.unlink()

    chunksize = 5_000_000
    part_idx = 0

    usecols = ["stay_id", "charttime", "itemid", "valuenum"]
    for chunk in pd.read_csv(
        CHARTEVENTS_FILE,
        chunksize=chunksize,
        parse_dates=["charttime"],
        usecols=lambda c: c in usecols,  # robust if extra cols exist
        low_memory=False,
    ):
        # Filter to state itemids and cohort stay_ids
        chunk = chunk[chunk["itemid"].isin(STATE_ITEMIDS)]
        chunk = chunk[chunk["stay_id"].isin(stay_ids)]
        if chunk.empty:
            continue

        out_part = TMP_STATE_DIR / f"part_{part_idx:04d}.parquet"
        chunk.to_parquet(out_part, index=False)
        part_idx += 1
        if part_idx % 5 == 0:
            print(f"  wrote {part_idx} parquet parts...")

    if part_idx == 0:
        raise RuntimeError("No relevant chartevents found after filtering. Check STATE_ITEMIDS and cohort stay_ids.")

    print(f"Pass 1 complete: wrote {part_idx} parquet parts to {TMP_STATE_DIR}")

    # ---------------- Pass 2: Build transitions per stay ----------------
    print("Pass 2: Building transitions per stay...")
    # Load all filtered chartevents parts (still filtered heavily)
    ce_all = pd.concat(
        [pd.read_parquet(p) for p in sorted(TMP_STATE_DIR.glob("part_*.parquet"))],
        ignore_index=True,
    )
    ce_all = ce_all.dropna(subset=["stay_id", "charttime", "itemid"])
    ce_all["stay_id"] = ce_all["stay_id"].astype(int)
    ce_all["itemid"] = ce_all["itemid"].astype(int)

    # Group event tables by stay_id for quick access
    pe_by_stay = {sid: df for sid, df in procedureevents.groupby("stay_id")}
    me_by_stay = {sid: df for sid, df in meds_events.groupby("stay_id")}
    ce_by_stay = {sid: df for sid, df in ce_all.groupby("stay_id")}

    rows: List[Dict] = []

    # Precompute state feature column names
    state_feat_cols = [f"{name}_{s}" for name in STATE_NAMES for s in STATE_SUMMARIES]
    action_cols = ACTION_COLS[:]  # fixed

    for i, stay in icustays.reset_index(drop=True).iterrows():
        sid = int(stay["stay_id"])
        hadm_id = int(stay["hadm_id"])
        intime = pd.Timestamp(stay["intime"])
        outtime = pd.Timestamp(stay["outtime"])
        died = bool(admissions_idx.get(hadm_id, 0) == 1)

        # Pull per-stay dataframes (may be missing)
        ce = ce_by_stay.get(sid, pd.DataFrame(columns=["stay_id", "charttime", "itemid", "valuenum"]))
        pe = pe_by_stay.get(sid, pd.DataFrame(columns=["stay_id", "starttime", "itemid"]))
        me = me_by_stay.get(sid, pd.DataFrame(columns=["stay_id", "starttime", "itemid"]))

        # Safety: ensure time columns present and Timestamp
        if not ce.empty and "charttime" in ce.columns:
            ce = ce.sort_values("charttime")
        if not pe.empty and "starttime" in pe.columns:
            pe = pe.sort_values("starttime")
        if me is not None and not me.empty and "starttime" in me.columns:
            me = me.sort_values("starttime")

        # Collect all potential "action event times" from procedures + meds
        times = []
        if not pe.empty:
            times.append(pe["starttime"].dropna())
        if me is not None and not me.empty:
            times.append(me["starttime"].dropna())

        if len(times) == 0:
            # No actions at all; you can choose to skip or create a single terminal transition.
            # Here: create one terminal transition with zero action vector.
            s = summarize_state_window(ce, max(intime, outtime - pd.Timedelta(hours=NEXT_STATE_HOURS)), outtime)
            row = {f"s_{k}": s.get(k, np.nan) for k in state_feat_cols}
            row.update({f"s_next_{k}": 0.0 for k in state_feat_cols})
            row.update({k: 0 for k in action_cols})
            row["reward"] = -1.0 if died else 1.0
            row["done"] = 1
            rows.append(row)
            continue

        all_times = pd.to_datetime(pd.concat(times)).sort_values()
        # Limit to within ICU stay (optional but recommended)
        all_times = all_times[(all_times >= intime) & (all_times <= outtime)]
        if all_times.empty:
            # Same as no-actions case
            s = summarize_state_window(ce, max(intime, outtime - pd.Timedelta(hours=NEXT_STATE_HOURS)), outtime)
            row = {f"s_{k}": s.get(k, np.nan) for k in state_feat_cols}
            row.update({f"s_next_{k}": 0.0 for k in state_feat_cols})
            row.update({k: 0 for k in action_cols})
            row["reward"] = -1.0 if died else 1.0
            row["done"] = 1
            rows.append(row)
            continue

        clusters = cluster_times(all_times.to_numpy(), ACTION_GROUP_MINUTES)

        # Track previous infusion rates per continuous med category for dose change detection
        prev_rates = {k: 0.0 for k in CONT_MED_ACTIONS.keys()}

        # Build transitions for each action cluster
        # For each cluster: state is pre-cluster (previous 4h), next_state is post-cluster (next 4h after cluster_end)
        for ci, (c_start, c_end) in enumerate(clusters):
            # Pre-state window (previous 4 hours ending at cluster start)
            s_start = max(intime, c_start - pd.Timedelta(hours=NEXT_STATE_HOURS))
            s_end = c_start

            # Post-state window (next 4 hours AFTER the action window ends)
            ns_start = c_end
            ns_end = min(outtime, c_end + pd.Timedelta(hours=NEXT_STATE_HOURS))

            s = summarize_state_window(ce, s_start, s_end)
            ns = summarize_state_window(ce, ns_start, ns_end) if ns_end > ns_start else {k: np.nan for k in state_feat_cols}

            a, prev_rates = build_action_vector_for_cluster(sid, pe, me, c_start, c_end, prev_rates)

            # Intermediate transitions: reward 0, done 0
            row = {}
            row.update({f"s_{k}": s.get(k, np.nan) for k in state_feat_cols})
            row.update({f"s_next_{k}": ns.get(k, np.nan) for k in state_feat_cols})
            row.update({k: int(a.get(k, 0)) for k in action_cols})
            row["reward"] = 0.0
            row["done"] = 0
            rows.append(row)

        # Terminal transition: assign terminal reward at end-of-stay
        # Use last cluster’s pre-state as the terminal "state" (or you can use last 4h before outtime).
        last_cluster_start, last_cluster_end = clusters[-1]
        term_s_start = max(intime, last_cluster_start - pd.Timedelta(hours=NEXT_STATE_HOURS))
        term_s_end = last_cluster_start
        term_s = summarize_state_window(ce, term_s_start, term_s_end)

        term_row = {}
        term_row.update({f"s_{k}": term_s.get(k, np.nan) for k in state_feat_cols})
        term_row.update({f"s_next_{k}": 0.0 for k in state_feat_cols})  # absorbing placeholder
        term_row.update({k: 0 for k in action_cols})  # optional: you can also repeat last action vector if you prefer
        term_row["reward"] = -1.0 if died else 1.0
        term_row["done"] = 1
        rows.append(term_row)

        if i % 100 == 0:
            print(f"  processed {i}/{len(icustays)} stays...")

    print("Writing output parquet...")
    out_df = pd.DataFrame(rows)

    # Optional: basic sanity checks
    if out_df.empty:
        raise RuntimeError("No transitions produced.")
    if out_df.isna().all(axis=1).any():
        print("Warning: some rows are entirely NaN (check state window coverage).")

    out_df.to_parquet(OUT_PATH, index=False)
    print(f"Saved dataset to: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
