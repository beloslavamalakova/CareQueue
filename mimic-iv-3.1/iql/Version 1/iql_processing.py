from __future__ import annotations

"""
Build an offline RL transitions dataset for IQL in (s, a, r, s') format.

This version uses definition modules for maintainability:
- definitions/state.py   : state mapping + summarization (TOP5 feature mapping)
- definitions/actions.py : action vector schema + action window logic
- definitions/reward.py  : reward definition

State columns are derived from teammates' feature_itemid_top5.csv (ICU itemids).
"""

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import definitions.state as st

import os, tempfile
tempfile.tempdir = os.environ.get("TMPDIR", None)

from definitions.state import (
    NEXT_STATE_HOURS,
    STATE_SUMMARIES,
    init_state_mapping_from_top5,
    state_feature_cols,
    summarize_state_window,
)

from definitions.actions import (
    ACTION_GROUP_MINUTES,
    ACTION_COLS,
    CONT_MED_ACTIONS,
    cluster_times,
    build_action_vector_for_cluster,
)
from definitions.reward import compute_reward, terminal_reward


# ----------------------------- Paths & I/O -----------------------------
BASE_DIR = Path(r"/home/20231942/MIMIC/mimic/raw")
ICU_DIR = BASE_DIR / "icu"
HOSP_DIR = BASE_DIR / "hosp"

ICUSTAYS_FILE = ICU_DIR / "icustays.csv.gz"
CHARTEVENTS_FILE = ICU_DIR / "chartevents.csv.gz"
PROCEDURE_FILE = ICU_DIR / "procedureevents.csv.gz"
INPUTEVENTS_FILE = ICU_DIR / "inputevents.csv.gz"
INGREDIENTEVENTS_FILE = ICU_DIR / "ingredientevents.csv.gz"  # optional; if missing, code will skip
ADMISSIONS_FILE = HOSP_DIR / "admissions.csv.gz"
DIAGNOSES_FILE = HOSP_DIR / "diagnoses_icd.csv.gz"

# Teammates' TOP5 mapping file (source of truth for state feature names/itemids)
TOP5_FEATURE_MAP_FILE = BASE_DIR / "interim" / "feature_itemid_top5.csv"

RUN_DIR = Path(os.environ.get("RUN_DIR", "."))

OUT_PATH = RUN_DIR / "sepsis_iql_actionvec_transitions.parquet"
TMP_STATE_DIR = RUN_DIR / "tmp_state_chartevents_parts"
TMP_STATE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0

# ----------------------------- Cohort -----------------------------
# Keep your existing cohort selection (sepsis via ICD10 prefixes)
SEPSIS_PREFIXES = ("A40", "A41", "R65")

import definitions.state as st
print("USING state.py from:", st.__file__)


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, **kwargs)


def main() -> None:
    print("Loading cohort tables...")
    icustays = safe_read_csv(ICUSTAYS_FILE, parse_dates=["intime", "outtime"])
    icustays = icustays.dropna(subset=["intime", "outtime"])

    admissions = safe_read_csv(ADMISSIONS_FILE, low_memory=False)
    diagnoses = safe_read_csv(DIAGNOSES_FILE, low_memory=False)

    # Cohort selection: sepsis by ICD10 prefixes
    sepsis_dx = diagnoses[
        (diagnoses["icd_version"] == 10)
        & diagnoses["icd_code"].astype(str).str.startswith(SEPSIS_PREFIXES)
    ]
    sepsis_hadm_ids = set(sepsis_dx["hadm_id"].unique())
    icustays = icustays.sample(n=len(icustays), random_state=SEED).reset_index(drop=True)
    # Keep only ICU stays whose admission (hadm_id) has a sepsis ICD code
    icustays = icustays[icustays["hadm_id"].isin(sepsis_hadm_ids)].copy()

    # optional but recommended: reset index after filtering
    icustays = icustays.reset_index(drop=True)

    # Outcome lookup: hospital_expire_flag
    admissions_idx = admissions.set_index("hadm_id")["hospital_expire_flag"]

    if icustays.empty:
        raise RuntimeError("Sepsis cohort is empty after filtering. Check ICD filtering and files.")

    stay_ids = set(icustays["stay_id"].astype(int).unique())
    print(f"Cohort stays: {len(stay_ids)}")

    # ----------------------------- Initialize state mapping from TOP5 -----------------------------
    init_state_mapping_from_top5(str(TOP5_FEATURE_MAP_FILE))
    # STATE_ITEMIDS is now populated inside definitions.state
    print(f"Loaded TOP5 ICU mapping: {len(st.STATE_ITEMIDS)} itemids, {len(state_feature_cols())//len(STATE_SUMMARIES)} features")

    # ----------------------------- Load action tables -----------------------------
    print("Loading procedureevents and medication events (inputevents/ingredientevents)...")
    procedureevents = safe_read_csv(PROCEDURE_FILE, parse_dates=["starttime"], low_memory=False)
    procedureevents = procedureevents[procedureevents["stay_id"].isin(stay_ids)].copy()

    # inputevents is required for meds; ingredientevents optional
    inputevents = safe_read_csv(INPUTEVENTS_FILE, parse_dates=["starttime", "endtime"], low_memory=False)
    inputevents = inputevents[inputevents["stay_id"].isin(stay_ids)].copy()

    ingredientevents = None
    if INGREDIENTEVENTS_FILE.exists():
        ingredientevents = safe_read_csv(INGREDIENTEVENTS_FILE, parse_dates=["starttime", "endtime"], low_memory=False)
        ingredientevents = ingredientevents[ingredientevents["stay_id"].isin(stay_ids)].copy()
        meds_events = pd.concat([inputevents, ingredientevents], ignore_index=True)
    else:
        meds_events = inputevents

    # Fail fast if required cols missing
    for df_name, df, cols in [
        ("procedureevents", procedureevents, ["stay_id", "starttime", "itemid"]),
        ("meds_events", meds_events, ["stay_id", "starttime", "itemid"]),
    ]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{df_name} missing required columns: {missing}")

    # ---------------- Pass 1: Extract relevant chartevents rows to Parquet parts ----------------
    print("Pass 1: Extracting relevant chartevents to Parquet parts (filtered by itemid + cohort stay_id).")

    # Clean old parts (ignore errors)
    # for f in TMP_STATE_DIR.glob("part_*.parquet"):
    #     try:
    #         f.unlink()
    #     except Exception:
    #         pass

    chunksize = 5_000_000
    existing_parts = sorted(TMP_STATE_DIR.glob("part_*.parquet"))
    part_idx = len(existing_parts)

    # ---- FULL RUN DEFAULTS ----
    MAX_FILTERED_ROWS = None   # set an int for debugging; None = no limit
    MAX_PARTS = None           # set an int for debugging; None = no limit
    filtered_rows = 0

    usecols = ["stay_id", "charttime", "itemid", "valuenum"]

    for chunk in pd.read_csv(
        CHARTEVENTS_FILE,
        chunksize=chunksize,
        parse_dates=["charttime"],
        usecols=lambda c: c in usecols,
        low_memory=False,
    ):
        # Filter to only itemids we care about + cohort stay_ids
        chunk = chunk[chunk["itemid"].isin(st.STATE_ITEMIDS)]
        chunk = chunk[chunk["stay_id"].isin(stay_ids)]
        if chunk.empty:
            continue

        out_part = TMP_STATE_DIR / f"part_{part_idx:04d}.parquet"

        if out_part.exists():
            print(f"  skipping existing {out_part.name}")
            part_idx += 1
            continue

        tmp_out = out_part.with_suffix(".parquet.tmp")
        chunk.to_parquet(tmp_out, index=False)
        tmp_out.rename(out_part)

        part_idx += 1
        filtered_rows += len(chunk)

        if part_idx % 5 == 0:
            print(f"  wrote {part_idx} parquet parts. filtered_rows={filtered_rows}")

        # Optional early stops (debugging)
        if MAX_PARTS is not None and part_idx >= MAX_PARTS:
            print(f"Early stop Pass 1: reached MAX_PARTS={MAX_PARTS}")
            break
        if MAX_FILTERED_ROWS is not None and filtered_rows >= MAX_FILTERED_ROWS:
            print(f"Early stop Pass 1: reached MAX_FILTERED_ROWS={MAX_FILTERED_ROWS}")
            break

    if part_idx == 0:
        raise RuntimeError(
            "No relevant chartevents found after filtering. "
            "Either the cohort stay_ids don't match this chartevents file, "
            "or STATE_ITEMIDS are not present in chartevents."
        )

    print(f"Pass 1 complete: wrote {part_idx} parquet parts to {TMP_STATE_DIR}")

    # ---------------- Pass 2: Build transitions per stay ----------------
    print("Pass 2: Building transitions per stay.")

    ce_all = pd.concat(
        [pd.read_parquet(p) for p in sorted(TMP_STATE_DIR.glob("part_*.parquet"))],
        ignore_index=True,
    )
    ce_all = ce_all.dropna(subset=["stay_id", "charttime", "itemid"])
    ce_all["stay_id"] = ce_all["stay_id"].astype(int)
    ce_all["itemid"] = ce_all["itemid"].astype(int)

    pe_by_stay = {sid: df.sort_values("starttime") for sid, df in procedureevents.groupby("stay_id")}
    me_by_stay = {sid: df.sort_values("starttime") for sid, df in meds_events.groupby("stay_id")}
    ce_by_stay = {sid: df.sort_values("charttime") for sid, df in ce_all.groupby("stay_id")}

    rows: List[Dict] = []

    state_feat_cols = state_feature_cols()  # <feature>_<summary> suffixes
    action_cols = ACTION_COLS[:]            # from definitions.actions

    for i, stay in icustays.reset_index(drop=True).iterrows():
        sid = int(stay["stay_id"])
        hadm_id = int(stay["hadm_id"])
        intime = pd.Timestamp(stay["intime"])
        outtime = pd.Timestamp(stay["outtime"])
        died = bool(admissions_idx.get(hadm_id, 0) == 1)

        ce = ce_by_stay.get(sid, pd.DataFrame(columns=["stay_id", "charttime", "itemid", "valuenum"]))
        pe = pe_by_stay.get(sid, pd.DataFrame(columns=["stay_id", "starttime", "itemid"]))
        me = me_by_stay.get(sid, pd.DataFrame(columns=["stay_id", "starttime", "itemid"]))

        # Collect all action event times from procedures + meds
        times = []
        if not pe.empty:
            times.append(pe["starttime"].dropna())
        if me is not None and not me.empty:
            times.append(me["starttime"].dropna())

        if len(times) == 0:
            # No actions: create single terminal transition with zero action vector.
            s = summarize_state_window(ce, max(intime, outtime - pd.Timedelta(hours=NEXT_STATE_HOURS)), outtime)

            row = {f"s_{k}": s.get(k, np.nan) for k in state_feat_cols}
            row.update({f"s_next_{k}": 0.0 for k in state_feat_cols})
            row.update({k: 0 for k in action_cols})
            row["reward"] = terminal_reward(died)
            row["done"] = 1
            rows.append(row)
            continue

        all_times = pd.to_datetime(pd.concat(times)).sort_values()
        all_times = all_times[(all_times >= intime) & (all_times <= outtime)]
        if all_times.empty:
            s = summarize_state_window(ce, max(intime, outtime - pd.Timedelta(hours=NEXT_STATE_HOURS)), outtime)

            row = {f"s_{k}": s.get(k, np.nan) for k in state_feat_cols}
            row.update({f"s_next_{k}": 0.0 for k in state_feat_cols})
            row.update({k: 0 for k in action_cols})
            row["reward"] = terminal_reward(died)
            row["done"] = 1
            rows.append(row)
            continue

        clusters = cluster_times(all_times.tolist(), ACTION_GROUP_MINUTES)

        # Track previous infusion rates per continuous med category for dose change detection
        prev_rates = {k: 0.0 for k in CONT_MED_ACTIONS.keys()}

        for ci, (c_start, c_end) in enumerate(clusters):
            # Pre-state: previous 4 hours ending at cluster start
            s_start = max(intime, c_start - pd.Timedelta(hours=NEXT_STATE_HOURS))
            s_end = c_start

            # Next state: next 4 hours after cluster end
            ns_start = c_end
            ns_end = min(outtime, c_end + pd.Timedelta(hours=NEXT_STATE_HOURS))

            s = summarize_state_window(ce, s_start, s_end)
            ns = summarize_state_window(ce, ns_start, ns_end) if ns_end > ns_start else {k: np.nan for k in state_feat_cols}

            a, prev_rates = build_action_vector_for_cluster(sid, pe, me, c_start, c_end, prev_rates)

            done = int(ci == len(clusters) - 1)
            reward = compute_reward(done=bool(done), died=died)

            row = {}
            row.update({f"s_{k}": s.get(k, np.nan) for k in state_feat_cols})
            row.update({f"s_next_{k}": ns.get(k, np.nan) for k in state_feat_cols})
            row.update({k: int(a.get(k, 0)) for k in action_cols})
            row["reward"] = float(reward)
            row["done"] = done
            rows.append(row)

        if i % 100 == 0:
            print(f"  processed {i}/{len(icustays)} stays.")

    print("Writing output parquet.")
    out_df = pd.DataFrame(rows)

    if out_df.empty:
        raise RuntimeError("No transitions produced.")

    out_df.to_parquet(OUT_PATH, index=False)
    print(f"Saved dataset to: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
