from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import pandas as pd

# ----------------------------- Time settings -----------------------------
ACTION_GROUP_MINUTES = 10  # group actions within 10 minutes


# ----------------------------- Actions definition -----------------------------
# (i) Procedure actions
PROCEDURE_ACTIONS: Dict[str, set] = {
    "proc_mech_ventilation": {225794},
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


def pick_rate_column(df: pd.DataFrame) -> Optional[str]:
    for c in CONT_RATE_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def cluster_times(times: List[pd.Timestamp], group_minutes: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Cluster timestamps into windows such that consecutive times within group_minutes are in the same cluster.
    Returns list of (cluster_start, cluster_end) timestamps.
    """
    times = [pd.Timestamp(t) for t in times]
    times = sorted(times)
    if not times:
        return []

    clusters = []
    start = times[0]
    last = times[0]

    for t in times[1:]:
        if (t - last) <= pd.Timedelta(minutes=group_minutes):
            last = t
        else:
            clusters.append((start, last))
            start = t
            last = t

    clusters.append((start, last))
    return clusters


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

