from __future__ import annotations
from typing import Dict, List, Tuple, Optional, cast
import numpy as np
import pandas as pd


# ----------------------------- Time settings -----------------------------
# (State window length is also used for next-state window length in your pipeline.)
NEXT_STATE_HOURS = 4  # collapse next 4 hours into a single state vector

# Collapse each variable over a window into summary stats (compact & informative)
STATE_SUMMARIES = ["mean", "min", "max", "std", "last", "trend"]  # trend = last - first


# ----------------------------- TOP5 feature mapping globals -----------------------------
# These get initialized by init_state_mapping_from_top5(...)
ICU_ITEMID_TO_FEATURE: Dict[int, str] = {}
FEATURES: List[str] = []
STATE_ITEMIDS: List[int] = []


def sanitize_feature_name(s: str) -> str:
    """Match teammates' sanitize logic (see merging_files.py)."""
    s = str(s).strip().lower()
    s = s.replace(" ", "_")
    s = s.replace("/", "_")
    return s


def load_state_mapping_from_top5(top5_path: str) -> Tuple[Dict[int, str], List[str]]:
    top5_df = pd.read_csv(top5_path)

    required = {"feature", "source", "id_type", "id"}
    missing = required - set(top5_df.columns)
    if missing:
        raise ValueError(
            f"TOP5 file missing columns: {missing}. Found columns: {list(top5_df.columns)}"
        )

    df = top5_df.copy()

    # --- Make string fields robust ---
    df["source"] = df["source"].astype(str).str.strip().str.lower()
    df["id_type"] = df["id_type"].astype(str).str.strip().str.lower()

    # --- Keep only ICU chart itemids (for chartevents) ---
    # Handles both exact "icu/d_items" and things like "icu/d_items.csv.gz"
    icu_mask = df["source"].str.contains(r"icu/d_items", na=False)
    df = df[(df["id_type"] == "itemid") & icu_mask].copy()

    # sanitize feature names
    df["feature"] = df["feature"].map(sanitize_feature_name)

    # numeric itemids
    df["itemid"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["itemid"])
    df["itemid"] = df["itemid"].astype(int)

    # IMPORTANT: build BOTH itemid_to_feature and features from the SAME ICU-filtered df
    itemids: List[int] = df["itemid"].astype(int).tolist()
    feats: List[str] = df["feature"].astype(str).tolist()

    itemid_to_feature: Dict[int, str] = dict(zip(itemids, feats))
    features: List[str] = sorted(set(feats))

    if not itemid_to_feature:
        # Helpful debug prints
        print("DEBUG: No ICU itemids found. Here are unique values:")
        print("  unique source (first 30):", top5_df["source"].astype(str).str.strip().unique()[:30])
        print("  unique id_type:", top5_df["id_type"].astype(str).str.strip().unique())
        raise RuntimeError(
            "No ICU itemids found in TOP5 mapping after filtering. "
            "Check 'source' contains 'icu/d_items' and id_type == 'itemid'."
        )

    return itemid_to_feature, features
    

def init_state_mapping_from_top5(top5_path: str) -> None:
    """Initialize module globals ICU_ITEMID_TO_FEATURE, FEATURES, STATE_ITEMIDS."""
    global ICU_ITEMID_TO_FEATURE, FEATURES, STATE_ITEMIDS
    ICU_ITEMID_TO_FEATURE, FEATURES = load_state_mapping_from_top5(top5_path)
    STATE_ITEMIDS = sorted(ICU_ITEMID_TO_FEATURE.keys())


def state_feature_cols() -> List[str]:
    """
    Return the flattened list of state feature column suffixes:
      <feature>_<summary>
    (without the s_ / s_next_ prefix).
    """
    if not FEATURES:
        raise RuntimeError("State FEATURES not initialized. Call init_state_mapping_from_top5(...) first.")
    return [f"{feat}_{s}" for feat in FEATURES for s in STATE_SUMMARIES]


def summarize_state_window(ce: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, float]:
    """
    Collapse chart events in [start, end) into summary features per *feature name*.

    IMPORTANT:
      - The mapping (itemid -> feature) must be initialized via init_state_mapping_from_top5.
      - Multiple itemids may map to the same feature; we aggregate across all such values.

    Returns keys like:
      <feature>_mean, <feature>_min, <feature>_max, <feature>_std, <feature>_last, <feature>_trend
    """
    if not FEATURES or not ICU_ITEMID_TO_FEATURE:
        raise RuntimeError("State mapping not initialized. Call init_state_mapping_from_top5(...) before summarizing.")

    out: Dict[str, float] = {}

    window = ce[(ce["charttime"] >= start) & (ce["charttime"] < end)].copy()
    if window.empty:
        for f in FEATURES:
            for s in STATE_SUMMARIES:
                out[f"{f}_{s}"] = np.nan
        return out

    # Map itemid -> feature name; drop anything not in mapping
    window["feature"] = window["itemid"].map(ICU_ITEMID_TO_FEATURE)
    window = window.dropna(subset=["feature"])
    if window.empty:
        for f in FEATURES:
            for s in STATE_SUMMARIES:
                out[f"{f}_{s}"] = np.nan
        return out

    # Ensure numeric values
    window["valuenum"] = pd.to_numeric(window["valuenum"], errors="coerce")
    window = window.dropna(subset=["valuenum"])

    for f in FEATURES:
        vals = window.loc[window["feature"] == f, ["charttime", "valuenum"]].dropna()
        if vals.empty:
            for s in STATE_SUMMARIES:
                out[f"{f}_{s}"] = np.nan
            continue

        v = vals["valuenum"].astype(float)
        out[f"{f}_mean"] = float(v.mean())
        out[f"{f}_min"] = float(v.min())
        out[f"{f}_max"] = float(v.max())
        out[f"{f}_std"] = float(v.std(ddof=0)) if len(v) > 1 else 0.0

        vals_sorted = vals.sort_values("charttime")
        first_v = float(vals_sorted["valuenum"].iloc[0])
        last_v = float(vals_sorted["valuenum"].iloc[-1])
        out[f"{f}_last"] = last_v
        out[f"{f}_trend"] = last_v - first_v

    return out
