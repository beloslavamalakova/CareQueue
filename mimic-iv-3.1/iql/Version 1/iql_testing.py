#!/usr/bin/env python3
"""
iql_severity_test.py

Drop-in "testing" script:
- Pick a random ICU stay from MIMIC (or a random state from a transitions parquet),
- build the *initial state* using the same TOP5 feature mapping + summary stats as training,
- load a trained IQL checkpoint (best.pt/final.pt) and schema_and_norm.json,
- compute V(s) and return a severity score.

Severity options implemented:
  1) severity_raw = -V(s) (bigger = worse)

"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Import architecture from your training script so the Value network matches exactly.
# (This avoids mismatch bugs if you change hidden sizes, dropout, etc.)
from iql_training import ValueV

# Reuse the SAME state construction logic (TOP5 mapping + summarize_state_window)
from definitions.state import (
    NEXT_STATE_HOURS,
    STATE_ITEMIDS,
    STATE_SUMMARIES,
    init_state_mapping_from_top5,
    state_feature_cols,
    summarize_state_window,
)

try:
    from sklearn.linear_model import LogisticRegression
except Exception:
    LogisticRegression = None


# -------------------------- Utilities --------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, **kwargs)


def load_schema(run_dir: Path) -> dict:
    schema_path = run_dir / "schema_and_norm.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing schema file: {schema_path}")
    with open(schema_path, "r") as f:
        return json.load(f)


def load_value_network(run_dir: Path, ckpt_name: str, device: str = "cpu") -> Tuple[ValueV, dict]:
    """
    Loads ValueV from checkpoint and returns (v_net, schema).
    """
    schema = load_schema(run_dir)

    ckpt_path = run_dir / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    hidden = int(cfg.get("hidden", 256))
    dropout = float(cfg.get("dropout", 0.0))

    state_cols = schema["state_cols"]
    v = ValueV(state_dim=len(state_cols), hidden=hidden, dropout=dropout).to(device)
    v.load_state_dict(ckpt["v"])
    v.eval()
    return v, schema


def vectorize_and_normalize_state(
    state_dict: Dict[str, float],
    state_cols: List[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    state_dict: keys are full column names (e.g. "s_hr_mean", ...)
    returns: normalized float32 vector aligned to state_cols
    """
    x = np.array([state_dict.get(c, np.nan) for c in state_cols], dtype=np.float32)
    x = (x - mean) / std
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32)


@torch.no_grad()
def compute_v_and_severity(
    v_net: ValueV,
    x_norm: np.ndarray,
    device: str = "cpu",
) -> Tuple[float, float]:
    """
    Returns (V(s), severity_raw=-V(s)).
    """
    t = torch.from_numpy(x_norm).unsqueeze(0).to(device)  # [1, d]
    v = float(v_net(t).item())
    return v, -v


# -------------------------- Cohort distribution for percentiles --------------------------

@dataclass
class SeverityDistribution:
    """
    Stores sorted severity values so we can compute percentile ranks quickly.
    """
    sorted_sev: np.ndarray  # ascending

    def percentile(self, sev: float) -> float:
        # percentile rank of sev among sorted_sev
        # returns 0..100
        idx = np.searchsorted(self.sorted_sev, sev, side="left")
        return 100.0 * idx / max(1, len(self.sorted_sev) - 1)


def build_severity_distribution_from_parquet(
    transitions_path: Path,
    v_net: ValueV,
    schema: dict,
    device: str,
    sample_rows: int = 20000,
    seed: int = 0,
) -> SeverityDistribution:
    """
    Computes severity=-V(s) for a random sample of states from the transitions parquet.
    This lets you report severity percentiles (0..100).

    Uses state columns from schema_and_norm.json, and normalization stats from schema.
    """
    if not transitions_path.exists():
        raise FileNotFoundError(f"Missing transitions parquet: {transitions_path}")

    df = pd.read_parquet(transitions_path, engine="pyarrow")
    n = len(df)
    if n == 0:
        raise RuntimeError("Transitions parquet is empty.")

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample_rows, n), replace=False)
    df_s = df.iloc[idx].reset_index(drop=True)

    state_cols = schema["state_cols"]
    mean = np.array(schema["state_mean"], dtype=np.float32)
    std = np.array(schema["state_std"], dtype=np.float32)

    # Vectorize in bulk for speed
    X = df_s[state_cols].to_numpy(dtype=np.float32)
    X = (X - mean) / std
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Batch through V(s)
    batch = 4096
    sevs = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i+batch]).to(device)
        vb = v_net(xb).detach().cpu().numpy().reshape(-1)
        sevs.append(-vb)  # severity = -V
    sevs = np.concatenate(sevs, axis=0)
    sevs_sorted = np.sort(sevs.astype(np.float32))
    return SeverityDistribution(sorted_sev=sevs_sorted)

# -------------------------- Random patient from MIMIC: build initial state --------------------------

def load_chartevents_for_stay(
    chartevents_file: Path,
    stay_id: int,
    itemids: List[int],
    chunksize: int = 2_000_000,
) -> pd.DataFrame:
    """
    Stream-read chartevents and return only rows for one stay_id and the needed itemids.
    This is slow but okay for "one random patient testing".
    """
    usecols = ["stay_id", "charttime", "itemid", "valuenum"]
    out = []

    for chunk in pd.read_csv(
        chartevents_file,
        chunksize=chunksize,
        usecols=lambda c: c in usecols,
        parse_dates=["charttime"],
        low_memory=False,
    ):
        chunk = chunk[(chunk["stay_id"] == stay_id) & (chunk["itemid"].isin(itemids))]
        if not chunk.empty:
            out.append(chunk)

    if not out:
        return pd.DataFrame(columns=usecols)

    df = pd.concat(out, ignore_index=True)
    df["stay_id"] = df["stay_id"].astype(int)
    df["itemid"] = df["itemid"].astype(int)
    return df


def pick_random_stay(icustays_file: Path, seed: int = 0) -> Tuple[int, pd.Timestamp]:
    icu = safe_read_csv(icustays_file, parse_dates=["intime", "outtime"], low_memory=False)
    icu = icu.dropna(subset=["stay_id", "intime", "outtime"])
    rng = np.random.default_rng(seed)
    row = icu.iloc[int(rng.integers(0, len(icu)))]
    return int(row["stay_id"]), pd.Timestamp(row["intime"])


def build_initial_state_for_stay(
    mimic_base_dir: Path,
    stay_id: int,
    intime: pd.Timestamp,
    top5_feature_map: Path,
) -> Dict[str, float]:
    """
    Builds the initial state for a stay using the FIRST 4 HOURS after ICU admission:
      [intime, intime + NEXT_STATE_HOURS)

    Returns a dict whose keys match the training schema, i.e. "s_<feature>_<summary>".
    """
    # Initialize mapping (creates STATE_ITEMIDS, FEATURES internally)
    init_state_mapping_from_top5(str(top5_feature_map))

    chartevents_file = mimic_base_dir / "icu" / "chartevents.csv.gz"
    ce = load_chartevents_for_stay(chartevents_file, stay_id, STATE_ITEMIDS)

    start = intime
    end = intime + pd.Timedelta(hours=NEXT_STATE_HOURS)

    # summarize_state_window returns keys like "<feat>_mean" (no "s_" prefix)
    s_suffix = summarize_state_window(ce, start, end)

    # Add s_ prefix to match trainer schema (state_cols are "s_*")
    return {f"s_{k}": float(v) if v is not None else np.nan for k, v in s_suffix.items()}


# -------------------------- Main CLI --------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True, help="Directory containing best.pt/final.pt and schema_and_norm.json")
    p.add_argument("--ckpt", type=str, default="best.pt", choices=["best.pt", "final.pt"], help="Which checkpoint to use")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")

    # Choose how to sample a state
    p.add_argument("--mode", type=str, default="mimic", choices=["mimic", "parquet"], help="Sample state from MIMIC or from transitions parquet")

    # MIMIC mode inputs
    p.add_argument("--mimic_base_dir", type=str, default="/home/20243009/mimic-iv-3.1", help="Base MIMIC-IV directory with icu/ and hosp/")
    p.add_argument("--top5_map", type=str, default="/home/20243009/mimic-iv-3.1/interim/feature_itemid_top5.csv", help="Path to feature_itemid_top5.csv")

    # Parquet mode inputs
    p.add_argument("--transitions", type=str, default="sepsis_iql_actionvec_transitions.parquet", help="Path to transitions parquet")

    # Severity extras
    p.add_argument("--with_percentile", action="store_true", help="Compute severity percentile (builds distribution from transitions sample)")
    p.add_argument("--with_calibration", action="store_true", help="Fit logistic calibration on terminal transitions (requires sklearn)")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample_rows", type=int, default=20000, help="Rows to sample for percentile distribution")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_dir = Path(args.run_dir)
    device = args.device

    v_net, schema = load_value_network(run_dir, args.ckpt, device=device)

    # Sample a state
    state_cols = schema["state_cols"]
    mean = np.array(schema["state_mean"], dtype=np.float32)
    std = np.array(schema["state_std"], dtype=np.float32)

    if args.mode == "parquet":
        df = pd.read_parquet(Path(args.transitions), engine="pyarrow")
        if df.empty:
            raise RuntimeError("Transitions parquet is empty.")
        row = df.sample(n=1, random_state=args.seed).iloc[0]
        state_dict = {c: float(row[c]) if c in row else np.nan for c in state_cols}
        meta = {"source": "parquet_random_row"}
    else:
        mimic_base = Path(args.mimic_base_dir)
        icustays_file = mimic_base / "icu" / "icustays.csv.gz"
        stay_id, intime = pick_random_stay(icustays_file, seed=args.seed)
        state_dict = build_initial_state_for_stay(
            mimic_base_dir=mimic_base,
            stay_id=stay_id,
            intime=intime,
            top5_feature_map=Path(args.top5_map),
        )
        meta = {"source": "mimic_random_stay", "stay_id": stay_id, "intime": str(intime)}

    # Compute V(s) + severity
    x = vectorize_and_normalize_state(state_dict, state_cols, mean, std)
    v, sev = compute_v_and_severity(v_net, x, device=device)

    print("---- Result ----")
    for k, v0 in meta.items():
        print(f"{k}: {v0}")
    print(f"V(s): {v:.6f}")
    print(f"severity_raw = -V(s): {sev:.6f}  (higher = worse)")

    print("----------------")


if __name__ == "__main__":
    main()
