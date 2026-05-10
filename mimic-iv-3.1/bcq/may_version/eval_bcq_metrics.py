#!/usr/bin/env python3
"""
Offline evaluation utilities for discrete BCQ.

Outputs:
  - bcq_patient_scores.csv
  - bcq_patient_summary.csv
  - bcq_policy_metrics.csv

Main patient-level score:
  predicted_value = max_a Q(s0, a)
using the initial ICU state s0 for each stay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = MLP(state_dim, n_actions, hidden, dropout)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class BehaviorPolicy(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = MLP(state_dim, n_actions, hidden, dropout)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


@torch.no_grad()
def bcq_action(q_values: torch.Tensor, bc_logits: torch.Tensor, threshold: float) -> torch.Tensor:
    probs = F.softmax(bc_logits, dim=-1)
    max_prob = probs.max(dim=1, keepdim=True).values
    mask = probs / (max_prob + 1e-8) > threshold
    masked_q = q_values.masked_fill(~mask, -1e9)
    return masked_q.argmax(dim=1)


def load_model(checkpoint_path: str, schema_path: str, device: str):
    with open(schema_path, "r") as f:
        schema = json.load(f)
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    n_actions = int(ckpt.get("n_actions", schema["n_actions"]))
    hidden = int(cfg.get("hidden", 128))
    dropout = float(cfg.get("dropout", 0.0))

    q = QNetwork(len(schema["state_cols"]), n_actions, hidden, dropout).to(device)
    bc = BehaviorPolicy(len(schema["state_cols"]), n_actions, hidden, dropout).to(device)
    q.load_state_dict(ckpt["q"])
    bc.load_state_dict(ckpt["bc"])
    q.eval(); bc.eval()
    return q, bc, schema, cfg


def normalize_states(df: pd.DataFrame, schema: Dict, prefix: str = "state") -> np.ndarray:
    if prefix == "state":
        cols = schema["state_cols"]
        mean = np.array(schema["state_mean"], dtype=np.float32)
        std = np.array(schema["state_std"], dtype=np.float32)
    else:
        cols = schema["next_state_cols"]
        mean = np.array(schema["next_state_mean"], dtype=np.float32)
        std = np.array(schema["next_state_std"], dtype=np.float32)
    x = df[cols].to_numpy(dtype=np.float32)
    x = (x - mean) / std
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def compute_patient_scores(data_path: str, checkpoint_path: str, schema_path: str, outdir: str, sample_size: int = 10_000, seed: int = 42, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(data_path).sort_values(["stay_id", "bin"]).reset_index(drop=True)
    q, bc, schema, cfg = load_model(checkpoint_path, schema_path, device)

    first_states = df.groupby("stay_id", as_index=False).first()
    if len(first_states) > sample_size:
        first_states = first_states.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    x = normalize_states(first_states, schema, prefix="state")
    with torch.no_grad():
        s_t = torch.from_numpy(x).to(device)
        q_values = q(s_t)
        logits = bc(s_t)
        actions = bcq_action(q_values, logits, float(cfg.get("bcq_threshold", 0.3)))
        values = q_values.gather(1, actions.view(-1, 1)).squeeze(1).cpu().numpy()
        action_np = actions.cpu().numpy()

    patient_df = pd.DataFrame({
        "stay_id": first_states["stay_id"].to_numpy(),
        "predicted_value": values,
        "bcq_action": action_np,
    })
    patient_df["p_survival"] = (patient_df["predicted_value"] / 100.0 + 1.0) / 2.0
    patient_df["p_survival"] = patient_df["p_survival"].clip(0.0, 1.0)
    threshold = patient_df["predicted_value"].quantile(0.10)
    patient_df["bottom_10_worst"] = (patient_df["predicted_value"] <= threshold).astype(int)
    bottom = patient_df[patient_df["bottom_10_worst"] == 1]

    summary = pd.DataFrame([{
        "n_patients": int(len(patient_df)),
        "mean_predicted_value": float(patient_df["predicted_value"].mean()),
        "pct_survival_bottom_10_worst_patients": float(100 * bottom["p_survival"].mean()),
    }])

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    patient_df.to_csv(out / "bcq_patient_scores.csv", index=False)
    summary.to_csv(out / "bcq_patient_summary.csv", index=False)
    return patient_df, summary


def compute_policy_metrics(data_path: str, checkpoint_path: str, schema_path: str, outdir: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> pd.DataFrame:
    df = pd.read_parquet(data_path).reset_index(drop=True)
    q, bc, schema, cfg = load_model(checkpoint_path, schema_path, device)
    x = normalize_states(df, schema, prefix="state")
    clinician_actions = df["action"].to_numpy(dtype=np.int64)

    rows = []
    batch = 8192
    all_bcq_actions = []
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(df), batch):
            s_t = torch.from_numpy(x[i:i + batch]).to(device)
            q_values = q(s_t)
            logits = bc(s_t)
            probs = F.softmax(logits, dim=-1)
            actions = bcq_action(q_values, logits, float(cfg.get("bcq_threshold", 0.3)))
            all_bcq_actions.append(actions.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    bcq_actions = np.concatenate(all_bcq_actions)
    probs = np.concatenate(all_probs)
    action_match = float(np.mean(bcq_actions == clinician_actions))

    # Behavior likelihood of selected BCQ actions and clinician actions.
    p_bcq = probs[np.arange(len(df)), bcq_actions]
    p_clin = probs[np.arange(len(df)), clinician_actions]
    mean_behavior_prob_bcq = float(np.mean(p_bcq))
    mean_behavior_prob_clinician = float(np.mean(p_clin))

    # Approximate KL between empirical BCQ action distribution and clinician distribution.
    n_actions = probs.shape[1]
    eps = 1e-8
    bcq_dist = np.bincount(bcq_actions, minlength=n_actions).astype(np.float64)
    clin_dist = np.bincount(clinician_actions, minlength=n_actions).astype(np.float64)
    bcq_dist = bcq_dist / max(bcq_dist.sum(), 1)
    clin_dist = clin_dist / max(clin_dist.sum(), 1)
    kl_bcq_vs_clin = float(np.sum(bcq_dist * (np.log(bcq_dist + eps) - np.log(clin_dist + eps))))

    metrics = pd.DataFrame([{
        "n_transitions": int(len(df)),
        "action_match_rate_vs_clinician": action_match,
        "mean_behavior_prob_bcq_action": mean_behavior_prob_bcq,
        "mean_behavior_prob_clinician_action": mean_behavior_prob_clinician,
        "kl_bcq_action_dist_vs_clinician": kl_bcq_vs_clin,
    }])

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "bcq_policy_metrics.csv", index=False)
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="Processed/transitions.parquet")
    p.add_argument("--checkpoint", default="Output/best.pt")
    p.add_argument("--schema", default="Output/schema_and_norm.json")
    p.add_argument("--outdir", default="Evaluation")
    p.add_argument("--sample_size", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    patient_scores, summary = compute_patient_scores(args.data, args.checkpoint, args.schema, args.outdir, args.sample_size, args.seed, args.device)
    metrics = compute_policy_metrics(args.data, args.checkpoint, args.schema, args.outdir, args.device)
    print("\n=== BCQ Patient Summary ===")
    print(summary.to_string(index=False))
    print("\n=== BCQ Policy Metrics ===")
    print(metrics.to_string(index=False))
    print(f"\nSaved outputs to {args.outdir}")


if __name__ == "__main__":
    main()
