#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, dropout: float = 0.0):
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


class ValueV(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.mlp = MLP(state_dim, 1, hidden=hidden, dropout=dropout)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.mlp(s)


def score_initial_patients(
    data_path: str,
    checkpoint_path: str,
    schema_path: str,
    sample_size: int = 10_000,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Load processed transitions
    df = pd.read_parquet(data_path)

    # Load schema and normalization
    with open(schema_path, "r") as f:
        schema = json.load(f)

    state_cols = schema["state_cols"]
    state_mean = np.array(schema["state_mean"], dtype=np.float32)
    state_std = np.array(schema["state_std"], dtype=np.float32)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]

    hidden = cfg["hidden"]
    dropout = cfg["dropout"]

    # Rebuild value net and load weights
    v = ValueV(
        state_dim=len(state_cols),
        hidden=hidden,
        dropout=dropout
    ).to(device)
    v.load_state_dict(ckpt["v"])
    v.eval()

    # Sort so first bin is the initial state
    df = df.sort_values(["stay_id", "bin"]).reset_index(drop=True)

    # Keep only first state per patient
    first_states = df.groupby("stay_id", as_index=False).first()

    # Sample 10,000 unique patients
    if len(first_states) < sample_size:
        raise ValueError(f"Only {len(first_states)} patients available, need {sample_size}")

    patient_df = first_states.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    # Normalize initial states
    X = patient_df[state_cols].to_numpy(dtype=np.float32)
    X = (X - state_mean) / state_std
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Predict V(s0)
    with torch.no_grad():
        x_t = torch.from_numpy(X).to(device)
        patient_df["predicted_value"] = v(x_t).squeeze(-1).cpu().numpy()

    # Convert predicted value to approximate survival probability
    # Assumes reward scale: death = -100, survival = +100
    patient_df["p_survival"] = (patient_df["predicted_value"] / 100 + 1) / 2

    # Bottom 10% worst patients = patients with the lowest predicted values
    bottom_10_threshold = patient_df["predicted_value"].quantile(0.10)
    bottom_10_df = patient_df[patient_df["predicted_value"] <= bottom_10_threshold]

    # Percentage survival of the bottom 10% worst patients
    pct_survival_bottom_10 = 100 * bottom_10_df["p_survival"].mean()

    # Optional flag for patient-level CSV
    patient_df["bottom_10_worst"] = (
        patient_df["predicted_value"] <= bottom_10_threshold
    ).astype(int)

    # Summary metrics
    summary = pd.DataFrame([{
        "n_patients": int(len(patient_df)),
        "mean_predicted_value": float(patient_df["predicted_value"].mean()),
        "pct_survival_bottom_10_worst_patients": float(pct_survival_bottom_10),
    }])

    # Return both patient-level and summary-level outputs
    return patient_df[[
        "stay_id",
        "predicted_value",
        "p_survival",
        "bottom_10_worst",
    ]], summary


if __name__ == "__main__":
    data_path = r"C:\Users\20231942\Desktop\Central Folder\TUe\Year 3\Honors\Code\CareQueue\mimic-iv-3.1\iql\Version 3\Processed\transitions.parquet"
    checkpoint_path = r"C:\Users\20231942\Desktop\Central Folder\TUe\Year 3\Honors\Code\CareQueue\mimic-iv-3.1\iql\Version 3\Output\best.pt"
    schema_path = r"C:\Users\20231942\Desktop\Central Folder\TUe\Year 3\Honors\Code\CareQueue\mimic-iv-3.1\iql\Version 3\Output\schema_and_norm.json"

    patient_scores, summary_df = score_initial_patients(
        data_path=data_path,
        checkpoint_path=checkpoint_path,
        schema_path=schema_path,
        sample_size=10_000,
        seed=42,
    )

    print("\n=== Predicted Outcome Summary ===")
    print(summary_df.to_string(index=False))

    patient_scores.to_csv("iql_patient_scores_10000.csv", index=False)
    summary_df.to_csv("iql_patient_summary_10000.csv", index=False)

    print("\nSaved:")
    print(" - iql_patient_scores_10000.csv")
    print(" - iql_patient_summary_10000.csv")