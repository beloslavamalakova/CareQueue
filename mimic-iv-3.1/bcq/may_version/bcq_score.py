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

    def forward(self, x):
        return self.net(x)


class QNetwork(nn.Module):
    """
    BCQ Q-network.
    Outputs Q(s, a) for all discrete actions.
    """
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = MLP(state_dim, n_actions, hidden=hidden, dropout=dropout)

    def forward(self, s):
        return self.net(s)


def score_initial_patients(
    data_path="./Processed/transitions.parquet",
    checkpoint_path="./Output/best.pt",
    schema_path="./Output/schema_and_norm.json",
    sample_size=10_000,
    seed=42,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    df = pd.read_parquet(data_path)

    with open(schema_path, "r") as f:
        schema = json.load(f)

    state_cols = schema["state_cols"]
    state_mean = np.array(schema["state_mean"], dtype=np.float32)
    state_std = np.array(schema["state_std"], dtype=np.float32)

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]

    hidden = cfg.get("hidden", 256)
    dropout = cfg.get("dropout", 0.0)
    n_actions = ckpt.get("n_actions", schema.get("n_actions", int(df["action"].max()) + 1))

    q = QNetwork(
        state_dim=len(state_cols),
        n_actions=n_actions,
        hidden=hidden,
        dropout=dropout,
    ).to(device)

    # Try common checkpoint key names
    if "q" in ckpt:
        q.load_state_dict(ckpt["q"])
    elif "q_net" in ckpt:
        q.load_state_dict(ckpt["q_net"])
    elif "q1" in ckpt:
        q.load_state_dict(ckpt["q1"])
    else:
        raise KeyError(f"Could not find Q-network in checkpoint. Available keys: {list(ckpt.keys())}")

    q.eval()

    df = df.sort_values(["stay_id", "bin"]).reset_index(drop=True)
    first_states = df.groupby("stay_id", as_index=False).first()

    if len(first_states) < sample_size:
        sample_size = len(first_states)

    patient_df = first_states.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    X = patient_df[state_cols].to_numpy(dtype=np.float32)
    X = (X - state_mean) / state_std
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    with torch.no_grad():
        x_t = torch.from_numpy(X).to(device)
        q_values = q(x_t)

        # BCQ patient score = best available Q-value
        patient_df["predicted_value"] = q_values.max(dim=1).values.cpu().numpy()
        patient_df["recommended_action"] = q_values.argmax(dim=1).cpu().numpy()

    # Reward scale: death = -100, survival = +100
    patient_df["p_survival"] = (patient_df["predicted_value"] / 100 + 1) / 2
    patient_df["p_survival"] = patient_df["p_survival"].clip(0, 1)

    bottom_10_threshold = patient_df["predicted_value"].quantile(0.10)
    bottom_10_df = patient_df[patient_df["predicted_value"] <= bottom_10_threshold]

    pct_survival_bottom_10 = 100 * bottom_10_df["p_survival"].mean()

    patient_df["bottom_10_worst"] = (
        patient_df["predicted_value"] <= bottom_10_threshold
    ).astype(int)

    summary = pd.DataFrame([{
        "n_patients": int(len(patient_df)),
        "mean_predicted_value": float(patient_df["predicted_value"].mean()),
        "pct_survival_bottom_10_worst_patients": float(pct_survival_bottom_10),
    }])

    return patient_df[[
        "stay_id",
        "predicted_value",
        "p_survival",
        "recommended_action",
        "bottom_10_worst",
    ]], summary


if __name__ == "__main__":
    patient_scores, summary_df = score_initial_patients(
        data_path="./Processed/transitions.parquet",
        checkpoint_path="./Output/best.pt",
        schema_path="./Output/schema_and_norm.json",
        sample_size=10_000,
        seed=42,
    )

    print("\n=== BCQ Predicted Outcome Summary ===")
    print(summary_df.to_string(index=False))

    patient_scores.to_csv("./Output/bcq_patient_scores_10000.csv", index=False)
    summary_df.to_csv("./Output/bcq_patient_summary_10000.csv", index=False)

    print("\nSaved:")
    print(" - ./Output/bcq_patient_scores_10000.csv")
    print(" - ./Output/bcq_patient_summary_10000.csv")
