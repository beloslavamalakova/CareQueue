#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class DiscreteBCPolicy(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        h = self.trunk(s)
        return self.head(h)


def score_initial_patients_bc(
    data_path: str,
    checkpoint_path: str,
    sample_size: int = 10_000,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Load transitions
    df = pd.read_parquet(data_path)

    # Load BC checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_cols = ckpt["state_cols"]
    state_mean = np.array(ckpt["state_mean"], dtype=np.float32)
    state_std = np.array(ckpt["state_std"], dtype=np.float32)
    n_actions = int(ckpt["n_actions"])

    cfg = ckpt["config"]
    hidden = cfg.get("hidden", 256)
    dropout = cfg.get("dropout", 0.0)

    # Rebuild BC policy
    policy = DiscreteBCPolicy(
        state_dim=len(state_cols),
        n_actions=n_actions,
        hidden=hidden,
        dropout=dropout,
    ).to(device)

    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    # Sort so first bin is the initial state
    if "bin" in df.columns:
        df = df.sort_values(["stay_id", "bin"]).reset_index(drop=True)
    else:
        print("[warning] No 'bin' column found. Using first row per stay_id based on current file order.")
        df = df.reset_index(drop=True)

    # Keep only first state per patient
    first_states = df.groupby("stay_id", as_index=False).first()

    # Sample patients
    if len(first_states) < sample_size:
        raise ValueError(f"Only {len(first_states)} patients available, need {sample_size}")

    patient_df = first_states.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    # Normalize initial states
    X = patient_df[state_cols].to_numpy(dtype=np.float32)
    X = (X - state_mean) / state_std
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Predict action probabilities
    with torch.no_grad():
        x_t = torch.from_numpy(X).to(device)
        logits = policy(x_t)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    # Add one column per action probability
    for a in range(n_actions):
        patient_df[f"p_action_{a}"] = probs[:, a]

    # Predicted action = most likely clinician-like action
    patient_df["predicted_action"] = probs.argmax(axis=1)

    # Confidence = probability assigned to predicted action
    patient_df["max_action_probability"] = probs.max(axis=1)

    # Entropy = uncertainty over actions
    # Higher entropy = model is less certain
    eps = 1e-12
    patient_df["action_entropy"] = -np.sum(probs * np.log(probs + eps), axis=1)

    # Optional: bottom 10% confidence patients
    # These are NOT "worst survival" patients.
    # They are patients where the BC model is least confident about clinician behavior.
    low_conf_threshold = patient_df["max_action_probability"].quantile(0.10)
    patient_df["bottom_10_low_confidence"] = (
        patient_df["max_action_probability"] <= low_conf_threshold
    ).astype(int)

    # Summary
    action_counts = patient_df["predicted_action"].value_counts(normalize=True).sort_index()

    summary_dict = {
        "n_patients": int(len(patient_df)),
        "mean_max_action_probability": float(patient_df["max_action_probability"].mean()),
        "mean_action_entropy": float(patient_df["action_entropy"].mean()),
        "low_confidence_threshold_bottom_10": float(low_conf_threshold),
    }

    for a in range(n_actions):
        summary_dict[f"predicted_action_{a}_rate"] = float(action_counts.get(a, 0.0))

    summary_df = pd.DataFrame([summary_dict])

    output_cols = (
        ["stay_id", "predicted_action", "max_action_probability", "action_entropy", "bottom_10_low_confidence"]
        + [f"p_action_{a}" for a in range(n_actions)]
    )

    return patient_df[output_cols], summary_df


if __name__ == "__main__":
    data_path = r"C:\Users\20243322\OneDrive - TU Eindhoven\Desktop\A - honors\CareQueue\mimic-iv-3.1\bc\25_actions_bc_sepsis.parquet"

    checkpoint_path = r"C:\Users\20243322\OneDrive - TU Eindhoven\Desktop\A - honors\CareQueue\mimic-iv-3.1\bc\bc_runs\discrete\bc_discrete_best.pt"

    patient_scores, summary_df = score_initial_patients_bc(
        data_path=data_path,
        checkpoint_path=checkpoint_path,
        sample_size=5_000,
        seed=42,
    )

    print("\n=== BC Action Probability Summary ===")
    print(summary_df.to_string(index=False))

    patient_scores.to_csv("bc_patient_action_probs_5000.csv", index=False)
    summary_df.to_csv("bc_patient_action_summary_5000.csv", index=False)

    print("\nSaved:")
    print(" - bc_patient_action_probs_5000.csv")
    print(" - bc_patient_action_summary_5000.csv")