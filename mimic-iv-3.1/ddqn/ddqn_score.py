"""
Adapted code from the IQL score file to evaluate the performance of the DDQN model.

"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Removed the MLP and ValueV networks, added QNet instead for DDQN (used same input param names)
class QNet(nn.Module):
    def __init__(self, state_dim, num_actions: int = 25, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x):
        return self.net(x)


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

    # Rebuild Q net and load weights
    q = QNet(
        state_dim=len(state_cols),
        num_actions=25,
        hidden=hidden,
    ).to(device)
    q.load_state_dict(ckpt["q"])
    q.eval()

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
        # For DDQN, we want the max Q(s,a)
        patient_df["predicted_value"] = q(x_t).max(dim=1).values.cpu().numpy()

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
    data_path = "/home/20243009/ddqn/sepsis_iql_actionvec_transitions.parquet"
    checkpoint_path = "/home/20243009/ddqn/ddqn_outputs/ddqn_model_best.pt"
    schema_path = "/home/20243009/ddqn/ddqn_outputs/schema_and_norm.json"

    patient_scores, summary_df = score_initial_patients(
        data_path=data_path,
        checkpoint_path=checkpoint_path,
        schema_path=schema_path,
        sample_size=10_000,
        seed=42,
    )

    print("\n=== Predicted Outcome Summary ===")
    print(summary_df.to_string(index=False))

    patient_scores.to_csv("ddqn_patient_scores_10000.csv", index=False)
    summary_df.to_csv("ddqn_patient_summary_10000.csv", index=False)

    print("\nSaved:")
    print(" - ddqn_patient_scores_10000.csv")
    print(" - ddqn_patient_summary_10000.csv")