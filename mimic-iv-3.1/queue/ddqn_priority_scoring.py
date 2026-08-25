#!/usr/bin/env python3
"""Generate queue-compatible priority scores from a trained DDQN model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class QNet(nn.Module):
    """Network architecture used by mimic-iv-3.1/ddqn/final_ddqn.py."""

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


def load_initial_patient_states(test_file: Path) -> pd.DataFrame:
    """Keep the first observed state per ICU stay for queue entry."""

    df = pd.read_parquet(test_file)
    required = {"stay_id", "bin"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Test parquet is missing required columns: {sorted(missing)}"
        )

    patients = (
        df.sort_values(["stay_id", "bin"])
        .groupby("stay_id", as_index=False)
        .first()
    )

    if patients["stay_id"].duplicated().any():
        raise RuntimeError("Expected exactly one row per stay_id after grouping.")

    return patients


def load_schema(schema_file: Path) -> tuple[list[str], np.ndarray, np.ndarray, int]:
    with schema_file.open("r", encoding="utf-8") as file:
        schema = json.load(file)

    required = {"state_cols", "state_mean", "state_std", "n_actions"}
    missing = required - set(schema)
    if missing:
        raise ValueError(f"Schema is missing required keys: {sorted(missing)}")

    state_cols = list(schema["state_cols"])
    state_mean = np.asarray(schema["state_mean"], dtype=np.float32)
    state_std = np.asarray(schema["state_std"], dtype=np.float32)
    n_actions = int(schema["n_actions"])

    if len(state_cols) != len(state_mean) or len(state_cols) != len(state_std):
        raise ValueError(
            "Schema state_cols, state_mean, and state_std lengths do not match."
        )
    if n_actions <= 0:
        raise ValueError("Schema n_actions must be positive.")

    return state_cols, state_mean, state_std, n_actions


def load_q_network(
    checkpoint_file: Path,
    state_dim: int,
    n_actions: int,
    device: torch.device,
) -> QNet:
    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=False,
    )
    if "q" not in checkpoint:
        raise ValueError("DDQN checkpoint is missing the 'q' state dictionary.")

    config = checkpoint.get("config", {})
    hidden = int(config.get("hidden", 256))
    network = QNet(state_dim, n_actions, hidden).to(device)
    network.load_state_dict(checkpoint["q"])
    network.eval()
    return network


def make_state_tensor(
    patient_df: pd.DataFrame,
    state_cols: list[str],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    missing = set(state_cols) - set(patient_df.columns)
    if missing:
        raise ValueError(
            f"Test parquet is missing state columns required by schema: {sorted(missing)}"
        )

    safe_std = np.where(state_std < 1e-6, 1.0, state_std)
    states = patient_df[state_cols].to_numpy(dtype=np.float32)
    states = (states - state_mean) / safe_std
    states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(states).to(device)


@torch.no_grad()
def score_patients(
    patient_df: pd.DataFrame,
    network: QNet,
    state_cols: list[str],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    device: torch.device,
) -> pd.DataFrame:
    states = make_state_tensor(
        patient_df, state_cols, state_mean, state_std, device
    )
    q_values = network(states).cpu().numpy()
    predicted_actions = q_values.argmax(axis=1)
    predicted_values = q_values.max(axis=1)

    output = pd.DataFrame(
        {
            "stay_id": patient_df["stay_id"].astype(int).to_numpy(),
            "priority_score": predicted_values.astype(np.float64),
            "predicted_value": predicted_values.astype(np.float64),
            "predicted_action": predicted_actions.astype(int),
        }
    )

    if output["stay_id"].duplicated().any():
        raise RuntimeError("Output contains duplicate stay_id values.")
    if not np.isfinite(output["priority_score"]).all():
        raise RuntimeError("Output contains non-finite priority scores.")

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create queue-compatible DDQN priority scores from max Q-values."
    )
    parser.add_argument("--test_file", required=True, help="Shared test.parquet file.")
    parser.add_argument("--checkpoint", required=True, help="Trained DDQN checkpoint.")
    parser.add_argument(
        "--schema",
        required=True,
        help="schema_and_norm.json produced with the DDQN training data.",
    )
    parser.add_argument("--output_file", required=True, help="Output parquet path.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cpu or cuda.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_file = Path(args.test_file).resolve()
    checkpoint_file = Path(args.checkpoint).resolve()
    schema_file = Path(args.schema).resolve()
    output_file = Path(args.output_file).resolve()
    device = torch.device(args.device)

    patient_df = load_initial_patient_states(test_file)
    state_cols, state_mean, state_std, n_actions = load_schema(schema_file)
    network = load_q_network(
        checkpoint_file, len(state_cols), n_actions, device
    )
    scores = score_patients(
        patient_df, network, state_cols, state_mean, state_std, device
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(output_file, index=False)

    print(f"Patients scored      : {len(scores):,}")
    print(f"Mean priority_score   : {scores['priority_score'].mean():.4f}")
    print(f"Median priority_score : {scores['priority_score'].median():.4f}")
    print(f"Saved                 : {output_file}")


if __name__ == "__main__":
    main()