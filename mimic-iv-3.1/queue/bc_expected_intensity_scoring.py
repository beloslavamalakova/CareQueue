#!/usr/bin/env python3
"""
Generate queue-compatible priority scores from a discrete Behavior Cloning model.

BC does not learn Q-values. This script therefore uses the BC policy's action
probabilities to compute an expected treatment intensity:

    priority_score = E[vasopressor_bin + fluid_bin] / max_possible_intensity

The output parquet contains one row per ICU stay and can be passed directly to
queue_simulation.py with --score_mode file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class DiscreteBCPolicy(nn.Module):
    """
    Same architecture used by mimic-iv-3.1/bc/bc_discrete.py.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: int = 256,
        dropout: float = 0.0,
    ):
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


def load_initial_patient_states(test_file: Path) -> pd.DataFrame:
    """
    Keep the first observed state per stay_id, matching the queue-entry logic.
    """

    df = pd.read_parquet(test_file)

    required = {
        "stay_id",
        "bin",
    }
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


def load_bc_policy(
    checkpoint_file: Path,
    device: torch.device,
) -> tuple[DiscreteBCPolicy, list[str], np.ndarray, np.ndarray, int]:
    """
    Load the trained BC checkpoint and rebuild the policy network.
    """

    ckpt = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=False,
    )

    required = {
        "model_state_dict",
        "state_cols",
        "state_mean",
        "state_std",
        "n_actions",
    }
    missing = required - set(ckpt.keys())

    if missing:
        raise ValueError(
            f"BC checkpoint is missing required keys: {sorted(missing)}"
        )

    state_cols = list(ckpt["state_cols"])
    state_mean = np.asarray(ckpt["state_mean"], dtype=np.float32)
    state_std = np.asarray(ckpt["state_std"], dtype=np.float32)
    n_actions = int(ckpt["n_actions"])

    if len(state_cols) != len(state_mean) or len(state_cols) != len(state_std):
        raise ValueError(
            "Checkpoint state_cols, state_mean, and state_std lengths do not match."
        )

    config = ckpt.get("config", {})
    hidden = int(config.get("hidden", 256))
    dropout = float(config.get("dropout", 0.0))

    policy = DiscreteBCPolicy(
        state_dim=len(state_cols),
        n_actions=n_actions,
        hidden=hidden,
        dropout=dropout,
    ).to(device)

    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    return policy, state_cols, state_mean, state_std, n_actions


def make_state_tensor(
    patient_df: pd.DataFrame,
    state_cols: list[str],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Normalize states exactly as the BC model saw them during training.
    """

    missing = set(state_cols) - set(patient_df.columns)

    if missing:
        raise ValueError(
            f"Test parquet is missing state columns required by checkpoint: {sorted(missing)}"
        )

    safe_std = np.where(state_std < 1e-6, 1.0, state_std)

    x = patient_df[state_cols].to_numpy(dtype=np.float32)
    x = (x - state_mean) / safe_std
    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return torch.from_numpy(x).to(device)


def build_action_intensity(
    n_actions: int,
    vaso_bins: int,
    fluid_bins: int,
) -> tuple[np.ndarray, float]:
    """
    Map action index to treatment intensity.

    For the current 5 x 5 action space:
        vaso_bin = action // 5
        fluid_bin = action % 5
        intensity = vaso_bin + fluid_bin
    """

    if vaso_bins <= 0 or fluid_bins <= 0:
        raise ValueError("--vaso_bins and --fluid_bins must be positive.")

    if n_actions != vaso_bins * fluid_bins:
        raise ValueError(
            "Checkpoint action count does not match action grid: "
            f"n_actions={n_actions}, vaso_bins*fluid_bins={vaso_bins * fluid_bins}"
        )

    intensity = np.zeros(n_actions, dtype=np.float32)

    for action in range(n_actions):
        vaso_bin = action // fluid_bins
        fluid_bin = action % fluid_bins
        intensity[action] = float(vaso_bin + fluid_bin)

    max_possible_intensity = float((vaso_bins - 1) + (fluid_bins - 1))

    if max_possible_intensity <= 0:
        raise ValueError("Max possible treatment intensity must be greater than zero.")

    return intensity, max_possible_intensity


@torch.no_grad()
def score_patients(
    patient_df: pd.DataFrame,
    policy: DiscreteBCPolicy,
    state_cols: list[str],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    n_actions: int,
    vaso_bins: int,
    fluid_bins: int,
    device: torch.device,
    include_action_probs: bool,
) -> pd.DataFrame:
    """
    Compute BC expected-intensity priority scores for initial patient states.
    """

    x = make_state_tensor(
        patient_df=patient_df,
        state_cols=state_cols,
        state_mean=state_mean,
        state_std=state_std,
        device=device,
    )

    logits = policy(x)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    action_intensity, max_possible_intensity = build_action_intensity(
        n_actions=n_actions,
        vaso_bins=vaso_bins,
        fluid_bins=fluid_bins,
    )

    raw_expected_intensity = probs @ action_intensity
    priority_score = raw_expected_intensity / max_possible_intensity

    eps = 1e-12
    action_entropy = -np.sum(
        probs * np.log(probs + eps),
        axis=1,
    )

    output = pd.DataFrame(
        {
            "stay_id": patient_df["stay_id"].astype(int).to_numpy(),
            "priority_score": priority_score.astype(np.float64),
            "raw_expected_intensity": raw_expected_intensity.astype(np.float64),
            "predicted_action": probs.argmax(axis=1).astype(int),
            "max_action_probability": probs.max(axis=1).astype(np.float64),
            "action_entropy": action_entropy.astype(np.float64),
        }
    )

    if include_action_probs:
        for action in range(n_actions):
            output[f"p_action_{action}"] = probs[:, action].astype(np.float64)

    if output["stay_id"].duplicated().any():
        raise RuntimeError("Output contains duplicate stay_id values.")

    if output["priority_score"].isna().any():
        raise RuntimeError("Output contains NaN priority_score values.")

    if not output["priority_score"].between(0.0, 1.0).all():
        raise RuntimeError("Expected all priority_score values to be between 0 and 1.")

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create queue-compatible BC priority scores using expected treatment intensity."
        )
    )

    parser.add_argument(
        "--test_file",
        required=True,
        type=str,
        help="Queue-preprocessed test.parquet file.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Trained BC checkpoint, e.g. bc_discrete_best.pt.",
    )

    parser.add_argument(
        "--output_file",
        required=True,
        type=str,
        help="Output parquet path, e.g. bc_scores.parquet.",
    )

    parser.add_argument(
        "--vaso_bins",
        type=int,
        default=5,
        help="Number of vasopressor bins in the discrete action space.",
    )

    parser.add_argument(
        "--fluid_bins",
        type=int,
        default=5,
        help="Number of fluid bins in the discrete action space.",
    )

    parser.add_argument(
        "--include_action_probs",
        action="store_true",
        help="Also save p_action_0 ... p_action_N diagnostic columns.",
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use, e.g. cpu or cuda.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    test_file = Path(args.test_file).resolve()
    checkpoint_file = Path(args.checkpoint).resolve()
    output_file = Path(args.output_file).resolve()

    device = torch.device(args.device)

    print()
    print("=" * 70)
    print("BC EXPECTED-INTENSITY QUEUE SCORING")
    print("=" * 70)
    print(f"Test file   : {test_file}")
    print(f"Checkpoint  : {checkpoint_file}")
    print(f"Output file : {output_file}")
    print(f"Device      : {device}")

    patient_df = load_initial_patient_states(test_file)

    print(f"Patients    : {len(patient_df):,}")

    policy, state_cols, state_mean, state_std, n_actions = load_bc_policy(
        checkpoint_file=checkpoint_file,
        device=device,
    )

    print(f"State dim   : {len(state_cols)}")
    print(f"Actions     : {n_actions}")
    print(f"Action grid : {args.vaso_bins} x {args.fluid_bins}")

    scores = score_patients(
        patient_df=patient_df,
        policy=policy,
        state_cols=state_cols,
        state_mean=state_mean,
        state_std=state_std,
        n_actions=n_actions,
        vaso_bins=args.vaso_bins,
        fluid_bins=args.fluid_bins,
        device=device,
        include_action_probs=args.include_action_probs,
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scores.to_parquet(
        output_file,
        index=False,
    )

    print()
    print("Score summary")
    print("-" * 70)
    print(f"Mean priority_score   : {scores['priority_score'].mean():.4f}")
    print(f"Median priority_score : {scores['priority_score'].median():.4f}")
    print(f"Min priority_score    : {scores['priority_score'].min():.4f}")
    print(f"Max priority_score    : {scores['priority_score'].max():.4f}")
    print()
    print(f"Saved: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
