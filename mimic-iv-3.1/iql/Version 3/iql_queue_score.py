#!/usr/bin/env python3
"""
Generate patient priority scores from a trained IQL model.

Input:
    - Common test.parquet
    - Trained IQL checkpoint
    - IQL normalization/schema file

For every ICU stay:
    1. Select the first observed state.
    2. Compute V(s0) using the trained IQL value network.
    3. Use V(s0) as the raw patient score.
    4. Convert raw scores to percentile ranks in [0, 1].

Output:
    iql_scores.parquet

Required queue columns:
    stay_id
    priority_score

Additional diagnostic column:
    raw_iql_score
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# Network definitions

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int = 256,
        dropout: float = 0.0,
    ):
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


class ValueV(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.mlp = MLP(
            state_dim,
            1,
            hidden=hidden,
            dropout=dropout,
        )

    def forward(self, s):
        return self.mlp(s)


# Load trained IQL value model

def load_iql_value(
    checkpoint_path: Path,
    schema_path: Path,
    device: str,
):

    with open(schema_path, "r") as f:
        schema = json.load(f)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    config = checkpoint["config"]

    state_cols = schema["state_cols"]

    hidden = int(
        config.get(
            "hidden",
            128,
        )
    )

    dropout = float(
        config.get(
            "dropout",
            0.0,
        )
    )

    v = ValueV(
        state_dim=len(state_cols),
        hidden=hidden,
        dropout=dropout,
    ).to(device)

    if "v" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain IQL value network key 'v'."
        )

    v.load_state_dict(
        checkpoint["v"]
    )

    v.eval()

    return v, schema


# Generate patient scores

def generate_scores(
    test_file: Path,
    checkpoint_path: Path,
    schema_path: Path,
    output_file: Path,
    device: str,
):

    print("=" * 60)
    print("IQL PATIENT PRIORITY SCORING")
    print("=" * 60)

    print(
        f"Test file       : {test_file}"
    )

    print(
        f"Checkpoint      : {checkpoint_path}"
    )

    print(
        f"Schema          : {schema_path}"
    )

    print(
        f"Device          : {device}"
    )

    # Load common test set

    df = pd.read_parquet(
        test_file
    )

    required = {
        "stay_id",
        "bin",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Test file missing columns: {sorted(missing)}"
        )

    # First state of each patient

    first_states = (
        df.sort_values(
            [
                "stay_id",
                "bin",
            ]
        )
        .groupby(
            "stay_id",
            as_index=False,
        )
        .first()
    )

    print(
        f"Test transitions : {len(df):,}"
    )

    print(
        f"Test patients    : {len(first_states):,}"
    )

    # Load IQL value network

    v, schema = load_iql_value(
        checkpoint_path=checkpoint_path,
        schema_path=schema_path,
        device=device,
    )

    state_cols = schema[
        "state_cols"
    ]

    missing_state_cols = (
        set(state_cols)
        - set(first_states.columns)
    )

    if missing_state_cols:
        raise ValueError(
            "Test file is missing state columns expected "
            f"by IQL: {sorted(missing_state_cols)}"
        )

    # Normalize using training statistics

    state_mean = np.asarray(
        schema["state_mean"],
        dtype=np.float32,
    )

    state_std = np.asarray(
        schema["state_std"],
        dtype=np.float32,
    )

    X = first_states[
        state_cols
    ].to_numpy(
        dtype=np.float32
    )

    X = (
        X - state_mean
    ) / state_std

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Compute V(s0)

    with torch.no_grad():

        states = torch.from_numpy(
            X
        ).to(device)

        raw_values = (
            v(states)
            .squeeze(-1)
            .cpu()
            .numpy()
        )

    # Build score dataframe

    scores = pd.DataFrame(
        {
            "stay_id":
                first_states[
                    "stay_id"
                ].astype(int),

            "raw_iql_score":
                raw_values.astype(float),
        }
    )

    if len(scores) == 1:

        scores[
            "priority_score"
        ] = 1.0

    else:

        ranks = scores[
            "raw_iql_score"
        ].rank(
            method="average"
        )

        scores[
            "priority_score"
        ] = (
            (ranks - 1)
            / (len(scores) - 1)
        )

    # Queue-required columns first

    scores = scores[
        [
            "stay_id",
            "priority_score",
            "raw_iql_score",
        ]
    ]
    # Validation

    if scores[
        "stay_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate stay_id values found."
        )

    if scores[
        "priority_score"
    ].isna().any():

        raise ValueError(
            "NaN priority scores generated."
        )

    if scores[
        "raw_iql_score"
    ].isna().any():

        raise ValueError(
            "NaN raw IQL scores generated."
        )

    if len(scores) != len(
        first_states
    ):

        raise RuntimeError(
            "Not every patient received a score."
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
    print("=" * 60)
    print("SCORING COMPLETE")
    print("=" * 60)

    print(
        f"Patients scored : {len(scores):,}"
    )

    print(
        f"Raw V mean      : "
        f"{scores['raw_iql_score'].mean():.4f}"
    )

    print(
        f"Raw V min       : "
        f"{scores['raw_iql_score'].min():.4f}"
    )

    print(
        f"Raw V max       : "
        f"{scores['raw_iql_score'].max():.4f}"
    )

    print(
        f"Priority min    : "
        f"{scores['priority_score'].min():.4f}"
    )

    print(
        f"Priority max    : "
        f"{scores['priority_score'].max():.4f}"
    )

    print()
    print(
        f"Saved to        : {output_file}"
    )

    print("=" * 60)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate IQL priority scores for "
            "the common patient queue."
        )
    )

    parser.add_argument(
        "--test_file",
        type=str,
        default="../../queue/test.parquet",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="Output_common/best.pt",
    )

    parser.add_argument(
        "--schema",
        type=str,
        default="Output_common/schema_and_norm.json",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="../../queue/iql_scores.parquet",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    args = parser.parse_args()

    generate_scores(
        test_file=Path(
            args.test_file
        ).resolve(),

        checkpoint_path=Path(
            args.checkpoint
        ).resolve(),

        schema_path=Path(
            args.schema
        ).resolve(),

        output_file=Path(
            args.output
        ).resolve(),

        device=args.device,
    )


if __name__ == "__main__":
    main()
