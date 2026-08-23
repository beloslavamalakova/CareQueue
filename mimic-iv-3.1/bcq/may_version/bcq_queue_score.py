#!/usr/bin/env python3
"""
Generate patient priority scores from a trained discrete BCQ model.

Input:
    - Common test.parquet
    - Trained BCQ checkpoint
    - BCQ normalization/schema file

For every ICU stay:
    1. Select the first observed state.
    2. Compute Q(s, a) for all actions.
    3. Apply the BCQ behavior-policy constraint.
    4. Select the highest-Q allowed action.
    5. Use its Q-value as the raw patient score.
    6. Convert raw scores to percentile ranks in [0, 1]
       for use by the common queue simulator.

Output:
    bcq_scores.parquet

Required queue columns:
    stay_id
    priority_score

Additional diagnostic columns:
    raw_bcq_score
    recommended_action
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Networks
# ============================================================

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int = 128,
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


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.net = MLP(
            state_dim,
            n_actions,
            hidden,
            dropout,
        )

    def forward(self, s):
        return self.net(s)


class BehaviorPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.net = MLP(
            state_dim,
            n_actions,
            hidden,
            dropout,
        )

    def forward(self, s):
        return self.net(s)


@torch.no_grad()
def bcq_select_actions(
    q_values: torch.Tensor,
    bc_logits: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """
    Select the highest-Q action among actions allowed by BCQ.

    An action is allowed when:

        P(a|s) / max_a P(a|s) > threshold
    """

    probs = F.softmax(
        bc_logits,
        dim=-1,
    )

    max_prob = probs.max(
        dim=1,
        keepdim=True,
    ).values

    mask = (
        probs / (max_prob + 1e-8)
    ) > threshold

    masked_q = q_values.masked_fill(
        ~mask,
        -1e9,
    )

    return masked_q.argmax(
        dim=1
    )


def load_bcq(
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

    n_actions = int(
        checkpoint.get(
            "n_actions",
            schema["n_actions"],
        )
    )

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

    q = QNetwork(
        state_dim=len(state_cols),
        n_actions=n_actions,
        hidden=hidden,
        dropout=dropout,
    ).to(device)

    bc = BehaviorPolicy(
        state_dim=len(state_cols),
        n_actions=n_actions,
        hidden=hidden,
        dropout=dropout,
    ).to(device)

    q.load_state_dict(
        checkpoint["q"]
    )

    bc.load_state_dict(
        checkpoint["bc"]
    )

    q.eval()
    bc.eval()

    return (
        q,
        bc,
        schema,
        config,
    )


def generate_scores(
    test_file: Path,
    checkpoint_path: Path,
    schema_path: Path,
    output_file: Path,
    device: str,
):

    print("=" * 60)
    print("BCQ PATIENT PRIORITY SCORING")
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

    # --------------------------------------------------------
    # One queue entry per patient:
    # first observed state of every ICU stay
    # --------------------------------------------------------

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

    # Load trained BCQ

    (
        q,
        bc,
        schema,
        config,
    ) = load_bcq(
        checkpoint_path=checkpoint_path,
        schema_path=schema_path,
        device=device,
    )

    state_cols = schema[
        "state_cols"
    ]

    # Make sure test file contains exactly the state
    # features expected by this trained model.

    missing_state_cols = (
        set(state_cols)
        - set(first_states.columns)
    )

    if missing_state_cols:
        raise ValueError(
            "Test file is missing state columns expected "
            f"by BCQ: {sorted(missing_state_cols)}"
        )


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


    threshold = float(
        config.get(
            "bcq_threshold",
            0.3,
        )
    )

    print(
        f"BCQ threshold   : {threshold}"
    )

    with torch.no_grad():

        states = torch.from_numpy(
            X
        ).to(device)

        q_values = q(
            states
        )

        bc_logits = bc(
            states
        )

        selected_actions = (
            bcq_select_actions(
                q_values=q_values,
                bc_logits=bc_logits,
                threshold=threshold,
            )
        )

        selected_q = (
            q_values
            .gather(
                1,
                selected_actions.view(
                    -1,
                    1,
                ),
            )
            .squeeze(1)
        )

        raw_scores = (
            selected_q
            .cpu()
            .numpy()
        )

        actions = (
            selected_actions
            .cpu()
            .numpy()
        )

    scores = pd.DataFrame(
        {
            "stay_id":
                first_states[
                    "stay_id"
                ].astype(int),

            "raw_bcq_score":
                raw_scores.astype(float),

            "recommended_action":
                actions.astype(int),
        }
    )

    # Normalize scores for common queue
    # Percentile ranking gives each model a common [0,1]
    # scoring scale while preserving its patient ordering.
    # Higher raw BCQ value -> higher normalized priority.

    if len(scores) == 1:

        scores[
            "priority_score"
        ] = 1.0

    else:

        ranks = scores[
            "raw_bcq_score"
        ].rank(
            method="average",
        )

        scores[
            "priority_score"
        ] = (
            (ranks - 1)
            / (len(scores) - 1)
        )


    scores = scores[
        [
            "stay_id",
            "priority_score",
            "raw_bcq_score",
            "recommended_action",
        ]
    ]


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
        f"Raw Q mean      : "
        f"{scores['raw_bcq_score'].mean():.4f}"
    )

    print(
        f"Raw Q min       : "
        f"{scores['raw_bcq_score'].min():.4f}"
    )

    print(
        f"Raw Q max       : "
        f"{scores['raw_bcq_score'].max():.4f}"
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
            "Generate BCQ priority scores for "
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
        default="../../queue/bcq_scores.parquet",
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
