'''
Run it like this:
python mimic-iv-3.1/evaluate_policy_comparison.py ^
  --data mimic-iv-3.1/bc/25_actions_bc_sepsis.parquet ^
  --bc-actions bc_patient_action_probs_5000.csv ^
  --bcq-ckpt mimic-iv-3.1/bcq/cache/bcq_run/bcq_ckpt_epoch_010.pt ^
  --bcq-scaler mimic-iv-3.1/bcq/cache/bcq_run/scaler_train.json ^
  --out policy_comparison.csv

Will return a table like this:
Model | Most Common Action | Action 0 Rate | Agreement with BC | Mean Q-value

for BC the Mean Q-value will become NaN since BC doesn't learn Q-values
'''



#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_initial_states(data_path, bc_actions_csv):
    df = pd.read_parquet(data_path)
    df = df.sort_values(["stay_id", "bin"]).reset_index(drop=True)

    first_states = df.groupby("stay_id", as_index=False).first()

    bc = pd.read_csv(bc_actions_csv)
    bc = bc[["stay_id", "predicted_action"]].rename(
        columns={"predicted_action": "bc_action"}
    )

    eval_df = first_states.merge(bc, on="stay_id", how="inner")

    if len(eval_df) == 0:
        raise ValueError("No matching stay_id values between parquet and BC CSV.")

    print(f"Loaded {len(eval_df)} matched initial patient states.")
    return eval_df


def make_state_tensor_from_scaler(eval_df, s_cols, scaler_json, device):
    with open(scaler_json, "r") as f:
        stats = json.load(f)

    x = eval_df[s_cols].to_numpy(dtype=np.float32)

    for j, c in enumerate(s_cols):
        mean = stats[c]["mean"]
        std = stats[c]["std"]
        if std == 0:
            std = 1.0
        x[:, j] = (x[:, j] - mean) / std

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.tensor(x, dtype=torch.float32, device=device)


def make_state_tensor_from_ckpt(eval_df, state_cols, state_mean, state_std, device):
    x = eval_df[state_cols].to_numpy(dtype=np.float32)
    x = (x - state_mean) / state_std
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.tensor(x, dtype=torch.float32, device=device)


def summarize_model(model_name, actions, values, bc_actions):
    actions = np.asarray(actions)
    values = np.asarray(values)
    bc_actions = np.asarray(bc_actions)

    return {
        "Model": model_name,
        "Most Common Action": int(pd.Series(actions).mode().iloc[0]),
        "Action 0 Rate": float((actions == 0).mean()),
        "Agreement with BC": float((actions == bc_actions).mean()),
        "Mean Q-value": float(values.mean()),
    }


@torch.no_grad()
def eval_bcq(eval_df, ckpt_path, scaler_json, hidden, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    s_cols = ckpt["s_cols"]
    num_actions = int(ckpt["num_actions"])
    tau = float(ckpt.get("tau", 0.3))

    qnet = MLP(len(s_cols), num_actions, hidden=hidden).to(device)
    bcnet = MLP(len(s_cols), num_actions, hidden=hidden).to(device)

    qnet.load_state_dict(ckpt["qnet"])
    bcnet.load_state_dict(ckpt["bcnet"])

    qnet.eval()
    bcnet.eval()

    x = make_state_tensor_from_scaler(eval_df, s_cols, scaler_json, device)

    q = qnet(x)
    probs = torch.softmax(bcnet(x), dim=1)

    allowed = probs > tau
    masked_q = q.clone()
    masked_q[~allowed] = -1e9

    actions = masked_q.argmax(dim=1).cpu().numpy()
    values = masked_q.max(dim=1).values.cpu().numpy()

    return summarize_model(
        "BCQ",
        actions,
        values,
        eval_df["bc_action"].to_numpy(),
    )


@torch.no_grad()
def eval_generic_q_model(model_name, eval_df, ckpt_path, hidden, device):
    """
    Use this for DDQN/IQL if the checkpoint contains:
      - state_cols
      - state_mean
      - state_std
      - model_state_dict OR qnet OR q_state_dict
      - n_actions OR num_actions
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    state_cols = ckpt.get("state_cols") or ckpt.get("s_cols")
    if state_cols is None:
        raise ValueError(f"{model_name}: checkpoint has no state_cols or s_cols.")

    n_actions = int(ckpt.get("n_actions", ckpt.get("num_actions", 25)))

    state_mean = np.array(ckpt["state_mean"], dtype=np.float32)
    state_std = np.array(ckpt["state_std"], dtype=np.float32)

    state_dict = (
        ckpt.get("model_state_dict")
        or ckpt.get("qnet")
        or ckpt.get("q_state_dict")
    )

    if state_dict is None:
        raise ValueError(
            f"{model_name}: could not find model_state_dict, qnet, or q_state_dict."
        )

    model = MLP(len(state_cols), n_actions, hidden=hidden).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    x = make_state_tensor_from_ckpt(
        eval_df,
        state_cols,
        state_mean,
        state_std,
        device,
    )

    q = model(x)
    actions = q.argmax(dim=1).cpu().numpy()
    values = q.max(dim=1).values.cpu().numpy()

    return summarize_model(
        model_name,
        actions,
        values,
        eval_df["bc_action"].to_numpy(),
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", required=True, help="Same 25-action parquet used for training.")
    ap.add_argument("--bc-actions", required=True, help="bc_patient_action_probs_5000.csv")

    ap.add_argument("--bcq-ckpt", default=None)
    ap.add_argument("--bcq-scaler", default=None)

    ap.add_argument("--iql-ckpt", default=None)
    ap.add_argument("--ddqn-ckpt", default=None)

    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--out", default="policy_comparison.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()
    device = torch.device(args.device)

    eval_df = load_initial_states(args.data, args.bc_actions)

    rows = []

    rows.append({
        "Model": "BC",
        "Most Common Action": int(pd.Series(eval_df["bc_action"]).mode().iloc[0]),
        "Action 0 Rate": float((eval_df["bc_action"] == 0).mean()),
        "Agreement with BC": 1.0,
        "Mean Q-value": np.nan,
    })

    if args.bcq_ckpt:
        if not args.bcq_scaler:
            raise ValueError("For BCQ, provide --bcq-scaler path to scaler_train.json.")
        rows.append(
            eval_bcq(
                eval_df=eval_df,
                ckpt_path=args.bcq_ckpt,
                scaler_json=args.bcq_scaler,
                hidden=args.hidden,
                device=device,
            )
        )

    if args.iql_ckpt:
        rows.append(
            eval_generic_q_model(
                model_name="IQL",
                eval_df=eval_df,
                ckpt_path=args.iql_ckpt,
                hidden=args.hidden,
                device=device,
            )
        )

    if args.ddqn_ckpt:
        rows.append(
            eval_generic_q_model(
                model_name="DDQN",
                eval_df=eval_df,
                ckpt_path=args.ddqn_ckpt,
                hidden=args.hidden,
                device=device,
            )
        )

    table = pd.DataFrame(rows)

    display_table = table.copy()
    display_table["Action 0 Rate"] = display_table["Action 0 Rate"].apply(
        lambda x: f"{100*x:.2f}%"
    )
    display_table["Agreement with BC"] = display_table["Agreement with BC"].apply(
        lambda x: f"{100*x:.2f}%"
    )

    print("\n=== Policy Comparison ===")
    print(display_table.to_string(index=False))

    table.to_csv(args.out, index=False)
    print(f"\nSaved raw numeric table to: {args.out}")


if __name__ == "__main__":
    main()