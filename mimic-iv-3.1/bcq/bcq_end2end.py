#!/usr/bin/env python3
"""
End-to-end discrete BCQ training from MIMIC transitions parquet.

Does:
  1) split transitions into train/val by stay_id (deterministic hash split)
  2) fit normalization (mean/std) on TRAIN only
  3) train Discrete BCQ with streaming Parquet batches (pyarrow.dataset)
  4) logs per-epoch metrics to <outdir>/metrics.csv

Expected input columns (from your DuckDB transitions builder):
  stay_id, bin, action, reward, done,
  s_*, s_next_*
"""

import os
import json
import math
import argparse
from pathlib import Path
import csv
from datetime import datetime

import numpy as np
import duckdb
import pyarrow.dataset as ds

import torch
import torch.nn as nn
import torch.optim as optim


# ----------------------------- Models -----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------- Utilities -----------------------------

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def duckdb_path_exists(path: str) -> bool:
    return Path(path).exists()


def infer_state_columns_from_parquet(parquet_path: str):
    dataset = ds.dataset(parquet_path, format="parquet")
    cols = dataset.schema.names
    s_cols = [c for c in cols if c.startswith("s_") and not c.startswith("s_next_")]
    s2_cols = [c for c in cols if c.startswith("s_next_")]
    s_cols = sorted(s_cols)
    s2_cols = sorted(s2_cols)
    return cols, s_cols, s2_cols


def split_train_val_by_stay_id(inp: str, train_out: str, val_out: str, train_frac: float = 0.9):
    """
    Deterministic split using hash(stay_id) % 100.
    No random seeds, stable across runs.
    """
    con = duckdb.connect()
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA memory_limit='8GB';")
    con.execute("PRAGMA temp_directory='./duckdb_tmp';")

    thr = int(train_frac * 100)

    con.execute(f"""
    COPY (
      SELECT *
      FROM read_parquet('{inp}')
      WHERE (abs(hash(stay_id)) % 100) < {thr}
    ) TO '{train_out}' (FORMAT PARQUET);
    """)

    con.execute(f"""
    COPY (
      SELECT *
      FROM read_parquet('{inp}')
      WHERE (abs(hash(stay_id)) % 100) >= {thr}
    ) TO '{val_out}' (FORMAT PARQUET);
    """)


def fit_normalizer_on_train(train_path: str, s_cols: list, s2_cols: list, scaler_out: str):
    """
    Fit mean/std on TRAIN only.
    Returns dict and also saves to scaler_out JSON.
    """
    con = duckdb.connect()
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA memory_limit='8GB';")
    con.execute("PRAGMA temp_directory='./duckdb_tmp';")

    all_cols = s_cols + s2_cols
    stats = {}

    for c in all_cols:
        mean, std = con.execute(
            f"SELECT avg({c}), stddev_pop({c}) FROM read_parquet('{train_path}')"
        ).fetchone()

        if std is None or std == 0 or (isinstance(std, float) and not np.isfinite(std)):
            std = 1.0
        if mean is None or (isinstance(mean, float) and not np.isfinite(mean)):
            mean = 0.0

        stats[c] = {"mean": float(mean), "std": float(std)}

    ensure_parent(Path(scaler_out))
    with open(scaler_out, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def compute_action_weights(train_path: str, num_actions: int):
    """
    Class weights for behavior cloning to reduce 'always action=0' collapse.
    Returns torch tensor of shape [num_actions].
    """
    con = duckdb.connect()
    counts = con.execute(f"""
      SELECT action, count(*) AS c
      FROM read_parquet('{train_path}')
      GROUP BY action
      ORDER BY action
    """).fetchall()

    c = np.ones(num_actions, dtype=np.float64)
    for a, cnt in counts:
        a = int(a)
        if 0 <= a < num_actions:
            c[a] = float(cnt)

    inv = 1.0 / np.sqrt(c)
    inv = inv / inv.mean()
    return torch.tensor(inv, dtype=torch.float32)


def batch_iterator(parquet_path: str, columns: list, batch_size: int):
    dataset = ds.dataset(parquet_path, format="parquet")
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)
    for record_batch in scanner.to_batches():
        yield record_batch


def record_batch_to_tensors(rb, s_cols, s2_cols, stats, device):
    def col_as_numpy(name):
        arr = rb.column(rb.schema.get_field_index(name))
        return np.array(arr.to_pylist(), dtype=np.float32)

    s = np.stack([col_as_numpy(c) for c in s_cols], axis=1)
    s2 = np.stack([col_as_numpy(c) for c in s2_cols], axis=1)

    for j, c in enumerate(s_cols):
        m, sd = stats[c]["mean"], stats[c]["std"]
        s[:, j] = (s[:, j] - m) / sd

    for j, c in enumerate(s2_cols):
        m, sd = stats[c]["mean"], stats[c]["std"]
        s2[:, j] = (s2[:, j] - m) / sd

    action = np.array(rb.column(rb.schema.get_field_index("action")).to_pylist(), dtype=np.int64)
    reward = np.array(rb.column(rb.schema.get_field_index("reward")).to_pylist(), dtype=np.float32)
    done = np.array(rb.column(rb.schema.get_field_index("done")).to_pylist(), dtype=np.float32)

    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    s2 = np.nan_to_num(s2, nan=0.0, posinf=0.0, neginf=0.0)

    S = torch.tensor(s, dtype=torch.float32, device=device)
    S2 = torch.tensor(s2, dtype=torch.float32, device=device)
    A = torch.tensor(action, dtype=torch.long, device=device)
    R = torch.tensor(reward, dtype=torch.float32, device=device)
    D = torch.tensor(done, dtype=torch.float32, device=device)
    return S, A, R, S2, D


@torch.no_grad()
def eval_epoch(qnet, bcnet, stats, val_path, s_cols, s2_cols, device, gamma, tau, batch_size, max_batches=200):
    qnet.eval()
    bcnet.eval()

    ce = nn.CrossEntropyLoss(reduction="mean")
    mse = nn.MSELoss(reduction="mean")

    bc_losses = []
    q_losses = []

    cols_needed = ["action", "reward", "done"] + s_cols + s2_cols
    for i, rb in enumerate(batch_iterator(val_path, cols_needed, batch_size)):
        if i >= max_batches:
            break

        S, A, R, S2, D = record_batch_to_tensors(rb, s_cols, s2_cols, stats, device)

        logits = bcnet(S)
        bc_losses.append(ce(logits, A).item())

        probs2 = torch.softmax(bcnet(S2), dim=1)
        mask = probs2 > tau
        q2 = qnet(S2).masked_fill(~mask, -1e9)
        target = R + (1.0 - D) * gamma * q2.max(dim=1).values

        qsa = qnet(S).gather(1, A.unsqueeze(1)).squeeze(1)
        q_losses.append(mse(qsa, target).item())

    qnet.train()
    bcnet.train()

    return float(np.mean(bc_losses)) if bc_losses else math.nan, float(np.mean(q_losses)) if q_losses else math.nan


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="cache/transitions_4h.parquet", help="Input transitions parquet")
    ap.add_argument("--outdir", default="cache/bcq_run", help="Output dir for splits, scaler, checkpoints")
    ap.add_argument("--train_frac", type=float, default=0.9)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.3, help="BCQ action filter threshold")
    ap.add_argument("--num_actions", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval_batches", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=1)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_path = str(outdir / "transitions_train.parquet")
    val_path = str(outdir / "transitions_val.parquet")
    scaler_path = str(outdir / "scaler_train.json")

    if not duckdb_path_exists(args.inp):
        raise FileNotFoundError(f"Input parquet not found: {args.inp}")

    print("[1/4] Splitting train/val by stay_id (deterministic)...")
    split_train_val_by_stay_id(args.inp, train_path, val_path, train_frac=args.train_frac)

    print("[2/4] Inferring state columns...")
    _, s_cols, s2_cols = infer_state_columns_from_parquet(train_path)
    if not s_cols or not s2_cols:
        raise RuntimeError("Could not find s_* and s_next_* columns in parquet. Check your transitions schema.")

    print(f"  State dim = {len(s_cols)}")
    print(f"  Using columns (state): {s_cols}")
    print(f"  Using columns (next):  {s2_cols}")

    print("[3/4] Fitting normalizer on TRAIN only...")
    stats = fit_normalizer_on_train(train_path, s_cols, s2_cols, scaler_path)
    print(f"  Saved scaler: {scaler_path}")

    # ---- metrics CSV logging ----
    metrics_path = outdir / "metrics.csv"
    write_header = not metrics_path.exists()
    metrics_f = open(metrics_path, "a", newline="")
    metrics_w = csv.DictWriter(metrics_f, fieldnames=[
        "timestamp", "epoch",
        "train_bc", "train_q", "val_bc", "val_q",
        "n_batches",
        "tau", "gamma", "lr", "batch_size", "hidden", "num_actions", "device"
    ])
    if write_header:
        metrics_w.writeheader()
        metrics_f.flush()
    print(f"[log] writing epoch metrics to: {metrics_path}")

    print("[4/4] Training Discrete BCQ (streaming batches)...")
    device = torch.device(args.device)
    print(f"  Device: {device}")

    action_w = compute_action_weights(train_path, args.num_actions).to(device)
    bc_loss_fn = nn.CrossEntropyLoss(weight=action_w)
    q_loss_fn = nn.MSELoss()

    d = len(s_cols)
    qnet = MLP(d, args.num_actions, hidden=args.hidden).to(device)
    q_tgt = MLP(d, args.num_actions, hidden=args.hidden).to(device)
    q_tgt.load_state_dict(qnet.state_dict())
    bcnet = MLP(d, args.num_actions, hidden=args.hidden).to(device)

    opt_q = optim.Adam(qnet.parameters(), lr=args.lr)
    opt_bc = optim.Adam(bcnet.parameters(), lr=args.lr)

    cols_needed = ["action", "reward", "done"] + s_cols + s2_cols

    try:
        for ep in range(1, args.epochs + 1):
            qnet.train()
            bcnet.train()

            bc_losses = []
            q_losses = []
            n_batches = 0

            for rb in batch_iterator(train_path, cols_needed, args.batch_size):
                S, A, R, S2, D = record_batch_to_tensors(rb, s_cols, s2_cols, stats, device)

                # BC step
                logits = bcnet(S)
                loss_bc = bc_loss_fn(logits, A)
                opt_bc.zero_grad()
                loss_bc.backward()
                opt_bc.step()

                # Q step (BCQ filtered max on next state)
                with torch.no_grad():
                    probs2 = torch.softmax(bcnet(S2), dim=1)
                    mask = probs2 > args.tau
                    q2 = q_tgt(S2).masked_fill(~mask, -1e9)
                    target = R + (1.0 - D) * args.gamma * q2.max(dim=1).values

                qsa = qnet(S).gather(1, A.unsqueeze(1)).squeeze(1)
                loss_q = q_loss_fn(qsa, target)

                opt_q.zero_grad()
                loss_q.backward()
                opt_q.step()

                bc_losses.append(loss_bc.item())
                q_losses.append(loss_q.item())
                n_batches += 1

            # target update
            q_tgt.load_state_dict(qnet.state_dict())

            # eval
            bc_val, q_val = eval_epoch(
                qnet, bcnet, stats, val_path, s_cols, s2_cols,
                device, args.gamma, args.tau,
                args.batch_size, max_batches=args.eval_batches
            )

            train_bc = float(np.mean(bc_losses)) if bc_losses else math.nan
            train_q = float(np.mean(q_losses)) if q_losses else math.nan

            # log row
            metrics_w.writerow({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "epoch": ep,
                "train_bc": train_bc,
                "train_q": train_q,
                "val_bc": float(bc_val),
                "val_q": float(q_val),
                "n_batches": int(n_batches),
                "tau": float(args.tau),
                "gamma": float(args.gamma),
                "lr": float(args.lr),
                "batch_size": int(args.batch_size),
                "hidden": int(args.hidden),
                "num_actions": int(args.num_actions),
                "device": str(device),
            })
            metrics_f.flush()

            print(
                f"Epoch {ep:03d} | "
                f"train: BC={train_bc:.4f} Q={train_q:.4f} ({n_batches} batches) | "
                f"val: BC={bc_val:.4f} Q={q_val:.4f}"
            )

            # checkpoint
            if args.save_every > 0 and (ep % args.save_every == 0):
                ckpt = {
                    "epoch": ep,
                    "qnet": qnet.state_dict(),
                    "q_tgt": q_tgt.state_dict(),
                    "bcnet": bcnet.state_dict(),
                    "s_cols": s_cols,
                    "s2_cols": s2_cols,
                    "num_actions": args.num_actions,
                    "gamma": args.gamma,
                    "tau": args.tau,
                }
                ckpt_path = outdir / f"bcq_ckpt_epoch_{ep:03d}.pt"
                torch.save(ckpt, ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")

    finally:
        metrics_f.close()

    print("Done.")


if __name__ == "__main__":
    main()
