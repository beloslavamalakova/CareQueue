#!/usr/bin/env python3
"""
Discrete-action BCQ training for offline MIMIC-IV transitions.

Expected parquet columns:
  s_*         : state features
  s_next_*    : next state features
  action      : integer in [0, n_actions-1]
  reward      : float
  done        : 0/1

BCQ idea:
  1. Learn a behavior cloning model G(a|s)
  2. Learn Q(s,a) with Double DQN-style targets
  3. At target/action-selection time, only allow actions whose behavior probability
     is close to the maximum behavior probability in that state:
         G(a|s) / max_a G(a|s) > bcq_threshold
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class BCQConfig:
    gamma: float = 0.99
    lr_q: float = 1e-4
    lr_bc: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 1024
    epochs: int = 30
    hidden: int = 128
    dropout: float = 0.0
    target_update_tau: float = 0.005
    bcq_threshold: float = 0.3
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    val_frac: float = 0.1
    num_workers: int = 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_unit_hash(value: object) -> float:
    h = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    return int(h[:15], 16) / float(16**15 - 1)


def split_by_stay_id(df: pd.DataFrame, val_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "stay_id" not in df.columns:
        raise ValueError("Expected a 'stay_id' column for stay-level splitting.")
    unique_stays = pd.Series(df["stay_id"].dropna().unique())
    val_stays = set(unique_stays[unique_stays.map(stable_unit_hash) < val_frac].tolist())
    val_mask = df["stay_id"].isin(val_stays)
    return df.loc[~val_mask].reset_index(drop=True), df.loc[val_mask].reset_index(drop=True)


def infer_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    state_cols = sorted([c for c in df.columns if c.startswith("s_") and not c.startswith("s_next_")])
    next_state_cols = sorted([c for c in df.columns if c.startswith("s_next_")])
    return state_cols, next_state_cols


def compute_norm_stats(df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    x = df[cols].to_numpy(dtype=np.float32)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


class OfflineDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        state_cols: List[str],
        next_state_cols: List[str],
        state_mean: np.ndarray,
        state_std: np.ndarray,
        next_state_mean: np.ndarray,
        next_state_std: np.ndarray,
    ) -> None:
        s = df[state_cols].to_numpy(dtype=np.float32)
        sp = df[next_state_cols].to_numpy(dtype=np.float32)
        s = (s - state_mean) / state_std
        sp = (sp - next_state_mean) / next_state_std
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        sp = np.nan_to_num(sp, nan=0.0, posinf=0.0, neginf=0.0)

        self.s = torch.from_numpy(s)
        self.sp = torch.from_numpy(sp)
        self.a = torch.from_numpy(df["action"].to_numpy(dtype=np.int64))
        self.r = torch.from_numpy(df["reward"].to_numpy(dtype=np.float32).reshape(-1, 1))
        self.d = torch.from_numpy(df["done"].to_numpy(dtype=np.float32).reshape(-1, 1))

    def __len__(self) -> int:
        return len(self.a)

    def __getitem__(self, idx: int):
        return self.s[idx], self.a[idx], self.r[idx], self.sp[idx], self.d[idx]


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, dropout: float = 0.0) -> None:
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


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = MLP(state_dim, n_actions, hidden, dropout)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class BehaviorPolicy(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = MLP(state_dim, n_actions, hidden, dropout)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for p_t, p_s in zip(target.parameters(), source.parameters()):
        p_t.data.mul_(1.0 - tau).add_(p_s.data, alpha=tau)


@torch.no_grad()
def bcq_select_actions(q_values: torch.Tensor, bc_logits: torch.Tensor, threshold: float) -> torch.Tensor:
    probs = F.softmax(bc_logits, dim=-1)
    max_prob = probs.max(dim=1, keepdim=True).values
    mask = probs / (max_prob + 1e-8) > threshold
    masked_q = q_values.masked_fill(~mask, -1e9)
    return masked_q.argmax(dim=1)


def evaluate(q: QNetwork, q_t: QNetwork, bc: BehaviorPolicy, dl: DataLoader, cfg: BCQConfig, device: torch.device) -> Dict[str, float]:
    q.eval(); q_t.eval(); bc.eval()
    total_q = 0.0
    total_bc = 0.0
    total_acc = 0.0
    total_n = 0

    with torch.no_grad():
        for s, a, r, sp, d in dl:
            s = s.to(device); a = a.to(device); r = r.to(device); sp = sp.to(device); d = d.to(device)

            q_sa = q(s).gather(1, a.view(-1, 1))
            next_actions = bcq_select_actions(q_t(sp), bc(sp), cfg.bcq_threshold).view(-1, 1)
            target = r + cfg.gamma * (1.0 - d) * q_t(sp).gather(1, next_actions)
            q_loss = F.mse_loss(q_sa, target)

            logits = bc(s)
            bc_loss = F.cross_entropy(logits, a)
            acc = (logits.argmax(dim=1) == a).float().mean()

            bs = s.shape[0]
            total_q += float(q_loss.item()) * bs
            total_bc += float(bc_loss.item()) * bs
            total_acc += float(acc.item()) * bs
            total_n += bs

    return {
        "q_loss": total_q / max(total_n, 1),
        "bc_loss": total_bc / max(total_n, 1),
        "bc_acc": total_acc / max(total_n, 1),
    }


def train_bcq(df: pd.DataFrame, cfg: BCQConfig, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    for col in ["action", "reward", "done"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    state_cols, next_state_cols = infer_columns(df)
    if not state_cols or not next_state_cols:
        raise ValueError("Could not infer s_* and s_next_* columns.")

    a_min, a_max = int(df["action"].min()), int(df["action"].max())
    if a_min < 0:
        raise ValueError("Action values must start at 0.")
    n_actions = a_max + 1

    df_tr, df_va = split_by_stay_id(df, cfg.val_frac)
    s_mean, s_std = compute_norm_stats(df_tr, state_cols)
    sp_mean, sp_std = compute_norm_stats(df_tr, next_state_cols)

    ds_tr = OfflineDataset(df_tr, state_cols, next_state_cols, s_mean, s_std, sp_mean, sp_std)
    ds_va = OfflineDataset(df_va, state_cols, next_state_cols, s_mean, s_std, sp_mean, sp_std)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    state_dim = ds_tr.s.shape[1]
    q = QNetwork(state_dim, n_actions, cfg.hidden, cfg.dropout).to(device)
    q_t = QNetwork(state_dim, n_actions, cfg.hidden, cfg.dropout).to(device)
    q_t.load_state_dict(q.state_dict())
    bc = BehaviorPolicy(state_dim, n_actions, cfg.hidden, cfg.dropout).to(device)

    opt_q = torch.optim.Adam(q.parameters(), lr=cfg.lr_q, weight_decay=cfg.weight_decay)
    opt_bc = torch.optim.Adam(bc.parameters(), lr=cfg.lr_bc, weight_decay=cfg.weight_decay)

    schema = {
        "state_cols": state_cols,
        "next_state_cols": next_state_cols,
        "action_col": "action",
        "n_actions": n_actions,
        "state_mean": s_mean.tolist(),
        "state_std": s_std.tolist(),
        "next_state_mean": sp_mean.tolist(),
        "next_state_std": sp_std.tolist(),
    }
    with open(os.path.join(save_dir, "schema_and_norm.json"), "w") as f:
        json.dump(schema, f, indent=2)

    metrics_path = os.path.join(save_dir, "metrics.csv")
    best_val = math.inf
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_q_loss", "train_bc_loss", "train_bc_acc", "val_q_loss", "val_bc_loss", "val_bc_acc"])
        writer.writeheader()

        for epoch in range(1, cfg.epochs + 1):
            q.train(); bc.train()
            run_q = run_bc = run_acc = 0.0
            count = 0

            for s, a, r, sp, d in dl_tr:
                s = s.to(device); a = a.to(device); r = r.to(device); sp = sp.to(device); d = d.to(device)

                logits = bc(s)
                bc_loss = F.cross_entropy(logits, a)
                opt_bc.zero_grad(set_to_none=True)
                bc_loss.backward()
                opt_bc.step()

                with torch.no_grad():
                    next_actions = bcq_select_actions(q_t(sp), bc(sp), cfg.bcq_threshold).view(-1, 1)
                    target = r + cfg.gamma * (1.0 - d) * q_t(sp).gather(1, next_actions)

                q_sa = q(s).gather(1, a.view(-1, 1))
                q_loss = F.mse_loss(q_sa, target)
                opt_q.zero_grad(set_to_none=True)
                q_loss.backward()
                opt_q.step()
                soft_update(q_t, q, cfg.target_update_tau)

                bs = s.shape[0]
                run_q += float(q_loss.item()) * bs
                run_bc += float(bc_loss.item()) * bs
                run_acc += float((logits.argmax(dim=1) == a).float().mean().item()) * bs
                count += bs

            train = {
                "q_loss": run_q / max(count, 1),
                "bc_loss": run_bc / max(count, 1),
                "bc_acc": run_acc / max(count, 1),
            }
            val = evaluate(q, q_t, bc, dl_va, cfg, device)
            row = {
                "epoch": epoch,
                "train_q_loss": train["q_loss"],
                "train_bc_loss": train["bc_loss"],
                "train_bc_acc": train["bc_acc"],
                "val_q_loss": val["q_loss"],
                "val_bc_loss": val["bc_loss"],
                "val_bc_acc": val["bc_acc"],
            }
            writer.writerow(row)
            f.flush()

            print(
                f"[Epoch {epoch:03d}/{cfg.epochs}] "
                f"train q={train['q_loss']:.4f} bc={train['bc_loss']:.4f} acc={train['bc_acc']:.4f} | "
                f"val q={val['q_loss']:.4f} bc={val['bc_loss']:.4f} acc={val['bc_acc']:.4f}"
            )

            val_score = val["q_loss"] + val["bc_loss"]
            if val_score < best_val:
                best_val = val_score
                torch.save({
                    "q": q.state_dict(),
                    "q_target": q_t.state_dict(),
                    "bc": bc.state_dict(),
                    "config": asdict(cfg),
                    "n_actions": n_actions,
                }, os.path.join(save_dir, "best.pt"))

    torch.save({
        "q": q.state_dict(),
        "q_target": q_t.state_dict(),
        "bc": bc.state_dict(),
        "config": asdict(cfg),
        "n_actions": n_actions,
    }, os.path.join(save_dir, "final.pt"))
    print(f"Saved checkpoints and metrics to {save_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="Processed/transitions.parquet")
    p.add_argument("--save_dir", type=str, default="Output")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr_q", type=float, default=1e-4)
    p.add_argument("--lr_bc", type=float, default=1e-4)
    p.add_argument("--bcq_threshold", type=float, default=0.3)
    p.add_argument("--target_update_tau", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BCQConfig(
        gamma=args.gamma,
        lr_q=args.lr_q,
        lr_bc=args.lr_bc,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden=args.hidden,
        bcq_threshold=args.bcq_threshold,
        target_update_tau=args.target_update_tau,
        seed=args.seed,
        device=args.device,
        val_frac=args.val_frac,
        num_workers=args.num_workers,
    )
    print(f"Loading parquet: {args.data}")
    df = pd.read_parquet(args.data)
    train_bcq(df, cfg, args.save_dir)


if __name__ == "__main__":
    main()
