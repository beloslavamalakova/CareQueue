#!/usr/bin/env python3
"""
IQL training for *lite* discrete-action dataset.

Expected parquet columns:
  s_*         : state features
  s_next_*    : next state features
  action      : integer in [0, n_actions-1]
  reward      : float
  done        : 0/1

Algorithm: Implicit Q-Learning (IQL) with:
  - Double Q critics: Q(s,a) with action embedding
  - Value network V(s) via expectile regression to Q
  - Policy pi(a|s) via advantage-weighted behavior cloning (AWR-BC)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ------------------------- Config -------------------------

@dataclass
class IQLConfig:
    gamma: float = 0.99
    expectile_tau: float = 0.6
    awr_beta: float = 2.0
    awr_clip: float = 100.0

    lr_q: float = 5e-5
    lr_v: float = 5e-5
    lr_pi: float = 5e-5
    weight_decay: float = 0.0

    batch_size: int = 1024
    epochs: int = 30
    num_workers: int = 2

    hidden: int = 256
    dropout: float = 0.0

    # Target critics (EMA)
    use_target_q: bool = True
    tau_target: float = 0.001

    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    val_frac: float = 0.1


# ------------------------- Utilities -------------------------

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    weight = torch.where(diff < 0, 1.0 - tau, tau)
    return (weight * diff.pow(2)).mean()


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p_t, p_s in zip(target.parameters(), source.parameters()):
            p_t.data.mul_(1.0 - tau).add_(p_s.data, alpha=tau)


def compute_norm_stats(df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    x = df[cols].to_numpy(dtype=np.float32)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def infer_columns_discrete(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    state_cols = [c for c in df.columns if c.startswith("s_") and not c.startswith("s_next_")]
    next_state_cols = [c for c in df.columns if c.startswith("s_next_")]
    state_cols.sort()
    next_state_cols.sort()
    return state_cols, next_state_cols


# ------------------------- Dataset -------------------------

class OfflineDiscreteDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        state_cols: List[str],
        next_state_cols: List[str],
        state_mean: np.ndarray,
        state_std: np.ndarray,
        next_state_mean: np.ndarray,
        next_state_std: np.ndarray,
    ):
        s = df[state_cols].to_numpy(dtype=np.float32)
        sp = df[next_state_cols].to_numpy(dtype=np.float32)

        s = (s - state_mean) / state_std
        sp = (sp - next_state_mean) / next_state_std

        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        sp = np.nan_to_num(sp, nan=0.0, posinf=0.0, neginf=0.0)

        a = df["action"].to_numpy(dtype=np.int64).reshape(-1, 1)
        r = df["reward"].to_numpy(dtype=np.float32).reshape(-1, 1)
        d = df["done"].to_numpy(dtype=np.float32).reshape(-1, 1)

        self.s = torch.from_numpy(s)
        self.sp = torch.from_numpy(sp)
        self.a = torch.from_numpy(a)   # [N,1] int64
        self.r = torch.from_numpy(r)
        self.d = torch.from_numpy(d)

    def __len__(self) -> int:
        return self.s.shape[0]

    def __getitem__(self, idx: int):
        return self.s[idx], self.a[idx], self.r[idx], self.sp[idx], self.d[idx]


# ------------------------- Networks -------------------------

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CriticQDiscrete(nn.Module):
    """
    Q(s,a) with an embedding for discrete action index.
    """
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256, dropout: float = 0.0, emb_dim: int = 32):
        super().__init__()
        self.a_emb = nn.Embedding(n_actions, emb_dim)
        self.mlp = MLP(state_dim + emb_dim, 1, hidden=hidden, dropout=dropout)

    def forward(self, s: torch.Tensor, a_idx: torch.Tensor) -> torch.Tensor:
        # a_idx: [B,1] or [B]
        a_idx = a_idx.view(-1).long()
        ae = self.a_emb(a_idx)
        x = torch.cat([s, ae], dim=-1)
        return self.mlp(x)


class ValueV(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.mlp = MLP(state_dim, 1, hidden=hidden, dropout=dropout)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.mlp(s)


class DiscretePolicy(nn.Module):
    """
    Categorical pi(a|s) over n_actions.
    """
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.n_actions = n_actions
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
        return self.head(h)  # logits [B, n_actions]

    def log_prob(self, s: torch.Tensor, a_idx: torch.Tensor) -> torch.Tensor:
        logits = self.forward(s)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(a_idx.view(-1).long())

    @torch.no_grad()
    def act_mode(self, s: torch.Tensor) -> torch.Tensor:
        logits = self.forward(s)
        return torch.argmax(logits, dim=-1)


# ------------------------- Training -------------------------

def train_iql_discrete(df: pd.DataFrame, cfg: IQLConfig, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(cfg.device)
    set_seed(cfg.seed)

    # Basic checks
    for col in ["reward", "done", "action"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    state_cols, next_state_cols = infer_columns_discrete(df)
    if not state_cols or not next_state_cols:
        raise ValueError("Could not infer state/next_state columns. Need 's_*' and 's_next_*'.")

    # Infer number of actions
    a_min = int(df["action"].min())
    a_max = int(df["action"].max())
    if a_min < 0:
        raise ValueError(f"Action has negative values (min={a_min}). Expect 0..K-1.")
    n_actions = a_max + 1

    # Split
    n = len(df)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_val = int(cfg.val_frac * n)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_va = df.iloc[val_idx].reset_index(drop=True)

    # Normalization from train only
    s_mean, s_std = compute_norm_stats(df_tr, state_cols)
    sp_mean, sp_std = compute_norm_stats(df_tr, next_state_cols)

    ds_tr = OfflineDiscreteDataset(df_tr, state_cols, next_state_cols, s_mean, s_std, sp_mean, sp_std)
    ds_va = OfflineDiscreteDataset(df_va, state_cols, next_state_cols, s_mean, s_std, sp_mean, sp_std)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    state_dim = ds_tr.s.shape[1]

    q1 = CriticQDiscrete(state_dim, n_actions, hidden=cfg.hidden, dropout=cfg.dropout).to(device)
    q2 = CriticQDiscrete(state_dim, n_actions, hidden=cfg.hidden, dropout=cfg.dropout).to(device)
    v  = ValueV(state_dim, hidden=cfg.hidden, dropout=cfg.dropout).to(device)
    pi = DiscretePolicy(state_dim, n_actions, hidden=cfg.hidden, dropout=cfg.dropout).to(device)

    # Target critics
    if cfg.use_target_q:
        q1_t = CriticQDiscrete(state_dim, n_actions, hidden=cfg.hidden, dropout=cfg.dropout).to(device)
        q2_t = CriticQDiscrete(state_dim, n_actions, hidden=cfg.hidden, dropout=cfg.dropout).to(device)
        q1_t.load_state_dict(q1.state_dict()); q2_t.load_state_dict(q2.state_dict())
        q1_t.eval(); q2_t.eval()
    else:
        q1_t = q1
        q2_t = q2

    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=cfg.lr_q, weight_decay=cfg.weight_decay)
    opt_v = torch.optim.Adam(v.parameters(), lr=cfg.lr_v, weight_decay=cfg.weight_decay)
    opt_pi = torch.optim.Adam(pi.parameters(), lr=cfg.lr_pi, weight_decay=cfg.weight_decay)

    mse = nn.MSELoss()

    def eval_epoch() -> Dict[str, float]:
        q1.eval(); q2.eval(); v.eval(); pi.eval()
        losses = {"q": 0.0, "v": 0.0, "pi": 0.0}
        count = 0

        with torch.no_grad():
            for s, a, r, sp, d in dl_va:
                s = s.to(device); a = a.to(device)
                r = r.to(device); sp = sp.to(device); d = d.to(device)

                v_sp = v(sp)
                y = r + cfg.gamma * (1.0 - d) * v_sp

                q1_sa = q1(s, a)
                q2_sa = q2(s, a)
                loss_q = 0.5 * (mse(q1_sa, y) + mse(q2_sa, y))

                q_min = torch.min(q1_sa, q2_sa)
                diff = q_min - v(s)
                loss_v = expectile_loss(diff, cfg.expectile_tau)

                adv = (q_min - v(s)).detach().squeeze(-1)
                w = torch.exp(cfg.awr_beta * adv).clamp(max=cfg.awr_clip)
                logp = pi.log_prob(s, a)
                loss_pi = -(w * logp).mean()

                bs = s.shape[0]
                losses["q"] += float(loss_q.item()) * bs
                losses["v"] += float(loss_v.item()) * bs
                losses["pi"] += float(loss_pi.item()) * bs
                count += bs

        for k in losses:
            losses[k] /= max(count, 1)
        return losses

    # Save schema + norm stats
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

    best_val = math.inf
    for epoch in range(1, cfg.epochs + 1):
        q1.train(); q2.train(); v.train(); pi.train()
        running = {"q": 0.0, "v": 0.0, "pi": 0.0}
        count = 0

        for s, a, r, sp, d in dl_tr:
            s = s.to(device); a = a.to(device)
            r = r.to(device); sp = sp.to(device); d = d.to(device)

            # ----- Q update -----
            with torch.no_grad():
                y = r + cfg.gamma * (1.0 - d) * v(sp)

            q1_sa = q1(s, a)
            q2_sa = q2(s, a)
            loss_q = 0.5 * (mse(q1_sa, y) + mse(q2_sa, y))

            opt_q.zero_grad(set_to_none=True)
            loss_q.backward()
            opt_q.step()

            if cfg.use_target_q:
                soft_update(q1_t, q1, cfg.tau_target)
                soft_update(q2_t, q2, cfg.tau_target)

            # ----- V update -----
            with torch.no_grad():
                q_min_t = torch.min(q1_t(s, a), q2_t(s, a))

            v_s = v(s)
            loss_v = expectile_loss(q_min_t - v_s, cfg.expectile_tau)

            opt_v.zero_grad(set_to_none=True)
            loss_v.backward()
            opt_v.step()

            # ----- Policy update (AWR-BC) -----
            with torch.no_grad():
                q_min_t2 = torch.min(q1_t(s, a), q2_t(s, a))
                adv = (q_min_t2 - v(s)).detach().squeeze(-1)
                w = torch.exp(cfg.awr_beta * adv).clamp(max=cfg.awr_clip)

            logp = pi.log_prob(s, a)
            loss_pi = -(w * logp).mean()

            opt_pi.zero_grad(set_to_none=True)
            loss_pi.backward()
            opt_pi.step()

            bs = s.shape[0]
            running["q"] += float(loss_q.item()) * bs
            running["v"] += float(loss_v.item()) * bs
            running["pi"] += float(loss_pi.item()) * bs
            count += bs

        for k in running:
            running[k] /= max(count, 1)

        val_losses = eval_epoch()
        val_score = val_losses["q"] + val_losses["v"] + val_losses["pi"]

        print(
            f"[Epoch {epoch:03d}/{cfg.epochs}] "
            f"train q={running['q']:.4f} v={running['v']:.4f} pi={running['pi']:.4f} | "
            f"val q={val_losses['q']:.4f} v={val_losses['v']:.4f} pi={val_losses['pi']:.4f}"
        )

        if val_score < best_val:
            best_val = val_score
            ckpt = {
                "q1": q1.state_dict(),
                "q2": q2.state_dict(),
                "v": v.state_dict(),
                "pi": pi.state_dict(),
                "config": cfg.__dict__,
                "n_actions": n_actions,
            }
            torch.save(ckpt, os.path.join(save_dir, "best.pt"))

    ckpt = {
        "q1": q1.state_dict(),
        "q2": q2.state_dict(),
        "v": v.state_dict(),
        "pi": pi.state_dict(),
        "config": cfg.__dict__,
        "n_actions": n_actions,
    }
    torch.save(ckpt, os.path.join(save_dir, "final.pt"))
    print(f"Saved checkpoints to {save_dir}")


# ------------------------- CLI -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=r"C:\Users\20231942\Desktop\Central Folder\TUe\Year 3\Honors\Code\CareQueue\mimic-iv-3.1\iql\Version 2\Processed\sepsis_lite_transitions_4h.parquet", help="Path to parquet transitions dataset")
    p.add_argument("--save_dir", type=str, default="Output", help="Output directory for checkpoints")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=256)

    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--expectile_tau", type=float, default=0.7)
    p.add_argument("--awr_beta", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--val_frac", type=float, default=0.1)

    p.add_argument("--use_target_q", action="store_true", help="Use EMA target Q networks (recommended)")
    p.add_argument("--tau_target", type=float, default=0.005)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = IQLConfig(
        gamma=args.gamma,
        expectile_tau=args.expectile_tau,
        awr_beta=args.awr_beta,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden=args.hidden,
        seed=args.seed,
        device=args.device,
        val_frac=args.val_frac,
        use_target_q=args.use_target_q,
        tau_target=args.tau_target,
    )

    print(f"Loading parquet: {args.data}")
    df = pd.read_parquet(args.data)
    train_iql_discrete(df=df, cfg=cfg, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
