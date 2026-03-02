#!/usr/bin/env python3
"""
IQL training for MIMIC-style offline RL dataset

This trainer expects a Parquet dataset with columns like:

State features:
  s_<feature>...

Next-state features:
  s_next_<feature>...

Action vector columns (examples):
  proc_*   -> binary (0/1)    (Bernoulli policy heads)
  drug_*   -> binary (0/1)    (Bernoulli policy heads)
  cont_*   -> integer code in {-2,-1,0,1,2}  (Categorical policy head with 5 classes)

And:
  reward (float)
  done   (0/1)

Algorithm: Implicit Q-Learning (IQL)
- Q(s,a): two critics (Double Q)
- V(s): expectile regression to Q(s,a)
- pi(a|s): advantage-weighted behavior cloning with hybrid action distribution

Run:
  python iql_train.py --data sepsis_iql_actionvec_transitions.parquet --epochs 30

"""

from __future__ import annotations

import argparse
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
    expectile_tau: float = 0.7      # 0.5 = MSE; >0.5 biases towards higher Q
    awr_beta: float = 3.0           # advantage temperature for exp(beta * A)
    awr_clip: float = 100.0         # clip weights to avoid blow-ups

    lr_q: float = 3e-4
    lr_v: float = 3e-4
    lr_pi: float = 3e-4
    weight_decay: float = 0.0

    batch_size: int = 1024
    epochs: int = 30
    num_workers: int = 2

    hidden: int = 256
    dropout: float = 0.0

    # Target/EMA for critics (helps stability)
    use_target_q: bool = True
    tau_target: float = 0.005  # soft update rate for Q target networks

    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Train/val split
    val_frac: float = 0.1


# ------------------------- Utilities -------------------------

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Expectile regression loss:
      L = |tau - I(diff < 0)| * diff^2
    where diff = (Q - V).
    """
    weight = torch.where(diff < 0, 1.0 - tau, tau)
    return (weight * diff.pow(2)).mean()


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p_t, p_s in zip(target.parameters(), source.parameters()):
            p_t.data.mul_(1.0 - tau).add_(p_s.data, alpha=tau)


# ------------------------- Dataset -------------------------

class OfflineRLDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        state_cols: List[str],
        next_state_cols: List[str],
        action_cols: List[str],
        cont_action_cols: List[str],
        bin_action_cols: List[str],
        state_mean: np.ndarray,
        state_std: np.ndarray,
        next_state_mean: np.ndarray,
        next_state_std: np.ndarray,
    ):
        self.state_cols = state_cols
        self.next_state_cols = next_state_cols
        self.action_cols = action_cols
        self.cont_action_cols = cont_action_cols
        self.bin_action_cols = bin_action_cols

        # Normalize states (z-score). Keep NaNs under control (fill with 0 after normalization).
        s = df[state_cols].to_numpy(dtype=np.float32)
        sp = df[next_state_cols].to_numpy(dtype=np.float32)

        s = (s - state_mean) / state_std
        sp = (sp - next_state_mean) / next_state_std

        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        sp = np.nan_to_num(sp, nan=0.0, posinf=0.0, neginf=0.0)

        # Actions:
        # - binary actions: keep as {0,1} float
        # - cont actions: values in {-2,-1,0,1,2} -> indices {0..4} for categorical heads
        a_bin = df[bin_action_cols].to_numpy(dtype=np.float32) if bin_action_cols else np.zeros((len(df), 0), np.float32)

        a_cont_raw = df[cont_action_cols].to_numpy(dtype=np.int64) if cont_action_cols else np.zeros((len(df), 0), np.int64)
        # Map -2,-1,0,1,2 -> 0..4
        a_cont = (a_cont_raw + 2).clip(0, 4)

        # For critic input we also build a single float action vector (bin + cont raw scaled)
        # Critics can take continuous floats; we feed cont codes as floats in [-2..2].
        a_cont_float = df[cont_action_cols].to_numpy(dtype=np.float32) if cont_action_cols else np.zeros((len(df), 0), np.float32)
        a_float = np.concatenate([a_bin, a_cont_float], axis=1).astype(np.float32)

        r = df["reward"].to_numpy(dtype=np.float32).reshape(-1, 1)
        d = df["done"].to_numpy(dtype=np.float32).reshape(-1, 1)

        self.s = torch.from_numpy(s)
        self.sp = torch.from_numpy(sp)
        self.a_float = torch.from_numpy(a_float)
        self.a_bin = torch.from_numpy(a_bin)
        self.a_cont = torch.from_numpy(a_cont)
        self.r = torch.from_numpy(r)
        self.d = torch.from_numpy(d)

    def __len__(self) -> int:
        return self.s.shape[0]

    def __getitem__(self, idx: int):
        return self.s[idx], self.a_float[idx], self.a_bin[idx], self.a_cont[idx], self.r[idx], self.sp[idx], self.d[idx]


def infer_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    # States
    state_cols = [c for c in df.columns if c.startswith("s_") and not c.startswith("s_next_")]
    next_state_cols = [c for c in df.columns if c.startswith("s_next_")]

    # Actions: everything that is not state/next_state/reward/done
    reserved = set(state_cols + next_state_cols + ["reward", "done"])
    action_cols = [c for c in df.columns if c not in reserved]

    # Heuristic: binary action columns (procedures/drugs) and cont action columns
    cont_action_cols = [c for c in action_cols if c.startswith("cont_")]
    bin_action_cols = [c for c in action_cols if c.startswith("proc_") or c.startswith("drug_")]

    # If user named columns differently, fall back:
    if not bin_action_cols and not cont_action_cols:
        # Treat all actions as continuous floats
        cont_action_cols = action_cols
        bin_action_cols = []

    # Ensure consistent ordering
    state_cols.sort()
    next_state_cols.sort()
    action_cols.sort()
    cont_action_cols.sort()
    bin_action_cols.sort()

    return state_cols, next_state_cols, action_cols, cont_action_cols, bin_action_cols


def compute_norm_stats(df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    x = df[cols].to_numpy(dtype=np.float32)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


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


class CriticQ(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.mlp = MLP(state_dim + action_dim, 1, hidden=hidden, dropout=dropout)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=-1)
        return self.mlp(x)


class ValueV(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.mlp = MLP(state_dim, 1, hidden=hidden, dropout=dropout)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.mlp(s)


class HybridPolicy(nn.Module):
    """
    Policy that factorizes into independent heads:
      - Bernoulli logits for binary dims (proc_*, drug_*)
      - Categorical logits (5 classes) per continuous-med dim, representing {-2,-1,0,1,2}
    """
    def __init__(self, state_dim: int, n_bin: int, n_cont: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.n_bin = n_bin
        self.n_cont = n_cont

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.bin_head = nn.Linear(hidden, n_bin) if n_bin > 0 else None
        self.cont_head = nn.Linear(hidden, n_cont * 5) if n_cont > 0 else None  # 5-class per cont dim

    def forward(self, s: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(s)
        out: Dict[str, torch.Tensor] = {}

        if self.bin_head is not None:
            out["bin_logits"] = self.bin_head(h)  # [B, n_bin]

        if self.cont_head is not None:
            logits = self.cont_head(h)            # [B, n_cont*5]
            out["cont_logits"] = logits.view(-1, self.n_cont, 5)  # [B, n_cont, 5]

        return out

    def log_prob(self, s: torch.Tensor, a_bin: torch.Tensor, a_cont_idx: torch.Tensor) -> torch.Tensor:
        out = self.forward(s)

        # Tensor, shape [B]
        logp = torch.zeros(s.shape[0], device=s.device, dtype=s.dtype)

        if "bin_logits" in out:
            dist_bin = torch.distributions.Bernoulli(logits=out["bin_logits"])
            logp = logp + dist_bin.log_prob(a_bin).sum(dim=-1)

        if "cont_logits" in out:
            dist_cont = torch.distributions.Categorical(logits=out["cont_logits"])
            logp = logp + dist_cont.log_prob(a_cont_idx).sum(dim=-1)

        return logp

    @torch.no_grad()
    def act_mode(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Deterministic mode action:
          - Bernoulli: threshold sigmoid(logit) >= 0.5
          - Categorical: argmax over 5 classes, mapped back to {-2..2}
        Returns:
          a_bin_mode: [B, n_bin] float {0,1}
          a_cont_code: [B, n_cont] float in {-2,-1,0,1,2}
        """
        out = self.forward(s)
        if self.n_bin > 0:
            p = torch.sigmoid(out["bin_logits"])
            a_bin = (p >= 0.5).float()
        else:
            a_bin = torch.zeros((s.shape[0], 0), device=s.device)

        if self.n_cont > 0:
            idx = out["cont_logits"].argmax(dim=-1)        # [B, n_cont] in 0..4
            a_cont = idx.float() - 2.0                     # map back to -2..2
        else:
            a_cont = torch.zeros((s.shape[0], 0), device=s.device)

        return a_bin, a_cont


# ------------------------- Training -------------------------

def train_iql(
    df: pd.DataFrame,
    cfg: IQLConfig,
    save_dir: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(cfg.device)
    set_seed(cfg.seed)

    # Column inference
    state_cols, next_state_cols, action_cols, cont_action_cols, bin_action_cols = infer_columns(df)

    if len(state_cols) == 0 or len(next_state_cols) == 0:
        raise ValueError("Could not infer state/next_state columns. Expect columns starting with 's_' and 's_next_'.")

    # Split
    n = len(df)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_val = int(cfg.val_frac * n)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_va = df.iloc[val_idx].reset_index(drop=True)

    # Normalization stats from train only
    s_mean, s_std = compute_norm_stats(df_tr, state_cols)
    sp_mean, sp_std = compute_norm_stats(df_tr, next_state_cols)

    # Dataset / Loader
    ds_tr = OfflineRLDataset(
        df_tr, state_cols, next_state_cols, action_cols, cont_action_cols, bin_action_cols,
        s_mean, s_std, sp_mean, sp_std
    )
    ds_va = OfflineRLDataset(
        df_va, state_cols, next_state_cols, action_cols, cont_action_cols, bin_action_cols,
        s_mean, s_std, sp_mean, sp_std
    )

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    state_dim = ds_tr.s.shape[1]
    action_dim_float = ds_tr.a_float.shape[1]
    n_bin = ds_tr.a_bin.shape[1]
    n_cont = ds_tr.a_cont.shape[1]

    # Models
    q1 = CriticQ(state_dim, action_dim_float, hidden=cfg.hidden, dropout=cfg.dropout).to(device)
    q2 = CriticQ(state_dim, action_dim_float, hidden=cfg.hidden, dropout=cfg.dropout).to(device)
    v  = ValueV(state_dim, hidden=cfg.hidden, dropout=cfg.dropout).to(device)
    pi = HybridPolicy(state_dim, n_bin=n_bin, n_cont=n_cont, hidden=cfg.hidden, dropout=cfg.dropout).to(device)

    # Optional target critics
    if cfg.use_target_q:
        q1_t = CriticQ(state_dim, action_dim_float, hidden=cfg.hidden, dropout=cfg.dropout).to(device)
        q2_t = CriticQ(state_dim, action_dim_float, hidden=cfg.hidden, dropout=cfg.dropout).to(device)
        q1_t.load_state_dict(q1.state_dict())
        q2_t.load_state_dict(q2.state_dict())
        q1_t.eval()
        q2_t.eval()
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
            for s, a_float, a_bin, a_cont, r, sp, d in dl_va:
                s = s.to(device); a_float = a_float.to(device)
                a_bin = a_bin.to(device); a_cont = a_cont.to(device)
                r = r.to(device); sp = sp.to(device); d = d.to(device)

                # Targets
                v_sp = v(sp)
                y = r + cfg.gamma * (1.0 - d) * v_sp

                q1_sa = q1(s, a_float)
                q2_sa = q2(s, a_float)
                loss_q = 0.5 * (mse(q1_sa, y) + mse(q2_sa, y))

                # V expectile
                q_min = torch.min(q1_sa, q2_sa)
                diff = q_min - v(s)
                loss_v = expectile_loss(diff, cfg.expectile_tau)

                # Policy AWR-BC
                adv = (q_min - v(s)).detach().squeeze(-1)
                w = torch.exp(cfg.awr_beta * adv).clamp(max=cfg.awr_clip)
                logp = pi.log_prob(s, a_bin, a_cont)
                loss_pi = -(w * logp).mean()

                bs = s.shape[0]
                losses["q"] += float(loss_q.item()) * bs
                losses["v"] += float(loss_v.item()) * bs
                losses["pi"] += float(loss_pi.item()) * bs
                count += bs

        for k in losses:
            losses[k] /= max(count, 1)
        return losses

    # Save schema + norm stats for later inference
    schema = {
        "state_cols": state_cols,
        "next_state_cols": next_state_cols,
        "action_cols": action_cols,
        "bin_action_cols": bin_action_cols,
        "cont_action_cols": cont_action_cols,
        "state_mean": s_mean.tolist(),
        "state_std": s_std.tolist(),
        "next_state_mean": sp_mean.tolist(),
        "next_state_std": sp_std.tolist(),
        "cont_code_mapping": {"-2": 0, "-1": 1, "0": 2, "1": 3, "2": 4},
    }
    with open(os.path.join(save_dir, "schema_and_norm.json"), "w") as f:
        import json
        json.dump(schema, f, indent=2)

    # Training loop
    best_val = math.inf
    for epoch in range(1, cfg.epochs + 1):
        q1.train(); q2.train(); v.train(); pi.train()

        running = {"q": 0.0, "v": 0.0, "pi": 0.0}
        count = 0

        for s, a_float, a_bin, a_cont, r, sp, d in dl_tr:
            s = s.to(device); a_float = a_float.to(device)
            a_bin = a_bin.to(device); a_cont = a_cont.to(device)
            r = r.to(device); sp = sp.to(device); d = d.to(device)

            # ---------------- Q update ----------------
            with torch.no_grad():
                v_sp = v(sp)
                y = r + cfg.gamma * (1.0 - d) * v_sp

            q1_sa = q1(s, a_float)
            q2_sa = q2(s, a_float)
            loss_q = 0.5 * (mse(q1_sa, y) + mse(q2_sa, y))

            opt_q.zero_grad(set_to_none=True)
            loss_q.backward()
            opt_q.step()

            # Target critics soft update
            if cfg.use_target_q:
                soft_update(q1_t, q1, cfg.tau_target)
                soft_update(q2_t, q2, cfg.tau_target)

            # ---------------- V update (expectile) ----------------
            with torch.no_grad():
                q1_sa_t = q1_t(s, a_float)
                q2_sa_t = q2_t(s, a_float)
                q_min_t = torch.min(q1_sa_t, q2_sa_t)

            v_s = v(s)
            diff = q_min_t - v_s
            loss_v = expectile_loss(diff, cfg.expectile_tau)

            opt_v.zero_grad(set_to_none=True)
            loss_v.backward()
            opt_v.step()

            # ---------------- Policy update (AWR-BC) ----------------
            with torch.no_grad():
                # Use target critics for advantage
                q1_sa_t = q1_t(s, a_float)
                q2_sa_t = q2_t(s, a_float)
                q_min_t = torch.min(q1_sa_t, q2_sa_t)
                adv = (q_min_t - v(s)).detach().squeeze(-1)
                w = torch.exp(cfg.awr_beta * adv).clamp(max=cfg.awr_clip)

            logp = pi.log_prob(s, a_bin, a_cont)  # [B]
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

        # Save best
        if val_score < best_val:
            best_val = val_score
            ckpt = {
                "q1": q1.state_dict(),
                "q2": q2.state_dict(),
                "v": v.state_dict(),
                "pi": pi.state_dict(),
                "config": cfg.__dict__,
            }
            torch.save(ckpt, os.path.join(save_dir, "best.pt"))

    # Save final
    ckpt = {
        "q1": q1.state_dict(),
        "q2": q2.state_dict(),
        "v": v.state_dict(),
        "pi": pi.state_dict(),
        "config": cfg.__dict__,
    }
    torch.save(ckpt, os.path.join(save_dir, "final.pt"))
    print(f"Saved checkpoints to {save_dir}")


# ------------------------- CLI -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to parquet transitions dataset")
    p.add_argument("--save_dir", type=str, default="iql_runs/run1", help="Output directory for checkpoints")

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

    # Basic checks
    for col in ["reward", "done"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    train_iql(df=df, cfg=cfg, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
