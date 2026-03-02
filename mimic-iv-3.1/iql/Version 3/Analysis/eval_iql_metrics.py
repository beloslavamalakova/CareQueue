#!/usr/bin/env python3
"""
Evaluate a trained IQL discrete policy with:
- FQE
- KL(pi || clinician)
- % Unsupported actions (based on clinician prob threshold)
- CWPDIS
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# Importing model definitions from training ----
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iql_training import DiscretePolicy, CriticQDiscrete

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-12)


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std_safe = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std_safe


def ensure_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    

class BehaviorMLP(nn.Module):
    """Small classifier to estimate clinician hehavior from any state"""
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)  # logits


def fit_behavior_model(
    s_train: np.ndarray,
    a_train: np.ndarray,
    n_actions: int,
    device: torch.device,
    seed: int,
    hidden: int = 256,
    epochs: int = 10,
    batch_size: int = 2048,
    lr: float = 1e-3,
) -> BehaviorMLP:
    set_seed(seed)
    model = BehaviorMLP(s_train.shape[1], n_actions, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(s_train).float(),
        torch.from_numpy(a_train).long(),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for _ in range(epochs):
        for sb, ab in dl:
            sb = sb.to(device)
            ab = ab.to(device)
            logits = model(sb)
            loss = loss_fn(logits, ab)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return model


@torch.no_grad()
def behavior_probs(model: nn.Module, s: torch.Tensor) -> torch.Tensor:
    logits = model(s)
    p = torch.softmax(logits, dim=-1)

    # prevent exact zeros (important for KL + CWPDIS stability)
    p = torch.clamp(p, min=1e-6)

    # renormalize so rows still sum to 1
    p = p / p.sum(dim=-1, keepdim=True)

    return p


# FQE Functions

@dataclass
class FQEConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 2048
    steps: int = 100
    target_tau: float = 0.005
    seed: int = 0


class TransitionDataset(Dataset):
    def __init__(self, s, a, r, sp, d):
        self.s = torch.from_numpy(s).float()
        self.a = torch.from_numpy(a).long()
        self.r = torch.from_numpy(r).float()
        self.sp = torch.from_numpy(sp).float()
        self.d = torch.from_numpy(d).float()

    def __len__(self):
        return self.s.shape[0]

    def __getitem__(self, idx):
        return self.s[idx], self.a[idx], self.r[idx], self.sp[idx], self.d[idx]


@torch.no_grad()
def pi_action_probs(pi: DiscretePolicy, s: torch.Tensor) -> torch.Tensor:
    logits = pi(s)
    p = torch.softmax(logits, dim=-1)
    p = torch.clamp(p, min=1e-6)
    p = p / p.sum(dim=-1, keepdim=True)
    return p

def train_fqe(
    s: np.ndarray,
    a: np.ndarray,
    r: np.ndarray,
    sp: np.ndarray,
    d: np.ndarray,
    pi: DiscretePolicy,
    n_actions: int,
    hidden: int,
    device: torch.device,
    cfg: FQEConfig,
) -> CriticQDiscrete:
    set_seed(cfg.seed)

    q = CriticQDiscrete(state_dim=s.shape[1], n_actions=n_actions, hidden=hidden).to(device)
    q_tgt = CriticQDiscrete(state_dim=s.shape[1], n_actions=n_actions, hidden=hidden).to(device)
    q_tgt.load_state_dict(q.state_dict())

    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)

    ds = TransitionDataset(s, a, r, sp, d)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    pi.eval()
    q.train()
    q_tgt.eval()

    step = 0
    while step < cfg.steps:
        for sb, ab, rb, spb, db in dl:
            sb = sb.to(device)
            ab = ab.to(device)
            rb = rb.to(device)
            spb = spb.to(device)
            db = db.to(device)

            with torch.no_grad():
                # target = r + gamma*(1-d)*E_{a'~pi}[Q_tgt(s',a')]
                probs = pi_action_probs(pi, spb)  # [B, A]
                # compute Q_tgt(s', a') for all actions:
                # easiest: loop over actions in chunks
                qsp_all = []
                for aidx in range(n_actions):
                    a_tensor = torch.full((spb.shape[0],), aidx, device=device, dtype=torch.long)
                    qsp_all.append(q_tgt(spb, a_tensor).squeeze(-1))
                qsp_all = torch.stack(qsp_all, dim=-1)  # [B, A]
                vsp = torch.sum(probs * qsp_all, dim=-1)  # [B]
                y = rb + cfg.gamma * (1.0 - db) * vsp

            q_sa = q(sb, ab).squeeze(-1)
            loss = torch.mean((q_sa - y) ** 2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            # EMA target update
            with torch.no_grad():
                for p, pt in zip(q.parameters(), q_tgt.parameters()):
                    pt.data.mul_(1.0 - cfg.target_tau).add_(cfg.target_tau * p.data)

            if step % 10 == 0:
                print(f"[FQE] step {step}/{cfg.steps}", flush=True)

            step += 1
            if step >= cfg.steps:
                break

    return q


@torch.no_grad()
def fqe_score_initial_states(
    q_fqe: CriticQDiscrete,
    pi: DiscretePolicy,
    s0: np.ndarray,
    n_actions: int,
    device: torch.device,
) -> float:
    s0t = torch.from_numpy(s0).float().to(device)
    probs = pi_action_probs(pi, s0t)  # [N, A]
    qs = []
    for aidx in range(n_actions):
        a_tensor = torch.full((s0t.shape[0],), aidx, device=device, dtype=torch.long)
        qs.append(q_fqe(s0t, a_tensor).squeeze(-1))
    qs = torch.stack(qs, dim=-1)  # [N, A]
    v0 = torch.sum(probs * qs, dim=-1)  # [N]
    return float(v0.mean().item())


# KL and Unsupported Actions functions

@torch.no_grad()
def kl_pi_vs_behavior(
    pi: DiscretePolicy,
    beh: nn.Module,
    s: np.ndarray,
    device: torch.device,
    eps: float = 1e-8,
    batch_size: int = 4096,
) -> float:
    pi.eval()
    beh.eval()

    s_t = torch.from_numpy(s).float()
    dl = DataLoader(torch.utils.data.TensorDataset(s_t), batch_size=batch_size, shuffle=False)

    kls = []
    for (sb,) in dl:
        sb = sb.to(device)
        pi_p = pi_action_probs(pi, sb)  # [B,A]
        b_p = behavior_probs(beh, sb)   # [B,A]
        kl = torch.sum(pi_p * (torch.log(pi_p + eps) - torch.log(b_p + eps)), dim=-1)
        kls.append(kl.detach().cpu().numpy())
    return float(np.mean(np.concatenate(kls)))


@torch.no_grad()
def unsupported_action_pct(
    pi: DiscretePolicy,
    beh: nn.Module,
    s: np.ndarray,
    device: torch.device,
    thresh: float = 1e-3,
    batch_size: int = 4096,
) -> float:
    pi.eval()
    beh.eval()

    s_t = torch.from_numpy(s).float()
    dl = DataLoader(torch.utils.data.TensorDataset(s_t), batch_size=batch_size, shuffle=False)

    total = 0
    bad = 0
    for (sb,) in dl:
        sb = sb.to(device)
        a_pi = torch.argmax(pi(sb), dim=-1)         # greedy action
        b_p = behavior_probs(beh, sb)              # [B,A]
        chosen_prob = b_p.gather(1, a_pi.view(-1, 1)).squeeze(1)
        bad += int((chosen_prob < thresh).sum().item())
        total += sb.shape[0]
    return 100.0 * bad / max(total, 1)


# CWPDIS Metric Functions

@torch.no_grad()
def compute_cwpdis(
    df: pd.DataFrame,
    pi: DiscretePolicy,
    beh: nn.Module,
    state_cols: List[str],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    n_actions: int,
    device: torch.device,
    gamma: float = 0.99,
    eps: float = 1e-8,
    max_episodes: int = 0,
    max_horizon: int = 0
) -> float:
    """
    CWPDIS over full dataset episodes.
    """
    ensure_cols(df, ["stay_id", "bin", "reward", "done", "action"])

    if max_episodes and max_episodes > 0:
        unique_ids = df["stay_id"].drop_duplicates()
        k = min(max_episodes, len(unique_ids))
        sampled = unique_ids.sample(k, random_state=0)
        df = df[df["stay_id"].isin(sampled)]

    pi.eval()
    beh.eval()

    # Sort to ensure proper order inside episodes
    df_sorted = df.sort_values(["stay_id", "bin"]).reset_index(drop=True)

    # Precompute pi(a|s) and b(a|s) for logged actions
    s = df_sorted[state_cols].to_numpy(dtype=np.float32)
    s = normalize(s, state_mean, state_std)
    a = df_sorted["action"].to_numpy(dtype=np.int64)
    r = df_sorted["reward"].to_numpy(dtype=np.float32)
    done = df_sorted["done"].to_numpy(dtype=np.float32)

    s_t = torch.from_numpy(s).float().to(device)
    a_t = torch.from_numpy(a).long().to(device)

    # probabilities for taken actions
    pi_probs_all = pi_action_probs(pi, s_t)     
    b_probs_all = behavior_probs(beh, s_t)       
    pi_taken = pi_probs_all.gather(1, a_t.view(-1, 1)).squeeze(1).cpu().numpy()
    b_taken = b_probs_all.gather(1, a_t.view(-1, 1)).squeeze(1).cpu().numpy()

    rho = pi_taken / (b_taken + eps)
    rho = np.clip(rho, 0.0, 100.0)   # cap extreme ratios for stability

    # Build episode indices
    stay_ids = df_sorted["stay_id"].to_numpy()
    # compute weights per timestep across episodes as cumulative product of rho
    cwpdis_sum = 0.0

    # Normalizing and building arrays per episode.
    episodes = []
    start = 0
    for i in range(1, len(df_sorted)):
        if stay_ids[i] != stay_ids[i-1]:
            episodes.append((start, i))
            start = i
    episodes.append((start, len(df_sorted)))

    # Determine max horizon
    maxT = max((j - i) for i, j in episodes)

    if max_horizon and max_horizon > 0:
        maxT = min(maxT, max_horizon)

    # Precompute per-episode cumulative weights and rewards
    # Store arrays length T for each episode
    ep_cumw = []
    ep_r = []
    for i, j in episodes:
        rr = rho[i:j]
        # cumulative product
        cw = np.cumprod(rr, axis=0)
        ep_cumw.append(cw)
        ep_r.append(r[i:j])

    # For each timestep compute normalized weights and add reward
    for t in range(maxT):
        w_t = []
        r_t = []
        for e in range(len(episodes)):
            if t < len(ep_cumw[e]):
                w_t.append(ep_cumw[e][t])
                r_t.append(ep_r[e][t])
        if not w_t:
            continue
        w_t = np.array(w_t, dtype=np.float64)
        r_t = np.array(r_t, dtype=np.float64)
        denom = np.sum(w_t) + 1e-12
        w_norm = w_t / denom
        cwpdis_sum += (gamma ** t) * float(np.sum(w_norm * r_t))

    return float(cwpdis_sum)


# Argument settings

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Transitions parquet")
    p.add_argument("--run_dir", type=str, required=True, help="Folder with best.pt and schema_and_norm.json")
    p.add_argument("--ckpt", type=str, default="best.pt", choices=["best.pt", "final.pt"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Splits / behavior model
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--beh_hidden", type=int, default=256)
    p.add_argument("--beh_epochs", type=int, default=10)
    p.add_argument("--beh_lr", type=float, default=1e-3)

    # Unsupported threshold
    p.add_argument("--unsupported_thresh", type=float, default=1e-3)

    # CWPDIS / FQE
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--fqe_steps", type=int, default=50)
    p.add_argument("--fqe_lr", type=float, default=1e-3)
    p.add_argument("--fqe_batch_size", type=int, default=2048)

    p.add_argument("--cwpdis_max_episodes", type=int, default=0,
              help="If >0, sample this many stay_ids for CWPDIS (0 = use all).")
    p.add_argument("--cwpdis_max_horizon", type=int, default=0,
              help="If >0, truncate CWPDIS to this many timesteps (0 = full length).")

    # RNG
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)

    # Load schema and checkpoint
    schema_path = os.path.join(args.run_dir, "schema_and_norm.json")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Missing schema_and_norm.json in {args.run_dir}")

    with open(schema_path, "r") as f:
        schema = json.load(f)

    ckpt_path = os.path.join(args.run_dir, args.ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_from_ckpt = ckpt.get("config", {})
    hidden = int(cfg_from_ckpt.get("hidden", 256))

    state_cols = schema["state_cols"]
    next_state_cols = schema["next_state_cols"]
    n_actions = int(schema["n_actions"])
    state_mean = np.array(schema["state_mean"], dtype=np.float32)
    state_std = np.array(schema["state_std"], dtype=np.float32)
    next_state_mean = np.array(schema["next_state_mean"], dtype=np.float32)
    next_state_std = np.array(schema["next_state_std"], dtype=np.float32)

    # Load data
    df = pd.read_parquet(args.data)
    ensure_cols(df, ["action", "reward", "done"])
    ensure_cols(df, state_cols)
    ensure_cols(df, next_state_cols)

    df[state_cols] = df[state_cols].fillna(0.0)
    df[next_state_cols] = df[next_state_cols].fillna(0.0)

    # Build arrays and normalize
    s = df[state_cols].to_numpy(dtype=np.float32)
    sp = df[next_state_cols].to_numpy(dtype=np.float32)
    a = df["action"].to_numpy(dtype=np.int64)
    r = df["reward"].to_numpy(dtype=np.float32)
    d = df["done"].to_numpy(dtype=np.float32)

    s_n = normalize(s, state_mean, state_std)
    sp_n = normalize(sp, next_state_mean, next_state_std)

    # Train/val split for behavior & KL/unsupported
    n = len(df)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_val = int(args.val_frac * n)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    s_tr, a_tr = s_n[tr_idx], a[tr_idx]
    s_val = s_n[val_idx]

    # Instantiate and load pi
    pi = DiscretePolicy(state_dim=s_n.shape[1], n_actions=n_actions, hidden=hidden).to(device)
    pi.load_state_dict(ckpt["pi"])
    pi.eval()

    # Fit clinician behavior model b(a|s)
    beh = fit_behavior_model(
        s_train=s_tr,
        a_train=a_tr,
        n_actions=n_actions,
        device=device,
        seed=args.seed,
        hidden=args.beh_hidden,
        epochs=args.beh_epochs,
        lr=args.beh_lr,
    )
    beh.eval()

    # KL and unsupported on validation states
    kl = kl_pi_vs_behavior(pi, beh, s_val, device=device)
    unsupported = unsupported_action_pct(
        pi, beh, s_val, device=device, thresh=args.unsupported_thresh
    )

    # CWPDIS on full dataset
    cwpdis = compute_cwpdis(
        df=df,
        pi=pi,
        beh=beh,
        state_cols=state_cols,
        state_mean=state_mean,
        state_std=state_std,
        n_actions=n_actions,
        device=device,
        gamma=args.gamma,
        max_episodes=args.cwpdis_max_episodes,
        max_horizon=args.cwpdis_max_horizon,
    )

    # FQE: train evaluation Q under pi, then score initial states
    # initial states: first bin of each stay_id
    
    if "stay_id" in df.columns and "bin" in df.columns:
        df0 = df.sort_values(["stay_id", "bin"]).groupby("stay_id", as_index=False).first()
        s0 = df0[state_cols].to_numpy(dtype=np.float32)
        s0 = normalize(s0, state_mean, state_std)
    else:
        # fallback: random subset of states as "initial"
        s0 = s_n[np.random.choice(len(s_n), size=min(10000, len(s_n)), replace=False)]

    fqe_cfg = FQEConfig(
        gamma=args.gamma,
        lr=args.fqe_lr,
        batch_size=args.fqe_batch_size,
        steps=args.fqe_steps,
        seed=args.seed,
    )
    q_fqe = train_fqe(
        s=s_n, a=a, r=r, sp=sp_n, d=d,
        pi=pi, n_actions=n_actions, hidden=hidden,
        device=device, cfg=fqe_cfg
    )
    fqe = fqe_score_initial_states(q_fqe, pi, s0, n_actions=n_actions, device=device)

    out = {
        "run_dir": args.run_dir,
        "ckpt": args.ckpt,
        "seed": args.seed,
        "fqe": fqe,
        "kl_pi_vs_clinician": kl,
        "unsupported_pct": unsupported,
        "cwpdis": cwpdis,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()