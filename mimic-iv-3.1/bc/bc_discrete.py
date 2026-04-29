#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset


# ------------------------- Config -------------------------
@dataclass
class BCConfig:
    lr: float = 5e-5
    weight_decay: float = 0.0
    batch_size: int = 1024
    epochs: int = 30
    num_workers: int = 2
    hidden: int = 256
    dropout: float = 0.0
    seed: int = 0
    val_frac: float = 0.1
    use_class_weights: bool = True
    label_smoothing: float = 0.0
    clip_grad_norm: float = 10.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------- Utilities -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def stable_unit_hash(value: object) -> float:
    h = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    return int(h[:15], 16) / float(16**15 - 1)


def split_by_stay_id(df: pd.DataFrame, val_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "stay_id" not in df.columns:
        raise ValueError("Expected a 'stay_id' column for stay-level splitting.")

    unique_stays = pd.Series(df["stay_id"].dropna().unique())
    val_stays = set(unique_stays[unique_stays.map(stable_unit_hash) < val_frac].tolist())

    val_mask = df["stay_id"].isin(val_stays)
    train_df = df.loc[~val_mask].reset_index(drop=True)
    val_df = df.loc[val_mask].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError(
            f"Empty split detected. train={len(train_df)}, val={len(val_df)}. "
            "Try a different val_frac or check stay_id values."
        )
    return train_df, val_df


def compute_action_weights(actions: np.ndarray, n_actions: int) -> np.ndarray:
    counts = np.bincount(actions.astype(np.int64), minlength=n_actions).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    weights = inv / inv.mean()
    return weights.astype(np.float32)


def action_distribution(actions: np.ndarray, n_actions: int) -> Dict[int, float]:
    counts = np.bincount(actions.astype(np.int64), minlength=n_actions).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return {i: 0.0 for i in range(n_actions)}
    return {i: float(c / total) for i, c in enumerate(counts)}


# ------------------------- Dataset -------------------------
class BCDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        state_cols: List[str],
        state_mean: np.ndarray,
        state_std: np.ndarray,
    ):
        s = df[state_cols].to_numpy(dtype=np.float32)
        s = (s - state_mean) / state_std
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        a = df["action"].to_numpy(dtype=np.int64)

        self.s = torch.from_numpy(s)
        self.a = torch.from_numpy(a)

    def __len__(self) -> int:
        return self.s.shape[0]

    def __getitem__(self, idx: int):
        return self.s[idx], self.a[idx]


# ------------------------- Model -------------------------
class DiscreteBCPolicy(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256, dropout: float = 0.0):
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

    @torch.no_grad()
    def act_mode(self, s: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.forward(s), dim=-1)


# ------------------------- Evaluation -------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    n_actions: int,
) -> Dict[str, object]:
    model.eval()
    losses: List[float] = []
    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []

    for s, a in loader:
        s = s.to(device)
        a = a.to(device)

        logits = model(s)
        loss = criterion(logits, a)

        losses.append(float(loss.item()))
        y_true.append(a.cpu().numpy())
        y_pred.append(torch.argmax(logits, dim=-1).cpu().numpy())

    y_true_np = np.concatenate(y_true)
    y_pred_np = np.concatenate(y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        labels=list(range(n_actions)),
        average=None,
        zero_division=0,
    )

    per_action = {
        int(i): {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(n_actions)
    }

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "macro_f1": float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "true_action_dist": action_distribution(y_true_np, n_actions),
        "pred_action_dist": action_distribution(y_pred_np, n_actions),
        "per_action": per_action,
    }


# ------------------------- Training -------------------------
def train_bc_discrete(df: pd.DataFrame, cfg: BCConfig, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    if "action" not in df.columns:
        raise ValueError("Missing required column: action")
    if "stay_id" not in df.columns:
        raise ValueError("Missing required column: stay_id")

    state_cols, next_state_cols = infer_columns_discrete(df)
    if not state_cols:
        raise ValueError("Could not infer state columns. Need 's_*' columns.")
    if not next_state_cols:
        print("[warning] No s_next_* columns found. BC does not use them, but RL methods usually do.")

    a_min = int(df["action"].min())
    a_max = int(df["action"].max())
    if a_min < 0:
        raise ValueError(f"Action has negative values (min={a_min}).")
    n_actions = a_max + 1

    train_df, val_df = split_by_stay_id(df, val_frac=cfg.val_frac)
    state_mean, state_std = compute_norm_stats(train_df, state_cols)

    train_ds = BCDataset(train_df, state_cols, state_mean, state_std)
    val_ds = BCDataset(val_df, state_cols, state_mean, state_std)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = DiscreteBCPolicy(
        state_dim=len(state_cols),
        n_actions=n_actions,
        hidden=cfg.hidden,
        dropout=cfg.dropout,
    ).to(device)

    class_weights = None
    if cfg.use_class_weights:
        weights_np = compute_action_weights(train_df["action"].to_numpy(), n_actions=n_actions)
        class_weights = torch.tensor(weights_np, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: List[Dict[str, object]] = []
    best_metric = -float("inf")
    best_epoch = -1

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        batch_losses: List[float] = []

        for s, a in train_loader:
            s = s.to(device, non_blocking=True)
            a = a.to(device, non_blocking=True)

            logits = model(s)
            loss = criterion(logits, a)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.clip_grad_norm is not None and cfg.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            optimizer.step()

            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        val_metrics = evaluate(model, val_loader, criterion, device, n_actions)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(record)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        score = float(val_metrics["macro_f1"])
        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "state_cols": state_cols,
                    "state_mean": state_mean,
                    "state_std": state_std,
                    "n_actions": n_actions,
                    "config": asdict(cfg),
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_metric,
                },
                os.path.join(save_dir, "bc_discrete_best.pt"),
            )

        with open(os.path.join(save_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    final_val_metrics = evaluate(model, val_loader, criterion, device, n_actions)

    summary = {
        "config": asdict(cfg),
        "n_rows": int(len(df)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "state_dim": int(len(state_cols)),
        "n_actions": int(n_actions),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_metric),
        "final_val": final_val_metrics,
    }

    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete.")
    print(json.dumps(summary, indent=2))


# ------------------------- CLI -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Discrete behavior cloning baseline for CareQueue-style offline RL data.")
    p.add_argument("--data", type=str, required=True, help="Path to parquet file with s_*, s_next_*, action, stay_id.")
    p.add_argument("--save-dir", type=str, default="./bc_runs/discrete")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--clip-grad-norm", type=float, default=10.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.data)

    cfg = BCConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        hidden=args.hidden,
        dropout=args.dropout,
        seed=args.seed,
        val_frac=args.val_frac,
        use_class_weights=not args.no_class_weights,
        label_smoothing=args.label_smoothing,
        clip_grad_norm=args.clip_grad_norm,
        device=args.device,
    )

    train_bc_discrete(df=df, cfg=cfg, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
