"""
This file contains the DDQN model used for project CareQueue. 

Data: sepsis_iql_actionvec_transitions.parquet (IQL processing script)
25 actions, vasopressors and fluids from the new data processing file
Also added age and sex to the states, previously forgotten
"""

import os
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import json

"""
Pick device depending on what is available
"""

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print("Device:", DEVICE)

"""
Defining the states in the action space
"""

# We have 6 vitals that we would like to monitor, namely, 
# heart rate, temperature, oxygen, systolic/diastolic/mean blood pressure
STATE_NAMES = ["HR", "TEMP", "SPO2", "SBP", "DBP", "MBP"]
#STATE_SUMMARIES = ["mean", "min", "max", "std", "last", "trend"]

#S_VITALS_COLS = [f"s_{name}_{s}" for name in STATE_NAMES for s in STATE_SUMMARIES]
S_VITALS_COLS = [f"s_{name}" for name in STATE_NAMES]
S_DEMO_COLS = ["s_age", "s_sex"] # demo stands for demographic, age and sex have been added now as demographic features
S_COLS = S_VITALS_COLS + S_DEMO_COLS

# naming the next state columns 
S2_COLS = [c.replace("s_", "s_next_", 1) for c in S_COLS]

STATE_DIM = len(S_COLS) # is now 8
ACT_DIM = 25 # action space size

"""
Defining the Replay Buffer:
This buffer contains the transitions in the (s, a, r, s', done) format for episodes
We have static MIMIC-IV dataset so we can just fill from the parquet file, no live env. interaction
state_dim: state vector dimension, size: num. transitions to store
"""

class ReplayBuffer:
    def __init__(self, state_dim, size, device):
        self.device = device
        self.size   = size
        self.ptr    = 0
        self.full   = False

        self.s  = np.zeros((size, state_dim), np.float32) # current states
        self.a  = np.zeros((size, 1), np.int64) # actions
        self.r  = np.zeros((size, 1), np.float32) # rewards
        self.s2 = np.zeros((size, state_dim), np.float32) # next states
        self.d  = np.zeros((size, 1), np.float32) # done or not, 1 means it is done

    # sampling some transitions
    def sample(self, bs):
        max_i = self.size if self.full else self.ptr
        idx   = np.random.randint(0, max_i, size=bs)
        return (
            torch.tensor(self.s[idx], device=self.device),
            torch.tensor(self.a[idx], device=self.device),
            torch.tensor(self.r[idx], device=self.device).squeeze(1),
            torch.tensor(self.s2[idx], device=self.device),
            torch.tensor(self.d[idx], device=self.device).squeeze(1),
        )

"""
The Q Network:
A simple network with two hidden layers, which maps state vectors to Q values per action
d: state dimension (input size), a: number of discrete actions (output size), h: hidden layer width
"""

class QNet(nn.Module):
    def __init__(self, d, a, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, a),
        )

    def forward(self, x):
        return self.net(x)


"""
Calculating the DDQN loss
"""

def ddqn_loss(q, q_tgt, batch, gamma):
    S, A, R, S2, D = batch
    
    # Q-value for the action actually taken
    qsa = q(S).gather(1, A).squeeze(1)

    with torch.no_grad():
        # "online" (main) network picks the best action, and the target network evaluates it, standard DDQN 
        a2 = q(S2).argmax(1, keepdim=True)
        q2 = q_tgt(S2).gather(1, a2).squeeze(1)
        y  = R + gamma * q2 * (1 - D)

    return nn.SmoothL1Loss()(qsa, y)

"""
Filling the Replay Buffer:
Since we have a static dataset, we can directly fill the replay buffer from the parquet file.
Some preprocessing: normalise state values so that they are between 0 and 1, missing values are replaced with 0
df: dataframe containing the transition style columns, state, next state, action, reward, done
buf: Replay Buffer
s_cols: current state columns
s2_cols: next state columns
"""

def fill_buffer(df: pd.DataFrame, buf: ReplayBuffer, s_cols, s2_cols):
    features = s_cols + s2_cols
    df = df.copy()
    
    # replace weird values with 0
    df[features] = df[features].fillna(0).replace([np.inf, -np.inf], 0)

    # normalisation
    mu = df[features].mean()
    sd = df[features].std() + 1e-6
    df[features] = (df[features] - mu) / sd
    
    # getting s_mean, s_std, sp_mean, sp_std for schema
    s_mean = mu[s_cols].values.astype(np.float32)
    s_std = sd[s_cols].values.astype(np.float32)
    sp_mean = mu[s2_cols].values.astype(np.float32)
    sp_std = sd[s2_cols].values.astype(np.float32)

    num_rows = len(df)
    buf.s[:num_rows]  = df[s_cols].values.astype(np.float32)
    buf.a[:num_rows]  = df[["action"]].values.astype(np.int64)
    buf.r[:num_rows]  = df[["reward"]].values.astype(np.float32)
    buf.s2[:num_rows] = df[s2_cols].values.astype(np.float32)
    buf.d[:num_rows]  = df[["done"]].values.astype(np.float32)

    buf.ptr  = num_rows % buf.size
    buf.full = num_rows >= buf.size
    print(f"Direct Buffer Fill: Loaded {num_rows} transitions.")
    
    return s_mean, s_std, sp_mean, sp_std


"""
Main training loop:
1. Load data and split into train and validation sets, 90/10 split
2. Fill replay buffers for train and validation (one each)
3. Initialize main network (q), target network (qt)
4. For each epoch, train on batches from the training buffer, then evaluate on the validation buffer
5. Save best checkpoint by the losses and final weights
"""

def main():
    DATA = os.environ.get("DATA_PATH", "sepsis_iql_actionvec_transitions.parquet")
    OUT = Path(os.environ.get("OUT_DIR", "ddqn_outputs"))
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA)

    # randomly split, since we do not want to overemphasise any particular data ourselves
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    train_buf = ReplayBuffer(STATE_DIM, len(train_df), DEVICE)
    val_buf = ReplayBuffer(STATE_DIM, len(val_df), DEVICE)

    s_mean, s_std, sp_mean, sp_std = fill_buffer(train_df, train_buf, S_COLS, S2_COLS)
    fill_buffer(val_df, val_buf, S_COLS, S2_COLS)

    # main and target networks start with same weights
    q = QNet(STATE_DIM, ACT_DIM).to(DEVICE)
    qt = QNet(STATE_DIM, ACT_DIM).to(DEVICE)
    qt.load_state_dict(q.state_dict())

    opt = optim.Adam(q.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

    epochs = 30
    steps_per_epoch = 1000
    bs = 256
    gamma = 0.99

    # here is where we get the data for the graphs from, it tracks the losses per epoch
    metrics_path = OUT / "metrics.csv"
    f = open(metrics_path, "w", newline="")
    w = csv.DictWriter(f, fieldnames=["epoch", "val_bc", "val_q"])
    w.writeheader()

    best_val_q = float("inf")

    for ep in range(1, epochs + 1):
        q.train()
        for _ in range(steps_per_epoch):
            batch = train_buf.sample(bs)
            loss = ddqn_loss(q, qt, batch, gamma)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # copy main target weights to target network once per epoch in the loop
        qt.load_state_dict(q.state_dict())
        scheduler.step()

        # validation loss
        q.eval()
        with torch.no_grad():
            vb, vq = [], []
            for _ in range(50):
                b = val_buf.sample(bs)
                vq.append(ddqn_loss(q, qt, b, gamma).item())
                S, A = b[0], b[1]
                vb.append(nn.CrossEntropyLoss()(q(S), A.squeeze(1)).item())

        val_bc = float(np.mean(vb))
        val_q  = float(np.mean(vq))

        w.writerow({"epoch": ep, "val_bc": val_bc, "val_q": val_q})
        f.flush()
        
        # save best checkpoint, it has the best val DDQN loss so far
        if val_q < best_val_q:
            best_val_q = val_q
            torch.save({"q": q.state_dict(), "config": {"hidden": 256}}, OUT / "ddqn_model_best.pt")

    f.close()
    torch.save({"q": q.state_dict(), "config": {"hidden": 256}}, OUT / "ddqn_model_final.pt")
    
    # save schema + norm stats for evaluating DDQN score
    schema = {
        "state_cols": S_COLS,
        "next_state_cols": S2_COLS,
        "action_col": "action",
        "n_actions": 25,
        "state_mean": s_mean.tolist(),
        "state_std": s_std.tolist(),
        "next_state_mean": sp_mean.tolist(),
        "next_state_std": sp_std.tolist(),
    }
    with open(os.path.join(OUT, "schema_and_norm.json"), "w") as fs:
        json.dump(schema, fs, indent=2)


if __name__ == "__main__":
    main()