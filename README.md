# CareQueue – Reinforcement Learning for Patient Prioritization

## 1. Project Overview

CareQueue is a Reinforcement Learning (RL) project that aims to **prioritize patients based on clinical data**, ensuring that those who need urgent care receive attention first.

Healthcare systems often rely on static or manual prioritization strategies, which can lead to inefficiencies. This project addresses that by learning **data-driven policies from historical ICU data** using offline RL.

The system is trained on the **MIMIC-IV dataset** and focuses on:
- Learning from patient trajectories in ICU settings  
- Modeling treatment decisions as sequential actions  
- Improving prioritization and intervention strategies  

This project explores multiple offline RL approaches, with a current focus on **BCQ (Batch-Constrained Q-Learning)** for safe policy learning.

---

## 2. Features / Capabilities

### 🔹 Current Capabilities

#### Data Processing
- Extraction and preprocessing of MIMIC-IV data  
- Feature aggregation and cleaning  
- Construction of RL datasets in `(s, a, r, s')` format  

#### Reinforcement Learning Models
- **IQL (Implicit Q-Learning)**  
- **BCQ (Batch-Constrained Q-Learning)** ← *current main pipeline*  
- **DDQN (Double Deep Q-Network)**  

---

### 🔹 Key Functionalities
- Converts raw ICU data into structured transition datasets  
- Trains offline RL agents without environment interaction  
- Supports large-scale datasets via **DuckDB + Parquet + streaming**  
- Logs and visualizes training performance  

---

### 🔹 Unique Aspects
- Uses **real-world ICU data (MIMIC-IV)**  
- Fully **offline RL (no simulator required)**  
- Focus on **safe decision-making in healthcare**  
- Implements **BCQ constraint mechanism to avoid unrealistic actions**

---

## 3. Project Structure

### BCQ

```
bcq/
├── bcq_end2end.py
├── build_states_duckdb.py
├── build_transitions_table.py
├── plots.py
```

### Core Files

- **bcq_end2end.py**
  - Full BCQ training pipeline  
  - Train/validation split (by `stay_id`, deterministic)  
  - Normalization (mean/std from train set only)  
  - Streaming training using PyArrow  
  - Logs metrics + saves checkpoints  

- **build_states_duckdb.py**
  - Aggregates ICU data into **4-hour bins**  
  - Produces patient state vectors (HR, BP, SpO2, etc.)

- **build_transitions_table.py**
  - Builds full transition dataset  
  - Defines:
    - Actions (medical procedures)
    - Rewards (+1 survival, -1 death)
    - Next states via time shifts  

- **plots.py**
  - Visualizes:
    - Validation BC loss  
    - Validation Q loss  

---

### IQL

```
iql/
 └── Version 3/
      ├── iql_processing.py
      ├── iql_training.py
      ├── eval_iql_metrics.py
      ├── iql_plots.py
      ├── run_sweep.py
```

### Core Files

- **iql_processing.py**
  - Builds the transition dataset from MIMIC-IV  
  - Performs preprocessing and feature selection  
  - Converts raw data into `(s, a, r, s')` format  

- **iql_training.py**
  - Trains the IQL (Implicit Q-Learning) model  
  - Learns:
    - Q-function (state-action value)
    - V-function (state value)
    - Policy via advantage-weighted regression  
  - Uses expectile regression for stable offline learning  

- **eval_iql_metrics.py**
  - Evaluates trained policies using offline RL metrics  
  - Includes:
    - FQE (Fitted Q Evaluation)  
    - KL divergence vs behavior policy  
    - CWPDIS estimator  

- **iql_plots.py**
  - Visualizes training and evaluation results  
  - Parses logs and generates performance plots  

- **run_sweep.py**
  - Runs hyperparameter sweeps across multiple configurations  
  - Automates experiments (e.g., seeds, learning rates)  
  - Stores and compares results  

---

### DDQN

```
ddqn/
 └── other
    ├── ddqn_processing.py
    ├── ddqn_processing_2.py
 └── outputs
 └── score
├── 25_SOFA2_preprocessing.py
├── ddqn_score.py
├── final_ddqn.py
├── graph.py
```
### Core Files

- **25_SOFA2_preprocessing.py**
  - Data preprocessing for main model
  - Accounts for the 25 actions (vasopressors, fluids)
  - Accounts for the SOFA score

- **ddqn_score.py**
  - Computes the comparison score for the DDQN model

- **final_ddqn.py**
  - Contains the final DDQN model used for project  
  - 25 discrete actions, state space dimension of 38
  - Contains a main ("online") network, and a target network
  - Target network updated periodically, classic DDQN  

- **graph.py**
  - Graphs plots based on losses computed in training script
  - Shows validation behaviour cloning loss over 30 epochs
  - Shows validation Q-Bellman loss over 30 epochs

---

## 4. How It Works (Architecture / Flow)

### 🔹 Step 1: State Construction

Patient data is aggregated into **4-hour time bins**, where each bin represents a state:
- Heart Rate  
- Blood Pressure  
- Temperature  
- Oxygen saturation  

These features are extracted from MIMIC-IV and aggregated using DuckDB for efficient large-scale processing.

---

### 🔹 Step 2: Transition Dataset

Each patient trajectory is converted into:

(s, a, r, s')

- **s** → current patient state  
- **a** → clinical intervention  
- **r** → outcome-based reward  
- **s'** → next state  

Rewards are defined as:
- +1 → patient survives  
- -1 → patient dies  
- 0 → intermediate steps  

This transition dataset is shared across all RL models.

---

### 🔹 Step 3: Model Training

Different RL algorithms are trained on the same transition dataset:

---

#### BCQ (Batch-Constrained Q-Learning)

BCQ combines **behavior cloning + Q-learning** to ensure safe offline learning.

- Learns a policy π(a|s) from clinician behavior  
- Learns Q(s,a) using Bellman updates  
- Restricts actions to those likely under the dataset:
  
  π(a|s) > τ  

- Only allowed actions are considered when computing targets  

➡️ Prevents unrealistic or unsafe actions (important for healthcare)

---

#### IQL (Implicit Q-Learning)

IQL avoids explicit behavior cloning constraints and instead uses **value-based weighting**.

- Learns:
  - Q-function Q(s,a)  
  - Value function V(s)  
- Uses **expectile regression** to estimate V(s)  
- Updates policy using **advantage-weighted regression**:
  
  Advantage = Q(s,a) - V(s)

➡️ Focuses on learning from high-quality actions without requiring action filtering

---

#### DDQN (Double Deep Q-Network)

DDQN is a value-based RL algorithm adapted for offline data.

- Learns Q(s,a) using neural networks  
- Uses a **target network** to stabilize training  
- Reduces overestimation bias compared to standard DQN  
- Selects actions using:
  
  max_a Q(s,a)

➡️ Simpler than BCQ/IQL but more prone to extrapolation error in offline settings

---

### 🔹 Step 4: Evaluation

Models are evaluated using offline metrics and training diagnostics.

#### BCQ
- Behavior cloning loss  
- Q-learning loss  

#### IQL
- FQE (Fitted Q Evaluation)  
- KL divergence vs clinician policy  
- CWPDIS  

#### DDQN
- BC, Q-loss convergence  
- Reward distribution analysis  

Outputs include:
- metrics.csv (training logs)  
- plots (loss curves and evaluation results)

---

## 5. Installation & Setup

### Requirements
- Python 3.9+
- PyTorch  
- pandas  
- numpy  
- DuckDB  
- pyarrow  
- matplotlib  

---

## 6. Data Setup

Download MIMIC-IV V 3.1

---

## 7. Usage

### Step 1: Build States

python build_states_duckdb.py

### Step 2: Build Transitions

python build_transitions_table.py

This creates the dataset used by all models.

### Step 3: Train Models

Train BCQ:

```
python bcq_end2end.py \
  --inp cache/transitions_4h.parquet \
  --outdir cache/bcq_run
```

Train IQL:

```
python iql_training.py \
  --data cache/transitions_4h.parquet \
  --save_dir output/iql_run
```

(Optional: run sweeps)
python run_sweep.py

Train DDQN

```
python final_ddqn.py \
  --data sepsis_iql_actionvec_transitions.parquet \
  --outdir ddqn_outputs
```


### Step 4: Evaluate / Visualize Results

BCQ
```
python plots.py --run_dir cache/bcq_run
```

IQL
```
python eval_iql_metrics.py
python iql_plots.py
```

DDQN

```
python graph.py
```

## 8. Testing

Testing not yet implemented.

## 9. Future improvements:

Queing system for simulation of patient arrivals

## 10. Contributing

This is a project developed by Honors Academy AI track students at TU Eindhoven:

- Ayush Jain

- Beloslava Malakova

- Anusha Astana

- Julia Kryłowicz

## 11.Documentation / References

(To be added)

## 12. Known Issues / Limitations

(To be added)

## 13. License

This project uses the MIMIC-IV dataset, which requires PhysioNet credentialing and adherence to its data usage agreement.
