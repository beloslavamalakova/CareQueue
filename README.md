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
ddqn\
├── ddqn_processing.py
├── ddqn_processing_2.py
├── build_states_duckdb.py
├── build_transitions_table.py
```
### Core Files

- **ddqn_processing.py**
  - Builds transition dataset for DDQN training  
  - Filters MIMIC-IV data to a sepsis patient cohort  
  - Constructs states using time bins  
  - Defines:
    - Actions (clinical procedures)
    - Rewards (+1 survival, -1 death)

- **ddqn_processing_2.py**
  - Optimized version of the processing pipeline  
  - Handles large datasets using chunked processing  
  - Reduces memory usage for HPC environments  

- **build_states_duckdb.py**
  - Aggregates raw ICU data into structured state vectors  
  - Uses DuckDB for fast large-scale processing  
  - Produces features like HR, BP, SpO2 per time bin  

- **build_transitions_table.py**
  - Constructs full `(s, a, r, s')` transition dataset  
  - Uses SQL joins to align:
    - States
    - Actions (procedures)
    - Outcomes (rewards)  
  - Outputs data in Parquet format for training  

---

## 4. How It Works (Architecture / Flow)

### 🔹 Step 1: State Construction

Patient data is aggregated into **4-hour time bins**, where each bin represents a state:
- Heart Rate  
- Blood Pressure  
- Temperature  
- Oxygen saturation  

---

### 🔹 Step 2: Transition Dataset

Each patient trajectory is converted into:
(s, a, r, s')

- **s** → current patient state  
- **a** → clinical intervention  
- **r** → outcome-based reward  
- **s'** → next state  

---

### 🔹 Step 3: BCQ Training (Core Logic)

BCQ combines **behavior cloning + Q-learning**, with a constraint on allowed actions.

#### 1. Behavior Cloning Network
Learns clinician policy:
π(a | s)

---

#### 2. Q-Network
Learns:
Q(s, a)

---

#### 3. BCQ Constraint Mechanism (Key Idea)

Only allows actions that are likely under the behavior policy:
π(a | s) > τ
Then selects:
max_a Q(s', a) (only over allowed actions)


This prevents unrealistic or unsafe decisions — crucial for healthcare.

---

#### 4. Training Details

- Streaming batches from Parquet (memory efficient)  
- Separate optimizers for:
  - BC network  
  - Q network  
- Target network for stability  
- Class-weighted BC loss (handles action imbalance)  

---

### 🔹 Step 4: Evaluation

Per epoch, the model logs:
- Behavior cloning loss  
- Q-learning loss  

Saved to: 
metrics.csv

Plots generated using: 
plots.py


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

Step 1: Build States
python build_states_duckdb.py

Step 2: Build Transitions
python build_transitions_table.py

Step 3: Train BCQ

```
python bcq_end2end.py \
  --inp cache/transitions_4h.parquet \
  --outdir cache/bcq_run
```

Step 4: Plot Results
python plots.py --run_dir cache/bcq_run

---

## 8. Testing

Testing not yet implemented.

## 9. Future improvements:

Queing system for simulation of patient arrivals

## 10. Contributing

This is a project developed by Honors Academy AI track students at TU Eindhoven:

Ayush Jain

Beloslava Malakova

Anusha Astana

Julia Kryłowicz

## 11.Documentation / References

(To be added)

## 12. Known Issues / Limitations

(To be added)

## 13. License

This project uses the MIMIC-IV dataset, which requires PhysioNet credentialing and adherence to its data usage agreement.