# BC Expected-Intensity Scoring Script

This README explains the queue-specific BC scoring workflow built around:

```text
bc_expected_intensity_scoring.py
```

The purpose of this script is to create a queue-compatible BC score file:

```text
bc_scores.parquet
```

with at least:

```text
stay_id
priority_score
```

## Current Completed Workflow

The current BC queue workflow now follows the main queue README more closely:

1. MIMIC-IV data was preprocessed into shared queue files:

```text
mimic-iv-3.1/queue/train.parquet
mimic-iv-3.1/queue/test.parquet
```

2. BC was retrained using the shared `train.parquet`.

3. The retrained BC checkpoint was used to score the shared `test.parquet`.

4. The queue simulation was rerun using the new `bc_scores.parquet`.

Current retrained BC checkpoint:

```text
mimic-iv-3.1/bc/bc_runs/discrete_queue_shared/bc_discrete_best.pt
```

Current BC score output:

```text
mimic-iv-3.1/queue/bc_scores.parquet
```

Current queue outputs:

```text
mimic-iv-3.1/queue/queue_bc/queue_results.parquet
mimic-iv-3.1/queue/queue_bc/queue_metrics.csv
```

## Why This Script Exists

The queue simulator is model-agnostic. It only needs:

```text
stay_id
priority_score
```

For RL models, the priority score can come from a Q-value.

BC is different. Behavior Cloning does not learn Q-values. It learns:

```text
Given this patient state, what treatment action would a clinician probably take?
```

So this script converts BC action probabilities into a single priority score using expected treatment intensity.

In plain language:

```text
If BC thinks a clinician would likely give stronger fluids/vasopressors,
the patient receives a higher queue priority score.
```

## Expected Treatment Intensity

The action space is currently:

```text
5 vasopressor bins x 5 fluid bins = 25 actions
```

Each action is decoded like this:

```text
vaso_bin = action // 5
fluid_bin = action % 5
intensity = vaso_bin + fluid_bin
```

This gives each action an intensity from 0 to 8.

The script computes:

```text
raw_expected_intensity =
    sum(probability_of_action * action_intensity)
```

Then it normalizes:

```text
priority_score = raw_expected_intensity / 8
```

So the final `priority_score` is between 0 and 1.

## What The Script Does

1. Loads the shared `test.parquet`.
2. Sorts by `stay_id` and `bin`.
3. Keeps the first observed row for each `stay_id`.
4. Loads the trained BC checkpoint.
5. Rebuilds the same BC neural network architecture used in `bc_discrete.py`.
6. Normalizes patient states using the checkpoint's saved `state_mean` and `state_std`.
7. Runs the BC model to get probabilities for all 25 actions.
8. Converts each action to treatment intensity.
9. Computes expected treatment intensity per patient.
10. Normalizes that score to 0-1.
11. Saves a parquet file that the queue simulation can read.

## Script Inputs

```text
--test_file
```

The shared queue-preprocessed `test.parquet`.

```text
--checkpoint
```

The trained BC checkpoint.

For the current completed run:

```text
mimic-iv-3.1/bc/bc_runs/discrete_queue_shared/bc_discrete_best.pt
```

```text
--output_file
```

Where to save the queue-compatible BC score file.

For the current completed run:

```text
mimic-iv-3.1/queue/bc_scores.parquet
```

## Script Output

The output parquet contains:

```text
stay_id
priority_score
raw_expected_intensity
predicted_action
max_action_probability
action_entropy
```

The queue simulator only requires:

```text
stay_id
priority_score
```

The other columns are diagnostic columns to help inspect and explain the BC scores.

If you run with:

```text
--include_action_probs
```

the script also saves:

```text
p_action_0
p_action_1
...
p_action_24
```

## Commands Used

### 1. Preprocess MIMIC-IV Into Shared Queue Data

```powershell
python .\mimic-iv-3.1\queue\common_preprocessing.py `
  --base "C:\Users\20243322\OneDrive - TU Eindhoven\Desktop\A - honors\mimic-iv-3.1" `
  --out_dir .\mimic-iv-3.1\queue `
  --tmp .\mimic-iv-3.1\queue\duckdb_tmp `
  --threads 8 `
  --mem_limit 8GB
```

This produced:

```text
Training rows: 975,723
Testing rows: 242,550
```

### 2. Retrain BC On Shared train.parquet

```powershell
python .\mimic-iv-3.1\bc\bc_discrete.py `
  --data .\mimic-iv-3.1\queue\train.parquet `
  --save-dir .\mimic-iv-3.1\bc\bc_runs\discrete_queue_shared `
  --epochs 30 `
  --batch-size 4096 `
  --num-workers 0 `
  --device cpu
```

Training summary:

```text
Training rows used by BC: 975,723
BC internal train rows: 880,325
BC internal validation rows: 95,398
Best epoch: 2
Best validation macro F1: 0.0532
Final validation accuracy: 0.2273
Final validation macro F1: 0.0499
```

### 3. Generate BC Queue Scores

```powershell
python .\mimic-iv-3.1\queue\bc_expected_intensity_scoring.py `
  --test_file .\mimic-iv-3.1\queue\test.parquet `
  --checkpoint .\mimic-iv-3.1\bc\bc_runs\discrete_queue_shared\bc_discrete_best.pt `
  --output_file .\mimic-iv-3.1\queue\bc_scores.parquet `
  --device cpu
```

Scoring summary:

```text
Test ICU stays scored: 7,354
Mean priority_score: 0.5031
Median priority_score: 0.4890
Min priority_score: 0.4615
Max priority_score: 0.6484
```

### 4. Run Queue Simulation

The queue simulation was run using:

```text
test_file = mimic-iv-3.1/queue/test.parquet
score_file = mimic-iv-3.1/queue/bc_scores.parquet
output_dir = mimic-iv-3.1/queue/queue_bc
arrival_rate = 1.0
service_hours = 0.75
alpha = 0.001
seed = 42
```

The queue was executed with queue visualization disabled to avoid printing thousands of snapshots.

Final queue metrics:

```text
Patients: 7,354
Mean waiting time: 0.8950 hours = 53.70 minutes
Median waiting time: 0.4680 hours = 28.08 minutes
90th percentile waiting time: 2.3488 hours = 140.93 minutes
Max waiting time: 15.2683 hours
Mean time in system: 1.6338 hours
Mean BC priority score: 0.5031
Mean SOFA: 2.7324
```

## Important Interpretation Note

This BC score is not a Q-value and not a direct survival estimate.

It means:

```text
Expected clinician treatment intensity according to the BC policy.
```

That makes it useful as a BC-based queue priority score, but it should be compared carefully against Q-value-based scores from RL models.
