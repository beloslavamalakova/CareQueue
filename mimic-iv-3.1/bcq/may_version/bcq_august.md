# BCQ -> Common Retraining and Queue Simulation

This describes how BCQ is integrated with the common preprocessing and queue simulation implemented by Ayush.

The overall pipeline is:

```text
common_preprocessing.py
        ↓
train.parquet + test.parquet
        ↓
Retrain BCQ on train.parquet
        ↓
Recalculate BCQ scores on test.parquet
        ↓
bcq_scores.parquet
        ↓
queue_simulation.py
        ↓
queue_bcq/
```

## 1. Run the Common Preprocessing

Ayush's common preprocessing is located in:

```text
mimic-iv-3.1/queue/common_preprocessing.py
```

It creates the common:

```text
train.parquet
test.parquet
```

used by all models.

From the `queue/` directory:

```bash
python3 common_preprocessing.py --base ..
```

This produces:

```text
queue/
├── train.parquet
└── test.parquet
```

## 2. Retrain BCQ

Since the preprocessing changed, the existing BCQ model needs to be retrained using the new common `train.parquet`.

BCQ training is located in:

```text
bcq/may_version/bcq_training.py
```

From `bcq/may_version/`:

```bash
python3 bcq_training.py \
    --data ../../queue/train.parquet \
    --save_dir Output_common \
    --epochs 30
```

The trained model and its normalization information are saved in:

```text
Output_common/
├── best.pt
├── final.pt
├── metrics.csv
└── schema_and_norm.json
```

The BCQ training code can directly use the common parquet because it expects the `s_*`, `s_next_*`, `action`, `reward`, and `done` variables provided by the common preprocessing.

## 3. Recalculate BCQ Patient Scores

The old BCQ scores cannot be reused because BCQ has now been retrained on the new common dataset.

The new scoring script is:

```text
bcq/may_version/bcq_queue_score.py
```

Run:

```bash
python3 bcq_queue_score.py
```

It:

1. Loads `queue/test.parquet`.
2. Takes the first state of every patient.
3. Loads the newly trained `Output_common/best.pt`.
4. Uses the BCQ policy to calculate the patient's Q-value.
5. Converts the BCQ scores to normalized priority scores.
6. Saves them in the format required by Ayush's queue.

Output:

```text
queue/bcq_scores.parquet
```

containing:

```text
stay_id
priority_score
raw_bcq_score
recommended_action
```

For the current run:

```text
Test patients: 7,354
Patients scored: 7,354

Raw BCQ score:
min  = 31.2809
mean = 57.5425
max  = 74.7343

Priority score:
0.0 – 1.0
```

## 4. Run Queue Simulation

Return to:

```text
mimic-iv-3.1/queue/
```

Ayush's `queue_simulation.py` is model-agnostic and only requires the model to provide `stay_id` and `priority_score`.

Run:

```bash
python3 queue_simulation.py \
    --test_file test.parquet \
    --score_mode file \
    --score_file bcq_scores.parquet \
    --output_dir queue_bcq
```

The queue uses the BCQ priority score together with waiting time:

```text
Effective Priority =
    Model Score + α × Waiting Time
```

and simulates patient arrivals, treatment and queue ordering.

## 5. Results

The final BCQ queue outputs are:

```text
queue/queue_bcq/
├── queue_results.parquet
└── queue_metrics.csv
```

`queue_results.parquet` contains the patient-level simulated queue outcomes, while `queue_metrics.csv` contains the aggregate queue metrics.

Therefore, the complete BCQ workflow is:

```text
Ayush common preprocessing
        ↓
train.parquet
        ↓
retrain BCQ
        ↓
test.parquet + new BCQ model
        ↓
recalculate BCQ scores
        ↓
bcq_scores.parquet
        ↓
Ayush queue simulation
        ↓
BCQ queue results
```
