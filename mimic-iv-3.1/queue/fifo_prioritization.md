# Patient Prioritization Analysis

## FIFO Baseline

Before comparing the learned prioritization methods, the queue simulation is run using a **First-In-First-Out (FIFO)** policy.

FIFO ignores all model-derived priority scores and the waiting-time weighting parameter (\alpha). Whenever the treatment server becomes available, the patient with the earliest arrival time is selected.

Run FIFO from the `queue/` directory:

```bash
python3 queue_simulation.py \
    --test_file test.parquet \
    --queue_policy fifo \
    --output_dir queue_fifo \
    --seed 42
```

This produces:

```text
queue_fifo/
├── queue_metrics.csv
└── queue_results.parquet
```

The FIFO run must use the **same arrival rate, service-time parameters, and random seed** as the model-based queue simulations. This ensures that patients have identical simulated arrivals and service times across policies, so differences arise only from queue ordering.

The current FIFO baseline for 7,354 test patients is:

| Metric               |       FIFO |
| -------------------- | ---------: |
| Mean waiting time    |  53.42 min |
| Median waiting time  |  41.83 min |
| P90 waiting time     | 125.87 min |
| Maximum waiting time |     6.92 h |
| Mean time in system  |    1.629 h |

## Purpose of the Patient Prioritization Analysis

The aggregate queue metrics show that learned policies can change the distribution of waiting times without substantially changing the overall mean waiting time.

The purpose of `patient_prioritization.py` is therefore to determine **which patients are moved earlier or later relative to FIFO**.

For each patient (i) and learned policy (m), we calculate:

[
\Delta W_i^{(m)}
================

W_i^{(m)} - W_i^{(\mathrm{FIFO})}.
]

Therefore:

* (\Delta W < 0): patient was treated **earlier** than under FIFO.
* (\Delta W > 0): patient was treated **later** than under FIFO.
* (\Delta W = 0): waiting time was unchanged.

The comparison is patient-level: the same `stay_id` is matched across FIFO and each learned policy.

## Current Policies

The current analysis uses:

```text
FIFO -> queue_fifo/queue_results.parquet
BCQ  -> queue_bcq_stable/queue_results.parquet
IQL  -> queue_iql_stable/queue_results.parquet
```

BC is currently excluded because the corresponding patient-level `queue_results.parquet` is not yet available in the shared queue directory. It can be added once the file is available.

DDQN can similarly be added after its scoring and queue simulation are completed.

## Clinical Information

Clinical information is obtained from the common:

```text
test.parquet
```

For each ICU stay, the first observed state is treated as the queue-entry state. Its `sofa_total` is therefore used as the patient's **entry SOFA score**.

Patients are currently grouped as:

```text
SOFA 2-3
SOFA 4-5
SOFA 6+
```

The current high-severity definition is:

```text
SOFA >= 6
```

This threshold is currently an analysis choice and can be adjusted if needed.

Patient outcome is derived from the **terminal transition**, rather than the first queue-entry transition:

```text
reward = +100 -> survived
reward = -100 -> died
```

This distinction is important because intermediate transitions have reward `0`.

## Running the Analysis

From the `queue/` directory:

```bash
python3 patient_prioritization.py
```

The expected directory structure is:

```text
queue/
├── patient_prioritization.py
├── queue_simulation.py
├── test.parquet
├── queue_fifo/
│   └── queue_results.parquet
├── queue_bcq_stable/
│   └── queue_results.parquet
└── queue_iql_stable/
    └── queue_results.parquet
```

## Outputs

The script creates:

```text
patient_prioritization/
├── patient_level_comparison.csv
├── severity_summary.csv
├── high_severity_summary.csv
├── outcome_summary.csv
└── policy_summary.csv
```

### `patient_level_comparison.csv`

Contains the patient-level comparison across policies, including:

* Entry SOFA
* Outcome
* FIFO waiting time
* BCQ/IQL waiting time
* Difference in waiting time relative to FIFO
* Whether each model treated the patient earlier than FIFO

This is the underlying data used to construct the other summaries.

### `severity_summary.csv`

Reports queue performance separately for:

```text
SOFA 2-3
SOFA 4-5
SOFA 6+
```

For each policy it reports:

* Number of patients
* Mean waiting time
* Median waiting time
* P90 waiting time
* Mean and median change relative to FIFO
* Percentage of patients treated earlier than FIFO

### `high_severity_summary.csv`

Provides a compact analysis specifically for patients with:

```text
SOFA >= 6
```

This is currently the main patient-prioritization result intended for the paper.

### `outcome_summary.csv`

Separates patients according to eventual hospital outcome:

```text
Survived
Died
```

and compares their waiting times across policies.

### `policy_summary.csv`

Provides overall prioritization statistics, including waiting-time metrics and the Spearman association between entry SOFA and treatment rank.

## Current Interpretation

The analysis is intended to distinguish **overall queue efficiency** from **patient prioritization**.

A learned policy may have approximately the same mean waiting time as FIFO while substantially changing who waits. Therefore, lower median waiting time alone should not be interpreted as universally better queue performance.

The main question is whether the learned policies redistribute waiting time in a clinically meaningful way—for example, by reducing waiting times for high-severity patients—while also examining potential increases in tail waiting times.

For this reason, FIFO should always be treated as the primary reference policy when interpreting the patient-prioritization outputs.
