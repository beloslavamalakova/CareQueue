# CareQueueAI – Patient Prioritisation & Queue Simulation

## Overview

The queueing workflow is split into **three main stages**:

1. Preprocess MIMIC-IV data into a common `train.parquet` and `test.parquet`.
2. Train/evaluate each model and generate a priority score for every test patient.
3. Run the queue simulation using these scores and generate results/metrics.

The queue simulation is **model-agnostic**: it does not assume DDQN, IQL, BCQ, BC, etc. It only requires a `stay_id` and `priority_score`.

---

### Script
`preprocessing.py`

This is the common preprocessing script for all models.

It:

- Takes the MIMIC-IV dataset as input.
- Aggregates physiological measurements into fixed **4-hour time bins**.
- Builds patient states and next states.
- Constructs the **5 × 5 = 25 discrete action space** from:
  - Vasopressor dosage
  - Fluid dosage
- Calculates treatment amounts.
- Calculates a simplified SOFA score.
- Identifies an eligible sepsis cohort using:
  - Evidence of infection
  - `SOFA >= 2`
- Assigns the final reward based on hospital survival:
  - `+100` for survival
  - `-100` for hospital death
  - `0` for intermediate transitions
- **Splits patients into training and testing sets**, ensuring that data from the same patient does not appear in both sets. This split is necessary for later comparison between models. 

### Outputs

    train.parquet

    test.parquet

These files are intended to be the common dataset used by all models.

### Model Training

All models use the common train.parquet. 

Important: Since the preprocessing has changed, so all models need to be retrained using the new train.parquet before comparing their outputs.

### Generating Patient Priority Scores

Each model needs its own scoring script. Each scoring script should:

    Input:
        test.parquet
        trained model/checkpoint

    Output:
        [model]_scores.parquet

The output must contain at least:

    stay_id
    
    priority_score

For the RL models, the priority score is currently intended to be derived from the model's predicted Q-value.

Note: The existing queueing script may need small modifications to work with the scripts that will be written, particularly with variable values, and perhaps data formatwise. 

### How the Queue Simulation Works

The simulation:

    Loads the test patients.

    Uses the first observed state of each ICU stay as the patient's queue-entry state.

    Assigns each patient a priority score.

    Simulates patient arrivals using a Poisson arrival process.

    Simulates treatment/service times using an exponential distribution.

    Adds patients to a priority queue when they arrive.

    Selects the patient with the highest effective priority when treatment becomes available.

    Records waiting time, treatment time, and time spent in the system.

    Saves the resulting queue experience and summary metrics.

The current priority rule is:

    Effective Priority =
        Model Score + α × Waiting Time

where α controls how strongly waiting time affects prioritisation.

    Higher effective priority means the patient is treated first.

    Earlier arrival time is used as a tie-breaker.

Important: This Is Not a Live Simulation

    The current queue simulation is not intended to be a realistic live hospital simulation.

    It is primarily a framework for generating comparable queueing outcomes/metrics for different prioritisation models.

    In particular, patient arrivals and treatment times are currently simulated rather than being driven by a real-time hospital system.

    The simulation variables should therefore be tested across a reasonable range rather than relying on a single arbitrary configuration.

### Current Simulation Parameters

Some parameters are currently set as defaults and have not yet been fully justified.

Important parameters include:

    arrival_rate

    service_hours

    alpha

    seed

For example:

    arrival_rate = 1 patient/hour

    service_hours = 0.75 hours

    alpha = 0.001

These values should be treated as current proof-of-concept values, not final experimental settings.

For the final comparison, it will likely be necessary to run the simulation across multiple parameter settings and/or multiple random seeds.

### Running the Queue Simulation to see current version

Random scoring is included for testing the queue independently of a trained model.

To test it out, try:

python queue_simulation.py \
    --test_file test.parquet \
    --score_mode random \
    --output_dir queue_random

Once model scores are ready, we would have for example, using DDQN:

python queue_simulation.py \
    --test_file test.parquet \
    --score_mode file \
    --score_file ddqn_scores.parquet \
    --output_dir queue_ddqn

The same process can be used for IQL, BCQ and BC.

### Queue Outputs

The simulation currently produces:

    queue_results.parquet

    queue_metrics.csv

queue_results.parquet contains patient-level results such as:

    Arrival time

    Treatment start/end

    Waiting time

    Time in system

    Priority score

    SOFA score

    Reward

queue_metrics.csv contains aggregate metrics such as:

- Number of patients

- Mean waiting time

- Median waiting time

- 90th percentile waiting time

- Maximum waiting time

- Mean time in system

- Mean priority score

- Mean SOFA

Note: The current metrics are only an initial implementation.

The metrics have not yet been examined in enough depth to determine whether they are:

- Correctly implemented

- Sufficient for comparing the models

- Appropriate for the research question

- Fair across different scoring methods

- This needs further work before using the queue simulation results as final evidence.

Potentially, we may also want metrics related to:

- Patient survival/mortality

- High-risk patient prioritisation

- Waiting time for high-risk patients

- SOFA/severity-based prioritisation

- Fairness of waiting times

- Queue utilisation

- Treatment throughput

- Potential trade-off between overall waiting time and prioritisation of high-risk patients

### Current To-Do List

Preprocessing / Models
- Retrain all models using the new train.parquet.
- Verify that the new test.parquet is compatible with all model testing code.
- Write DDQN scoring script.
- Write IQL scoring script.
- Write BCQ scoring script.
- Write BC scoring script.

Standardise all score outputs to:

- stay_id
- priority_score
- Queue Simulation
    - Test every model's score file with queue_simulation.py.
    - Fix any variable/state-column mismatches between the scoring scripts and queue simulation.
    - Check that the score direction is consistent across models.
    - Review the current arrival-rate and service-time assumptions.
    - Review the alpha waiting-time parameter.
    - Run multiple random seeds where computationally feasible.
    - Test a range of simulation parameter values.
- Metrics / Evaluation
    - Thoroughly review the current metrics.
    - Decide which metrics are actually appropriate for model comparison.
    - Add any missing patient-outcome or prioritisation metrics.
    - Determine how many simulation runs are needed for reliable comparisons.
    - Consider confidence intervals / variability across simulation runs.
    - Establish a consistent experimental setup so every model is evaluated under identical simulation conditions.
