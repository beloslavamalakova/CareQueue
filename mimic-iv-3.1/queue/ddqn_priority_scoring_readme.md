# DDQN Queue Scoring

`ddqn_priority_scoring.py` creates a queue-compatible parquet file from a
trained DDQN checkpoint. It writes one row per ICU stay with:

```text
stay_id
priority_score
predicted_value
predicted_action
```

`priority_score` is the DDQN estimate `max_a Q(s, a)` at the first observed
state for each stay. Higher Q-values therefore receive higher queue priority,
matching the queue simulator's scoring rule. The score is intentionally not
converted to the existing `p_survival` metric because that conversion assumes
a particular reward calibration and is not part of the checkpoint contract.

## Workflow

The shared queue preprocessing creates `train.parquet` and `test.parquet` for
all models. The DDQN model must be retrained on the shared `train.parquet` for
a fair comparison with the retrained BC model. Scoring itself is inference
only and does not retrain or modify the parquet files.

Example after training DDQN on the shared data:

```powershell
python .\mimic-iv-3.1\queue\ddqn_priority_scoring.py `
  --test_file .\mimic-iv-3.1\queue\test.parquet `
  --checkpoint .\mimic-iv-3.1\ddqn\outputs\ddqn_model_best.pt `
  --schema .\mimic-iv-3.1\ddqn\outputs\schema_and_norm.json `
  --output_file .\mimic-iv-3.1\queue\ddqn_scores.parquet `
  --device cpu
```

Then run the model-agnostic simulator:

```powershell
python .\mimic-iv-3.1\queue\queue_simulation.py `
  --test_file .\mimic-iv-3.1\queue\test.parquet `
  --score_mode file `
  --score_file .\mimic-iv-3.1\queue\ddqn_scores.parquet `
  --output_dir .\mimic-iv-3.1\queue\queue_ddqn
```