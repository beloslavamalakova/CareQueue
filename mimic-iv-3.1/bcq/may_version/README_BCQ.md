# BCQ Version May

This folder contains a discrete-action Batch-Constrained Q-Learning pipeline for offline RL on MIMIC-IV ICU transitions.

## Files

- `bcq_processing.py` builds the transition dataset.
- `bcq_training.py` trains the BCQ model.
- `eval_bcq_metrics.py` computes patient-level and policy-level offline metrics.
- `bcq_plots.py` plots training curves.
- `run_sweep.py` runs hyperparameter sweeps.

## Example usage

```bash
python bcq_processing.py --base /path/to/mimic-iv-3.1 --out Processed/transitions.parquet
python bcq_training.py --data Processed/transitions.parquet --save_dir Output --epochs 30
python eval_bcq_metrics.py --data Processed/transitions.parquet --checkpoint Output/best.pt --schema Output/schema_and_norm.json --outdir Evaluation
python bcq_plots.py --metrics Output/metrics.csv
```

## Notes

The action space has 25 actions, created from 5 vasopressor bins and 5 fluid bins:

```text
action = vaso_bin * 5 + fluid_bin
```

The BCQ policy chooses the best Q-action among actions that are sufficiently likely under the learned behavior policy.
