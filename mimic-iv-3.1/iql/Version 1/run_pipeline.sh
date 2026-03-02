#!/usr/bin/env bash

#SBATCH --job-name=iql_pipe
#SBATCH --partition=tue.default.q
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/Honors/logs/iql_pipe_%j.out
#SBATCH --error=/home/%u/Honors/logs/iql_pipe_%j.err

set -euo pipefail

# Allow reusing an existing run directory by passing RUN_ID as first argument:
#   sbatch run_pipeline.sh 4732964
#RUN_ID="${1:-${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}}"

RUN_ID = "4732964"
SCRATCH_BASE="/scratch-shared/$USER"
RUN_DIR="$SCRATCH_BASE/carequeue_runs/$RUN_ID"

# --- Node-local temp to avoid NFS .nfs* cleanup errors (multiprocessing finalizers) ---
JOB_TMP="${SLURM_TMPDIR:-/tmp/$USER/iql_${SLURM_JOB_ID:-manual}}"
mkdir -p "$JOB_TMP"
export TMPDIR="$JOB_TMP"
export TEMP="$JOB_TMP"
export TMP="$JOB_TMP"

# Put common caches on node-local temp too (avoids NFS cache issues)
export XDG_CACHE_HOME="$JOB_TMP/cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export JOBLIB_TEMP_FOLDER="$JOB_TMP"
export MPLCONFIGDIR="$JOB_TMP/matplotlib"
mkdir -p "$PIP_CACHE_DIR" "$TORCH_HOME" "$MPLCONFIGDIR"

mkdir -p "$RUN_DIR"
export RUN_DIR="$RUN_DIR"

echo "RUN_ID=$RUN_ID"
echo "TMPDIR=$TMPDIR"
echo "RUN_DIR=$RUN_DIR"
echo "HOST=$(hostname)"
echo "TIME=$(date)"

REPO_IQL_DIR="/home/20231942/Honors/CareQueue/mimic-iv-3.1/iql"
cd "$REPO_IQL_DIR"

# Activate venv (must exist)
source /home/$USER/Honors/venvs/iql/bin/activate
python -c "import numpy, pandas, pyarrow, torch; print('deps ok')"
echo "Using python: $(which python)"
python --version

PROC_OUT_SOURCE="sepsis_iql_actionvec_transitions.parquet"
TRANSITIONS="$RUN_DIR/$PROC_OUT_SOURCE"

echo "=== STEP 1: processing ==="

# Skip processing if transitions already exists in RUN_DIR (prevents re-running 12h processing)
if [ -f "$TRANSITIONS" ]; then
  echo "Skipping processing: transitions already exists at $TRANSITIONS"
else
  echo "Running processing..."
  python iql_processing.py

  # Processing may write either into RUN_DIR or into the repo directory.
  if [ -f "$TRANSITIONS" ]; then
    echo "Found transitions in RUN_DIR: $TRANSITIONS"
  elif [ -f "$REPO_IQL_DIR/$PROC_OUT_SOURCE" ]; then
    echo "Found transitions in repo dir; moving to RUN_DIR"
    mv "$REPO_IQL_DIR/$PROC_OUT_SOURCE" "$TRANSITIONS"
  else
    echo "ERROR: transitions parquet not found after processing."
    echo "Looked for:"
    echo "  - $TRANSITIONS"
    echo "  - $REPO_IQL_DIR/$PROC_OUT_SOURCE"
    echo "Files in RUN_DIR:"
    ls -lah "$RUN_DIR" | tail -n 80
    echo "Files in iql dir:"
    ls -lah "$REPO_IQL_DIR" | tail -n 80
    exit 1
  fi
fi

ls -lh "$TRANSITIONS"

echo "=== STEP 2: training (CPU) ==="
python iql_training.py \
  --data "$TRANSITIONS" \
  --save_dir "$RUN_DIR" \
  --epochs 30 \
  --batch_size 256 \
  --hidden 256 \
  --val_frac 0.1 \
  --device cpu

echo "=== Copy outputs back to home (so they don't expire) ==="
HOME_OUT="/home/20231942/Honors/CareQueue/mimic-iv-3.1/iql/outputs/$RUN_ID"
mkdir -p "$HOME_OUT"
cp -r "$RUN_DIR"/* "$HOME_OUT"/

echo "DONE. Outputs copied to: $HOME_OUT"
