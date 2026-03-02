#!/bin/bash
#SBATCH --job-name=iql_sweep_v3
#SBATCH --partition=tue.gpu2.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/Honors/logs/%x_%j.out
#SBATCH --error=/home/%u/Honors/logs/%x_%j.err

set -euo pipefail

echo "=== Job info ==="
echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

echo "=== GPU ==="
nvidia-smi

echo "=== Paths ==="
echo "PWD before cd: $(pwd)"

# HPC SBatch code
# Activate your venv
source /home/20231942/Honors/venvs/iql/bin/activate

echo "Python: $(which python)"
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_build', torch.version.cuda); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# ----- Go to Version 3
cd /home/%u/Honors/iql/Version\ 3
echo "PWD after cd: $(pwd)"
ls -la

# Performance / stability env metrics
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Run sweep
# Matches run_sweep.py CLI: --data --out_root --device --seeds --epochs --batch_size --val_frac --unsupported_thresh --gamma --fqe_steps
python Analysis/run_sweep.py \
  --data Processed/sepsis_lite_transitions_4h.parquet \
  --out_root experiments \
  --device cuda \
  --seeds 0,1,2,3,4 \
  --epochs 30 \
  --batch_size 1024 \
  --val_frac 0.1 \
  --unsupported_thresh 1e-3 \
  --gamma 0.99 \
  --fqe_steps 10000

echo "=== Done ==="
date