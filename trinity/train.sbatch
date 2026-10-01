#!/bin/bash
#SBATCH -J lora-train
#SBATCH -A gts-szonouz6
#SBATCH -q embers                     # free tier: no charge, but jobs can be preempted
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH -o logs/train-%j.out
#SBATCH --requeue                     # if preempted, Slurm puts the job back in the queue
# Usage (from the repo root):  CONFIG=pi05_libero_lora sbatch trinity/train.sbatch
# Resubmitting the same CONFIG/EXP continues from the last saved checkpoint.

source ~/.openpi_env
cd $PROJ/openpi
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.9   # training: let JAX reserve 90% of GPU memory
export WANDB_MODE=offline                                                      # no Weights & Biases account needed

CONFIG=${CONFIG:-pi05_libero_lora}
EXP=${EXP:-run1}
STEPS=${STEPS:-}                     # optional override, e.g. STEPS=300 for a timing run

ARGS="--exp-name=$EXP"
[ -n "$STEPS" ] && ARGS="$ARGS --num-train-steps=$STEPS"
# Start fresh the first time; continue from the last checkpoint on resubmission
if [ -d checkpoints/$CONFIG/$EXP ]; then ARGS="$ARGS --resume"; fi

echo "Training $CONFIG ($EXP) on $(hostname): $ARGS"
uv run scripts/train.py $CONFIG $ARGS
