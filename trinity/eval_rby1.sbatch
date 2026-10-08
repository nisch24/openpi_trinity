#!/bin/bash
#SBATCH -J rby1-eval
#SBATCH -A gts-szonouz6
#SBATCH -q embers                     # free tier: no charge, but jobs can be preempted
#SBATCH --gres=gpu:1                   # one GPU; submit via trinity/submit.sh to skip V100 and RTX 6000 nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -o logs/rby1-eval-%j.out
#SBATCH --requeue                     # if preempted, Slurm puts the job back in the queue
# Scores a trained RB-Y1 checkpoint on the held-out episodes (no simulator; see trinity/eval_rby1.py).
# Usage (from the repo root):
#   CONFIG=pi05_rby1_lora CKPT=checkpoints/pi05_rby1_lora/run1/29999 bash trinity/submit.sh eval_rby1
# Results: one row per checkpoint in logs/rby1_results.csv

source ~/.openpi_env
cd $PROJ/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5   # inference only: leave room on shared GPUs

CONFIG=${CONFIG:-pi05_rby1_lora}
CKPT=${CKPT:?set CKPT to a checkpoint step dir, e.g. checkpoints/$CONFIG/run1/29999}
STRIDE=${STRIDE:-5}                   # score every 5th held-out frame (~2,000 frames)

echo "Evaluating $CONFIG @ $CKPT on held-out RB-Y1 episodes (stride $STRIDE)"
uv run python trinity/eval_rby1.py --config $CONFIG --checkpoint $CKPT --stride $STRIDE
