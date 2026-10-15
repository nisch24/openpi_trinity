#!/bin/bash
#SBATCH -J lora-train
#SBATCH -A gts-szonouz6
#SBATCH -q embers                     # free tier: no charge, but jobs can be preempted
#SBATCH --gres=gpu:1                   # one GPU; submit via trinity/submit.sh to skip V100 and RTX 6000 nodes
#SBATCH --cpus-per-task=8             # PACE caps GPU nodes at 8 CPUs per GPU
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH -o logs/train-%j.out
#SBATCH --requeue                     # if preempted, Slurm puts the job back in the queue
# Usage (from the repo root):  CONFIG=pi05_libero_lora bash trinity/submit.sh train
# Resubmitting the same CONFIG/EXP continues from the last saved checkpoint.
# submit.sh skips v100 and rtx_6000 (no native bf16; rtx_6000 also has only 24 GB).
# To force one type instead:  sbatch --gres=gpu:h200:1 trinity/train.sbatch

source ~/.openpi_env
cd $PROJ/openpi
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.9   # training: let JAX reserve 90% of GPU memory
export PYTHONUNBUFFERED=1             # print the "Step N: loss=..." lines as they happen (otherwise buffered, lost on timeout)

CONFIG=${CONFIG:-pi05_libero_lora}
EXP=${EXP:-run1}
STEPS=${STEPS:-}                     # optional override, e.g. STEPS=300 for a timing run
WORKERS=${WORKERS:-6}                # data-loader processes; openpi's default of 2 can leave the GPU waiting on image decoding

# Weights & Biases is off: its "must resume" step can fail after an embers preemption, which would
# break automatic requeue. Training loss is still printed to the log every 100 steps.
ARGS="--exp-name=$EXP --no-wandb-enabled --num-workers=$WORKERS"
[ -n "$STEPS" ] && ARGS="$ARGS --num-train-steps=$STEPS"
# Start fresh the first time; continue from the last checkpoint on resubmission
if [ -d checkpoints/$CONFIG/$EXP ]; then ARGS="$ARGS --resume"; fi

echo "Training $CONFIG ($EXP) on $(hostname): $ARGS"
uv run scripts/train.py $CONFIG $ARGS
