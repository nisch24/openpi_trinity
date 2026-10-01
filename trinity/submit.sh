#!/bin/bash
# Submit a Trinity job on any fast GPU, skipping V100 and RTX 6000 nodes.
# Those two lack native bfloat16, the number format openpi runs in, so they are much slower.
# Allowed: A100, H100, H200, L40S, RTX PRO 6000 Blackwell.
#
# PACE keeps each GPU type in its own Slurm partition, and a generic "--gres=gpu:1" request
# lands in the default (CPU) partition and is rejected. So this script asks Slurm which
# partitions hold the allowed GPU types and submits to all of them; Slurm uses the first free one.
#
# Usage (from the repo root; settings pass through as environment variables):
#   TRIALS=2 bash trinity/submit.sh eval
#   CONFIG=pi05_libero_lora bash trinity/submit.sh train
set -euo pipefail
JOB=${1:?usage: bash trinity/submit.sh eval|train}
FAST='gpu:(a100|h100|h200|l40s|rtx_pro_6000_blackwell):'
SLOW='gpu:(v100|rtx_6000):'

# Partitions containing at least one allowed GPU type ("*" marks the default partition; drop it)
PARTS=$(sinfo -h -o "%P %G" | awk -v re="$FAST" '$2 ~ re {sub(/\*$/, "", $1); print $1}' | sort -u | paste -sd, -)
# Individual nodes with slow GPUs, in case a partition mixes types
SLOWNODES=$(sinfo -N -h -o "%N %G" | awk -v re="$SLOW" '$2 ~ re {print $1}' | sort -u | paste -sd, -)

if [ -z "$PARTS" ]; then
  echo "No partition with an allowed GPU type was found. Paste the output of:  sinfo -h -o \"%P %G\" | sort -u"
  exit 1
fi

mkdir -p logs
ARGS="-p $PARTS"
[ -n "$SLOWNODES" ] && ARGS="$ARGS --exclude=$SLOWNODES"
echo "sbatch $ARGS trinity/$JOB.sbatch"
sbatch $ARGS trinity/$JOB.sbatch
