#!/bin/bash
# Submit a Trinity job on any fast GPU, skipping V100 and RTX 6000 nodes.
# Those two lack native bfloat16, the number format openpi runs in, so they are much slower.
# Allowed in practice: A100, H100, H200, L40S, RTX PRO 6000 Blackwell.
#
# Usage (from the repo root; settings pass through as environment variables):
#   TRIALS=2 bash trinity/submit.sh eval
#   CONFIG=pi05_libero_lora bash trinity/submit.sh train
set -euo pipefail
JOB=${1:?usage: bash trinity/submit.sh eval|train}

# Ask Slurm, at submit time, which nodes carry the slow GPU types, and exclude them.
SLOW=$(sinfo -N -h -o "%N %G" | awk '$2 ~ /gpu:(v100|rtx_6000):/ {print $1}' | sort -u | paste -sd, -)
EXCLUDE=${SLOW:+--exclude=$SLOW}

mkdir -p logs
sbatch $EXCLUDE trinity/$JOB.sbatch
