#!/bin/bash
#SBATCH -J rby1-data
#SBATCH -A gts-szonouz6
#SBATCH -q embers                     # free tier: no charge, but jobs can be preempted
#SBATCH --cpus-per-task=16            # CPU-only: video decoding + image writing, no GPU needed
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o logs/rby1-data-%j.out
#SBATCH --requeue                     # if preempted, rerun from scratch (conversion overwrites its output)
# Usage (from the repo root):  sbatch trinity/convert_rby1.sbatch
# Quick check first:           MAX_EPISODES=5 sbatch trinity/convert_rby1.sbatch
# Steps: download Rainbow's v3.0 dataset -> convert to openpi's LeRobot v2.1 format -> norm stats per config.
# Writes two datasets: trinity/<name> (training) and trinity/<name>_heldout (every 20th episode, for
# trinity/eval_rby1.py). Norm stats come from the training set only.

source ~/.openpi_env                  # sets PROJ and HF_TOKEN (download is rate-limited without a token)
cd $PROJ/openpi
set -e                                # stop at the first failing step

REPO=${REPO:-rainbowrobotics/simtos_0412_mirror_v2}
ARGS="--repo-id $REPO"
[ -n "$MAX_EPISODES" ] && ARGS="$ARGS --max-episodes $MAX_EPISODES"

echo "Converting $REPO on $(hostname): $ARGS"
uv run python trinity/convert_rby1.py $ARGS

# Each config keeps its own copy of the stats (assets/<config>/trinity/<name>/norm_stats.json).
# All RB-Y1 configs share data and transforms, so compute once and copy.
# Stats come from a random sample of frames: the loader decodes every image (~7 s per 32 frames),
# so all ~205k frames would take ~12 h, while 12,800 random frames take ~45 min and give the same stats.
STATS_FRAMES=${STATS_FRAMES:-12800}
uv run scripts/compute_norm_stats.py --config-name pi05_rby1_lora --max-frames $STATS_FRAMES
SRC=assets/pi05_rby1_lora/trinity/${REPO#*/}
for C in pi05_rby1 pi05_rby1_lora_strict pi05_rby1_lora_vision pi05_rby1_lora_vision_r64 pi05_rby1_lora_vision_r128; do
  mkdir -p assets/$C/trinity && cp -r $SRC assets/$C/trinity/
done
echo "Done. Norm stats in assets/pi05_rby1*/trinity/${REPO#*/}/"
