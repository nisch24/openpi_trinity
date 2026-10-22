#!/bin/bash
#SBATCH -J robotwin-data
#SBATCH -A gts-szonouz6
#SBATCH -q embers                     # free tier: no charge, but jobs can be preempted
#SBATCH --cpus-per-task=16            # CPU-only: JPEG decoding + image writing, no GPU needed
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o logs/robotwin-data-%j.out
#SBATCH --requeue                     # if preempted, rerun: downloads are kept, conversion overwrites its output
# Usage (from the repo root), one job per arm:
#   for A in aloha-agilex franka ur5; do ARM=$A sbatch trinity/convert_robotwin.sbatch; done
# Quick check first:  ARM=franka MAX_EPISODES=3 sbatch trinity/convert_robotwin.sbatch
# Steps: download the 5 tasks' clean_50 zips -> convert to LeRobot v2.1 (trinity/robotwin_<arm>)
#        -> norm stats for that arm's 4 configs (A, B, C r16, C r128).

source ~/.openpi_env                  # sets PROJ and HF_TOKEN
cd $PROJ/openpi
set -e                                # stop at the first failing step

ARM=${ARM:?set ARM to aloha-agilex, franka or ur5}
SHORT=${ARM%%-*}                      # config name uses "aloha" for aloha-agilex
ARGS="--arm $ARM --src-root $PROJ/robotwin_data"
[ -n "$MAX_EPISODES" ] && ARGS="$ARGS --max-episodes $MAX_EPISODES"

echo "Converting RoboTwin $ARM on $(hostname): $ARGS"
uv run python trinity/convert_robotwin.py $ARGS

# The 4 variants share data and transforms, so compute stats once and copy
uv run scripts/compute_norm_stats.py --config-name pi05_robotwin_${SHORT}_lora
SRC=assets/pi05_robotwin_${SHORT}_lora/trinity
for S in _strict _vision _vision_r128; do
  mkdir -p assets/pi05_robotwin_${SHORT}_lora$S && cp -r $SRC assets/pi05_robotwin_${SHORT}_lora$S/
done
echo "Done. Norm stats in assets/pi05_robotwin_${SHORT}_lora*/trinity/robotwin_$ARM/"
