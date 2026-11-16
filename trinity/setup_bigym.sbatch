#!/bin/bash
#SBATCH -J bigym-setup
#SBATCH -A gts-szonouz6
#SBATCH -q inferno
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:a100:1             # a GPU to render the demos' camera images
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o logs/bigym-setup-%j.out
# One-time BiGym setup (from the repo root):  sbatch trinity/setup_bigym.sbatch
# Quick test first:  MAX_DEMOS=2 sbatch trinity/setup_bigym.sbatch
# Steps: 1. simulator (repo, venv)  2. replay the human demos with cameras, keeping the ones that
#        solve the task  3. write the dataset  4. norm stats for the 2 configs

source ~/.openpi_env                  # sets PROJ and HF_TOKEN
cd $PROJ/openpi
set -e
MAX_DEMOS=${MAX_DEMOS:-50}            # successful demos kept per task
VENV=$PWD/examples/bigym/.venv
REPO=$PROJ/bigym
DATA=$PROJ/bigym_data
export MUJOCO_GL=egl

# 1. Simulator: BiGym at a pinned commit, installed in place because it finds its scene files next to its code
[ -d $REPO/.git ] || git clone https://github.com/chernyadev/bigym $REPO
git -C $REPO checkout -q d98124454dfa3bc697f75536daec5989a47bfa8d
if [ -x $VENV/bin/python ] && $VENV/bin/python -c "import bigym, demonstrations, openpi_client" 2>/dev/null; then
  echo "BiGym venv already built; skipping."
else
  rm -rf $VENV && mkdir -p examples/bigym
  uv venv --python 3.10 $VENV
  VIRTUAL_ENV=$VENV uv pip install -e $REPO imageio imageio-ffmpeg tyro -e packages/openpi-client
  echo "BiGym venv ready."
fi

# 2. Replay the demos in the simulator (BiGym's venv); prints how many solve each task
$VENV/bin/python trinity/convert_bigym.py --step dump --src-root $DATA --max-demos $MAX_DEMOS

# 3. Write the dataset (openpi's venv)
uv run python trinity/convert_bigym.py --step build --src-root $DATA

# 4. Norm stats: both configs share data and transforms, so compute once and copy
uv run scripts/compute_norm_stats.py --config-name pi05_bigym_lora
mkdir -p assets/pi05_bigym_lora_vision_r128 && cp -r assets/pi05_bigym_lora/trinity assets/pi05_bigym_lora_vision_r128/
echo "Done. Norm stats in assets/pi05_bigym_lora*/trinity/bigym_h1/"
