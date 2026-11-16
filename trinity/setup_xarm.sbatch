#!/bin/bash
#SBATCH -J xarm-setup
#SBATCH -A gts-szonouz6
#SBATCH -q inferno
#SBATCH --gres=gpu:1                  # one GPU so the EGL render check matches the eval jobs
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o logs/xarm-setup-%j.out
# One-time xArm setup (from the repo root):  sbatch trinity/setup_xarm.sbatch
#   1. builds the simulator's client venv (Python 3.10, gym-xarm, MuJoCo)
#   2. checks that a lift can count as success, and how good the demos are (trinity/check_xarm.py)
#   3. downloads lerobot/xarm_lift_medium and computes norm stats for the 2 configs

source ~/.openpi_env
cd $PROJ/openpi
set -e
VENV=examples/xarm_sim/.venv

# 1. Client venv for the simulator (separate from openpi's own env, like ALOHA's)
if [ -x $VENV/bin/python ] && $VENV/bin/python -c "import gym_xarm, openpi_client, gymnasium; assert gymnasium.__version__ == '0.29.1'" 2>/dev/null; then
  echo "xArm client venv already built; skipping."
else
  rm -rf $VENV && mkdir -p examples/xarm_sim
  uv venv --python 3.10 $VENV
  # gymnasium 1.x changed its MuJoCo renderer's arguments, which gym-xarm 0.1.1 still uses the old way
  VIRTUAL_ENV=$PWD/$VENV uv pip install "gym-xarm==0.1.1" "gymnasium==0.29.1" "gymnasium-robotics==1.2.4" "numpy<2" imageio imageio-ffmpeg tyro -e packages/openpi-client
  echo "xArm client venv ready."
fi

# 2. Checks: can success happen at all, and how good are the demos?
MUJOCO_GL=egl $VENV/bin/python trinity/check_xarm.py --step sim || echo "SUCCESS CHECK UNKNOWN: the simulator step failed"
uv run python trinity/check_xarm.py --step data || echo "DATA CHECK FAILED: see the error above"

# 3. Norm stats: both configs share data and transforms, so compute once and copy
uv run scripts/compute_norm_stats.py --config-name pi05_xarm_lora
mkdir -p assets/pi05_xarm_lora_vision_r128 && cp -r assets/pi05_xarm_lora/lerobot assets/pi05_xarm_lora_vision_r128/
echo "Done. Norm stats in assets/pi05_xarm_lora*/lerobot/xarm_lift_medium/"
