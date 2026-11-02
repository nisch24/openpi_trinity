#!/bin/bash
#SBATCH -J arm-spacing
#SBATCH -A gts-szonouz6
#SBATCH -q inferno
#SBATCH -p gpu-rtx6000
#SBATCH --gres=gpu:rtx_6000:1         # an RTX-class GPU, the type the scoring jobs render on
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0:30:00
#SBATCH -o logs/arm-spacing-%j.out
# Does the simulator, set up as we score, start where the demos started? See trinity/check_arm_spacing.py.
# Usage (from the repo root):  sbatch trinity/check_arm_spacing.sbatch
# Look for "SPACING OK" / "SPACING DIFFERS" / "SPACING UNKNOWN" per arm at the end of the log.

source ~/.openpi_env
cd $PROJ/openpi
SCRIPT=$PROJ/openpi/trinity/check_arm_spacing.py

# How RoboTwin uses the spacing, for reference
sed -n 105,120p $PROJ/RoboTwin/envs/robot/robot.py

for ARM in aloha-agilex franka ur5; do   # aloha-agilex is the control: it has no spacing and should match
  FRAME=$PROJ/robotwin_data/first_frame_$ARM.npz
  uv run python $SCRIPT --step demo --arm $ARM --frame $FRAME --src-root $PROJ/robotwin_data || { echo "SPACING UNKNOWN: $ARM: no demo frame"; continue; }
  # A fresh process per arm, like the setup smoke test
  (cd $PROJ/RoboTwin && .venv/bin/python $SCRIPT --step sim --arm $ARM --frame $FRAME) || echo "SPACING UNKNOWN: $ARM: simulator step failed"
done
