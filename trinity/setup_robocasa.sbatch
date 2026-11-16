#!/bin/bash
#SBATCH -J robocasa-setup
#SBATCH -A gts-szonouz6
#SBATCH -q inferno
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:a100:1             # a GPU so the render check matches the eval jobs
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o logs/robocasa-setup-%j.out
# One-time setup for a RoboCasa-based robot (from the repo root), one job per robot:
#   ROBOT=robocasa sbatch trinity/setup_robocasa.sbatch    # Panda on an Omron base, kitchen tasks
#   ROBOT=gr1 sbatch trinity/setup_robocasa.sbatch         # GR-1 humanoid upper body, tabletop tasks
# Quick test first:  ROBOT=gr1 EPISODES=2 sbatch trinity/setup_robocasa.sbatch
# Steps: 1. simulator (repo, venv, assets)  2. download and convert the demos
#        3. simulator check against the demos  4. norm stats for the robot's 2 configs
# Both simulators follow NVIDIA's GR00T setup scripts (same repos, pins and package versions).

source ~/.openpi_env                  # sets PROJ and HF_TOKEN
cd $PROJ/openpi
set -e
ROBOT=${ROBOT:?set ROBOT to robocasa or gr1}
EPISODES=${EPISODES:-100}             # demos per task
VENV=$PWD/examples/$ROBOT/.venv       # each simulator gets its own venv (both install a package named robocasa)
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

# Per robot: simulator repo, pinned commit, robosuite version, asset download command
if [ "$ROBOT" = robocasa ]; then
  REPO=$PROJ/robocasa; URL=https://github.com/squarefk/robocasa; PIN=d89d481ce9c76da7f179466981676e268aa842e5
  ROBOSUITE=85abee228d1c43ab1939bce33028099945d453b4
  ASSETS="robocasa/scripts/download_kitchen_assets.py"
elif [ "$ROBOT" = gr1 ]; then
  REPO=$PROJ/robocasa-gr1-tabletop-tasks; URL=https://github.com/robocasa/robocasa-gr1-tabletop-tasks
  PIN=4840e671596f93ca03651524b9f72ffb1aadfeff
  ROBOSUITE=v1.5.1
  ASSETS="robocasa/scripts/download_tabletop_assets.py -y"
else
  echo "Unknown ROBOT=$ROBOT"; exit 1
fi

# 1. Simulator: repo at its pin, venv, assets (each part is skipped if already done)
[ -d $REPO/.git ] || git clone $URL $REPO
git -C $REPO checkout -q $PIN
if [ -x $VENV/bin/python ] && $VENV/bin/python -c "import robocasa, openpi_client, mujoco; assert mujoco.__version__ == '3.2.6'" 2>/dev/null; then
  echo "$ROBOT venv already built; skipping."
else
  rm -rf $VENV && mkdir -p examples/$ROBOT
  uv venv --python 3.12 $VENV
  export VIRTUAL_ENV=$VENV
  uv pip install setuptools wheel
  PINS="numpy==1.26.4 mujoco==3.2.6 gymnasium==0.29.1"   # both simulators check for exactly these when imported
  # robosuite's newer pins ask for mujoco>=3.3 while RoboCasa checks for 3.2.6 (GR00T overrides it the same
  # way), so install robosuite without its dependency list, then that list with MuJoCo pinned
  uv pip install --no-deps "git+https://github.com/ARISE-Initiative/robosuite.git@$ROBOSUITE"
  uv pip install $PINS numba scipy "qpsolvers[quadprog]" Pillow opencv-python pynput termcolor tqdm
  if [ "$ROBOT" = robocasa ]; then
    # This fork pins numpy and numba versions that do not exist for Python 3.12, so install it
    # without its dependency list and give it compatible versions (as GR00T's setup does)
    uv pip install -e $REPO --no-deps --config-settings editable_mode=compat
    uv pip install $PINS numba scipy pygame Pillow opencv-python pyyaml pynput tqdm termcolor h5py lxml hidapi tianshou
  else
    uv pip install -e $REPO $PINS --config-settings editable_mode=compat
  fi
  uv pip install $PINS imageio imageio-ffmpeg tyro -e packages/openpi-client
  unset VIRTUAL_ENV
  echo "$ROBOT venv ready."
fi
if [ ! -f $REPO/.trinity_assets_done ]; then
  (cd $REPO && yes y | $VENV/bin/python $ASSETS)   # answers the scripts' "proceed?" questions
  touch $REPO/.trinity_assets_done
fi

# 2. Demos: download the first $EPISODES per task and convert them (openpi's venv)
uv run python trinity/convert_groot_sim.py --robot $ROBOT --episodes $EPISODES --src-root $PROJ/groot_sim_data

# 3. Check: the simulator builds, renders, accepts actions, and starts in a state like the demos'
$VENV/bin/python trinity/eval_robocasa.py --robot $ROBOT --check \
  --reference $PROJ/groot_sim_data/${ROBOT}_reference.json || echo "SIM CHECK BROKEN: see the error above"

# 4. Norm stats: both configs share data and transforms, so compute once and copy
uv run scripts/compute_norm_stats.py --config-name pi05_${ROBOT}_lora
mkdir -p assets/pi05_${ROBOT}_lora_vision_r128 && cp -r assets/pi05_${ROBOT}_lora/trinity assets/pi05_${ROBOT}_lora_vision_r128/
echo "Done. Norm stats in assets/pi05_${ROBOT}_lora*/trinity/"
