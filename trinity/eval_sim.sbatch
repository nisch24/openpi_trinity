#!/bin/bash
#SBATCH -J sim-eval
#SBATCH -A gts-szonouz6
#SBATCH -q inferno
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:a100:1             # the policy server and the simulator share this GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o logs/eval-%j.out           # same name pattern as the other evals, so collect_results.py reads it
# Success-rate evaluation for robocasa, gr1 or bigym: 5 tasks x 20 episodes. Usage (from the repo root):
#   ROBOT=gr1 CONFIG=pi05_gr1_lora CKPT=checkpoints/pi05_gr1_lora/run1/29999 sbatch trinity/eval_sim.sbatch
# Needs that robot's setup job once (trinity/setup_robocasa.sbatch or setup_bigym.sbatch).

source ~/.openpi_env
cd $PROJ/openpi
ROBOT=${ROBOT:?set ROBOT to robocasa, gr1 or bigym}
CONFIG=${CONFIG:?set CONFIG, e.g. pi05_gr1_lora}
CKPT=${CKPT:?set CKPT to a checkpoint step dir}
TRIALS=${TRIALS:-20}                  # episodes per task, each from a different fixed starting scene

# Header in LIBERO's format so trinity/collect_results.py lists this run (suite = the robot)
echo "Evaluating $CONFIG @ $CKPT on $ROBOT ($TRIALS trials/task), node $(hostname)"

# Free port per job (shared GPU nodes), then start the policy server in the background
PORT=${PORT:-$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')}
echo "Using port $PORT"
uv run scripts/serve_policy.py --port $PORT policy:checkpoint --policy.config=$CONFIG --policy.dir=$CKPT &
SERVER=$!
until grep -q "server listening" logs/eval-$SLURM_JOB_ID.out 2>/dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "Server exited before starting"; exit 1; }
  sleep 10
done

# Run the robot's simulator against it, rendering without a display
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
PY=examples/$ROBOT/.venv/bin/python
OUT=logs/videos/$SLURM_JOB_ID
if [ "$ROBOT" = bigym ]; then
  $PY trinity/eval_bigym.py --port $PORT --episodes-per-task $TRIALS --src-root $PROJ/bigym_data --video-out-path $OUT
else
  $PY trinity/eval_robocasa.py --robot $ROBOT --port $PORT --episodes-per-task $TRIALS --video-out-path $OUT
fi
kill $SERVER
