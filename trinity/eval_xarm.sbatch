#!/bin/bash
#SBATCH -J xarm-eval
#SBATCH -A gts-szonouz6
#SBATCH -q inferno
#SBATCH --gres=gpu:1                   # pass --gres=gpu:a100:1 to avoid V100s (no bf16)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -o logs/eval-%j.out           # same name pattern as the other evals, so collect_results.py reads it
# Success-rate evaluation on the simulated xArm (lift a cube). Usage (from the repo root):
#   CONFIG=pi05_xarm_lora CKPT=checkpoints/pi05_xarm_lora/run1/29999 \
#     sbatch --gres=gpu:a100:1 trinity/eval_xarm.sbatch
# Needs the client venv once: sbatch trinity/setup_xarm.sbatch

source ~/.openpi_env
cd $PROJ/openpi
CONFIG=${CONFIG:?set CONFIG, e.g. pi05_xarm_lora}
CKPT=${CKPT:?set CKPT to a checkpoint step dir}
TRIALS=${TRIALS:-100}                 # episodes, each from a different fixed starting scene

# Header in LIBERO's format so trinity/collect_results.py lists this run (suite = xarm_lift)
echo "Evaluating $CONFIG @ $CKPT on xarm_lift ($TRIALS trials/task), node $(hostname)"

# Free port per job (shared GPU nodes), then start the policy server in the background
PORT=${PORT:-$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')}
echo "Using port $PORT"
uv run scripts/serve_policy.py --port $PORT policy:checkpoint --policy.config=$CONFIG --policy.dir=$CKPT &
SERVER=$!
until grep -q "server listening" logs/eval-$SLURM_JOB_ID.out 2>/dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "Server exited before starting"; exit 1; }
  sleep 10
done

# Run the xArm simulator against it, rendering without a display
source examples/xarm_sim/.venv/bin/activate
export MUJOCO_GL=${MUJOCO_GL:-egl}   # try MUJOCO_GL=osmesa if EGL errors appear
python trinity/eval_xarm_sim.py --port $PORT --num-episodes $TRIALS --video-out-path logs/videos/$SLURM_JOB_ID
kill $SERVER
