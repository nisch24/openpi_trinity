#!/bin/bash
#SBATCH -J robotwin-eval
#SBATCH -A gts-szonouz6
#SBATCH -q inferno
#SBATCH -p gpu-l40s
#SBATCH --gres=gpu:l40s:1             # an RTX-class GPU: SAPIEN renders with Vulkan (A100/H100 have no RT cores)
#SBATCH --cpus-per-task=4             # L40S nodes allow at most 4 CPUs per GPU
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o logs/eval-%j.out           # same name pattern as LIBERO evals, so collect_results.py reads it
# Success-rate evaluation on RoboTwin 2.0, one arm, 5 tasks. Usage (from the repo root):
#   ARM=franka CONFIG=pi05_robotwin_franka_lora_vision_r128 \
#     CKPT=checkpoints/pi05_robotwin_franka_lora_vision_r128/run1/4999 sbatch trinity/eval_robotwin.sbatch
# Needs trinity/robotwin_feasibility.sbatch and trinity/setup_robotwin.sbatch run once.
# The policy server and the simulator share the one GPU (pi0.5 inference ~8 GB, SAPIEN a few GB).

source ~/.openpi_env
cd $PROJ/openpi
ARM=${ARM:?set ARM to aloha-agilex, franka or ur5}
CONFIG=${CONFIG:?set CONFIG, e.g. pi05_robotwin_franka_lora}
CKPT=${CKPT:?set CKPT to a checkpoint step dir}
TRIALS=${TRIALS:-20}                  # episodes per task

# Header in LIBERO's format so trinity/collect_results.py lists this run (suite = robotwin_<arm>)
echo "Evaluating $CONFIG @ $CKPT on robotwin_$ARM ($TRIALS trials/task), node $(hostname)"

# Fail fast on a broken GPU (atl1-1-01-004-31-0 has shown ECC errors and "no supported devices")
# (nvidia-smi still lists a faulty GPU, so ask JAX itself, the way the policy server will)
nvidia-smi --query-gpu=name,ecc.errors.uncorrected.volatile.total --format=csv
uv run python -c "import jax; print('GPU OK:', jax.devices('gpu'))" || { echo "GPU UNUSABLE on $(hostname)"; exit 1; }

# Free port per job (shared GPU nodes), then start the policy server in the background
PORT=${PORT:-$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')}
echo "Using port $PORT"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5   # leave GPU memory for the simulator
uv run scripts/serve_policy.py --port $PORT policy:checkpoint --policy.config=$CONFIG --policy.dir=$CKPT &
SERVER=$!
until grep -q "server listening" logs/eval-$SLURM_JOB_ID.out 2>/dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "Server exited before starting"; exit 1; }
  sleep 10
done

# Run RoboTwin's simulator against it (from RoboTwin's folder, in its own venv)
cd $PROJ/RoboTwin
.venv/bin/python $PROJ/openpi/trinity/eval_robotwin.py --arm $ARM --port $PORT --num-episodes $TRIALS
kill $SERVER
