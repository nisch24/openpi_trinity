#!/bin/bash
#SBATCH -J aloha-eval
#SBATCH -A gts-szonouz6
#SBATCH -q embers                     # free tier: no charge, but jobs can be preempted
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -o logs/eval-%j.out           # same name pattern as LIBERO evals, so collect_results.py reads it
#SBATCH --requeue
# Success-rate evaluation on ALOHA sim (transfer cube). Usage (from the repo root):
#   CONFIG=pi05_aloha_sim_lora CKPT=checkpoints/pi05_aloha_sim_lora/run1/19999 \
#     sbatch -p gpu-v100 --gres=gpu:v100:1 trinity/eval_aloha.sbatch
# Needs the client venv once: bash trinity/setup_aloha_client.sh

source ~/.openpi_env
cd $PROJ/openpi
CONFIG=${CONFIG:?set CONFIG, e.g. pi05_aloha_sim_lora}
CKPT=${CKPT:?set CKPT to a checkpoint step dir}
TRIALS=${TRIALS:-50}                  # episodes, each from a different fixed starting scene

# Header in LIBERO's format so trinity/collect_results.py lists this run (suite = aloha_transfer_cube)
echo "Evaluating $CONFIG @ $CKPT on aloha_transfer_cube ($TRIALS trials/task), node $(hostname)"

# Free port per job (shared GPU nodes), then start the policy server in the background
PORT=${PORT:-$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')}
echo "Using port $PORT"
uv run scripts/serve_policy.py --port $PORT policy:checkpoint --policy.config=$CONFIG --policy.dir=$CKPT &
SERVER=$!
until grep -q "server listening" logs/eval-$SLURM_JOB_ID.out 2>/dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "Server exited before starting"; exit 1; }
  sleep 10
done

# Run the ALOHA simulator against it, rendering without a display
source examples/aloha_sim/.venv/bin/activate
export MUJOCO_GL=${MUJOCO_GL:-egl}   # try MUJOCO_GL=osmesa if EGL errors appear
python trinity/eval_aloha_sim.py --port $PORT --num-episodes $TRIALS --video-out-path logs/videos/$SLURM_JOB_ID
kill $SERVER
