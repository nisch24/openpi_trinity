#!/bin/bash
#SBATCH -J libero-eval
#SBATCH -A gts-szonouz6
#SBATCH -q embers                     # free tier: no charge, but jobs can be preempted
#SBATCH --gres=gpu:1                   # one GPU; submit via trinity/submit.sh to skip V100 and RTX 6000 nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o logs/eval-%j.out
#SBATCH --requeue                     # if preempted, Slurm puts the job back in the queue
# Usage (from the repo root):
#   bash trinity/submit.sh eval                                   # official pi05_libero baseline, libero_spatial
#   CONFIG=pi05_libero_lora CKPT=checkpoints/pi05_libero_lora/run1/29999 SUITE=libero_10 bash trinity/submit.sh eval

source ~/.openpi_env
cd $PROJ/openpi
CONFIG=${CONFIG:-pi05_libero}                               # training config the checkpoint was made with
CKPT=${CKPT:-gs://openpi-assets/checkpoints/pi05_libero}    # checkpoint to evaluate
SUITE=${SUITE:-libero_spatial}                              # libero_spatial | libero_object | libero_goal | libero_10
TRIALS=${TRIALS:-50}                                        # rollouts per task (10 tasks per suite)

echo "Evaluating $CONFIG @ $CKPT on $SUITE ($TRIALS trials/task), node $(hostname)"

# Start the policy server in the background on this GPU
uv run scripts/serve_policy.py --port 8000 policy:checkpoint --policy.config=$CONFIG --policy.dir=$CKPT &
SERVER=$!
# Wait until it accepts connections (first load compiles, so this can take minutes)
until (echo > /dev/tcp/localhost/8000) 2>/dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "Server exited before starting"; exit 1; }
  sleep 10
done

# Run the LIBERO simulator against it, rendering without a display
source examples/libero/.venv/bin/activate
export PYTHONPATH=${PYTHONPATH:-}:$PWD/third_party/libero
export MUJOCO_GL=${MUJOCO_GL:-egl}                          # try MUJOCO_GL=osmesa if EGL errors appear
python examples/libero/main.py --args.task-suite-name $SUITE \
  --args.num-trials-per-task $TRIALS --args.video-out-path logs/videos/$SLURM_JOB_ID
kill $SERVER
