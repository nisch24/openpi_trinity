#!/bin/bash
# Submit one stage for all four new robots at once (xArm, RoboCasa Panda, BiGym H1, GR-1 tabletop).
# Run from the repo root on PACE, one stage at a time, each after the previous has finished:
#   bash trinity/run_all.sh setup     # 4 jobs: simulators, demos, checks, norm stats
#   bash trinity/run_all.sh train     # 8 jobs: A and C r128 for each robot, 30k steps
#   bash trinity/run_all.sh eval      # 8 jobs: success rate of each trained model
# Fewer robots:  ROBOTS="xarm gr1" bash trinity/run_all.sh train
# A training job that hits its time limit continues from its last checkpoint when "train" is run again.
set -euo pipefail
STAGE=${1:?usage: bash trinity/run_all.sh setup|train|eval}
ROBOTS=${ROBOTS:-"xarm robocasa bigym gr1"}
GPU=${GPU:-a100}                      # PACE keeps each GPU type in a partition named gpu-<type>
TIME=${TIME:-20:00:00}                # training time limit per submission
EXP=${EXP:-run1}
STEP=${STEP:-29999}                   # checkpoint to score (the last step of a 30k run)
SBATCH="sbatch -q inferno -p gpu-$GPU --gres=gpu:$GPU:1"   # paid queue, one GPU of the chosen type
mkdir -p logs

for ROBOT in $ROBOTS; do
  case $STAGE in
    setup)
      case $ROBOT in
        xarm) $SBATCH trinity/setup_xarm.sbatch ;;
        bigym) $SBATCH trinity/setup_bigym.sbatch ;;
        robocasa | gr1) ROBOT=$ROBOT $SBATCH trinity/setup_robocasa.sbatch ;;
        *) echo "Unknown robot $ROBOT"; exit 1 ;;
      esac ;;
    train)
      for SUFFIX in "" _vision_r128; do   # A (SigLIP fully trained) and C r128 (SigLIP adapter)
        CONFIG=pi05_${ROBOT}_lora$SUFFIX EXP=$EXP $SBATCH -t $TIME trinity/train.sbatch
      done ;;
    eval)
      for SUFFIX in "" _vision_r128; do
        CONFIG=pi05_${ROBOT}_lora$SUFFIX
        CKPT=checkpoints/$CONFIG/$EXP/$STEP
        if [ ! -d $CKPT ]; then echo "Skipping $CONFIG: $CKPT does not exist yet"; continue; fi
        if [ $ROBOT = xarm ]; then
          CONFIG=$CONFIG CKPT=$CKPT $SBATCH trinity/eval_xarm.sbatch
        else
          ROBOT=$ROBOT CONFIG=$CONFIG CKPT=$CKPT $SBATCH trinity/eval_sim.sbatch
        fi
      done ;;
    *) echo "Unknown stage $STAGE"; exit 1 ;;
  esac
done
