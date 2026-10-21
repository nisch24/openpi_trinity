#!/bin/bash
#SBATCH -J aloha-setup
#SBATCH -A gts-szonouz6
#SBATCH -q embers                     # free tier: no charge, but jobs can be preempted
#SBATCH --cpus-per-task=8             # CPU only
#SBATCH --mem=32G
#SBATCH -t 3:00:00
#SBATCH -o logs/aloha-setup-%j.out
#SBATCH --requeue                     # safe to rerun: each step overwrites its own output
# One-time ALOHA sim setup (from the repo root):  sbatch trinity/setup_aloha.sbatch
#   1. builds the simulator's client venv (Python 3.10, gym-aloha, MuJoCo)
#   2. downloads lerobot/aloha_sim_transfer_cube_human and computes norm stats for the 4 variants

source ~/.openpi_env
cd $PROJ/openpi
set -e

# 1. Client venv for the simulator (separate from openpi's own env, like LIBERO's)
rm -rf examples/aloha_sim/.venv
uv venv --python 3.10 examples/aloha_sim/.venv
VIRTUAL_ENV=$PWD/examples/aloha_sim/.venv uv pip install -r examples/aloha_sim/requirements.txt -e packages/openpi-client
echo "ALOHA client venv ready."

# 2. Norm stats: the 4 configs share data and transforms, so compute once and copy
uv run scripts/compute_norm_stats.py --config-name pi05_aloha_sim_lora
SRC=assets/pi05_aloha_sim_lora/lerobot
for C in pi05_aloha_sim_lora_strict pi05_aloha_sim_lora_vision pi05_aloha_sim_lora_vision_r128; do
  mkdir -p assets/$C && cp -r $SRC assets/$C/
done
echo "Done. Norm stats in assets/pi05_aloha_sim_lora*/lerobot/aloha_sim_transfer_cube_human/"
