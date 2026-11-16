#!/bin/bash
#SBATCH -J bigym-setup
#SBATCH -A gts-szonouz6
#SBATCH -q inferno
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:a100:1             # a GPU to render the demos' camera images
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o logs/bigym-setup-%j.out
# One-time BiGym setup (from the repo root):  sbatch trinity/setup_bigym.sbatch
# Quick test first:  MAX_DEMOS=2 sbatch trinity/setup_bigym.sbatch
# Steps: 1. simulator (repo, venv)  2. replay the human demos with cameras, keeping the ones that
#        solve the task  3. write the dataset  4. norm stats for the 2 configs

source ~/.openpi_env                  # sets PROJ and HF_TOKEN
cd $PROJ/openpi
set -e
MAX_DEMOS=${MAX_DEMOS:-50}            # successful demos kept per task
VENV=$PWD/examples/bigym/.venv
REPO=$PROJ/bigym
DATA=$PROJ/bigym_data
export MUJOCO_GL=egl

# 1. Simulator: BiGym at a pinned commit, installed in place because it finds its scene files next to its code
[ -d $REPO/.git ] || git clone https://github.com/chernyadev/bigym $REPO
git -C $REPO checkout -q d98124454dfa3bc697f75536daec5989a47bfa8d
if [ -x $VENV/bin/python ] && $VENV/bin/python -c "import bigym, demonstrations, openpi_client, quaternion" 2>/dev/null; then
  echo "BiGym venv already built; skipping."
else
  rm -rf $VENV && mkdir -p examples/bigym
  uv venv --python 3.10 $VENV
  export VIRTUAL_ENV=$VENV
  # BiGym needs its author's Gymnasium fork, whose pyproject.toml lists one extra twice under two
  # spellings; uv refuses to read that file. So clone the fork, rename the second spelling, install it.
  GYM=$PROJ/bigym_gymnasium
  [ -d $GYM/.git ] || git clone https://github.com/stepjam/Gymnasium $GYM
  git -C $GYM checkout -q -f 0.29.2
  python3 - $GYM/pyproject.toml <<'PY'
import re, sys
path = sys.argv[1]
lines, seen, section = open(path).read().splitlines(), set(), None
for i, line in enumerate(lines):
    if line.startswith("["):
        section = line.strip()
    elif section == "[project.optional-dependencies]" and (m := re.match(r"([A-Za-z0-9_.-]+)(\s*=.*)", line)):
        name = re.sub(r"[-_.]+", "-", m[1]).lower()  # the normalized name uv compares
        if name in seen:
            lines[i] = f"{m[1]}-duplicate{m[2]}"
        seen.add(name if name not in seen else name + "-duplicate")
open(path, "w").write("\n".join(lines) + "\n")
PY
  uv pip install -e $GYM
  # BiGym itself without its dependency list (it names the fork by URL), then that list minus the fork,
  # plus numpy-quaternion, which some BiGym tasks import as `quaternion` but setup.py does not list
  uv pip install -e $REPO --no-deps
  uv pip install --no-deps "mojo @ git+https://github.com/stepjam/mojo.git@0.1.1"   # in case it also names the fork
  uv pip install "numpy==1.26.*" "safetensors==0.6.2" "mujoco==3.1.5" "dm_control==1.0.19" pyquaternion \
    "numpy-quaternion<2024" mujoco_utils wget pyyaml dearpygui "pyopenxr<1.1.5001" \
    imageio imageio-ffmpeg tyro -e packages/openpi-client
  unset VIRTUAL_ENV
  echo "BiGym venv ready."
fi

# 2. Replay the demos in the simulator (BiGym's venv); prints how many solve each task
$VENV/bin/python trinity/convert_bigym.py --step dump --src-root $DATA --max-demos $MAX_DEMOS

# 3. Write the dataset (openpi's venv)
uv run python trinity/convert_bigym.py --step build --src-root $DATA

# 4. Norm stats: both configs share data and transforms, so compute once and copy
uv run scripts/compute_norm_stats.py --config-name pi05_bigym_lora
mkdir -p assets/pi05_bigym_lora_vision_r128 && cp -r assets/pi05_bigym_lora/trinity assets/pi05_bigym_lora_vision_r128/
echo "Done. Norm stats in assets/pi05_bigym_lora*/trinity/bigym_h1/"
