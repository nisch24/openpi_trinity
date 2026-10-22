#!/bin/bash
#SBATCH -J robotwin-check
#SBATCH -A gts-szonouz6
#SBATCH -q inferno
#SBATCH -p gpu-l40s
#SBATCH --gres=gpu:l40s:1             # an RTX-class GPU: SAPIEN renders with Vulkan ray tracing (A100/H100 have no RT cores)
#SBATCH --cpus-per-task=4             # L40S nodes allow at most 4 CPUs per GPU
#SBATCH --mem=32G
#SBATCH -t 1:30:00
#SBATCH -o logs/robotwin-check-%j.out
# One-time check that RoboTwin 2.0's simulator (SAPIEN, Vulkan) runs headless on a PACE GPU node.
# Usage (from the openpi repo root):  sbatch trinity/robotwin_feasibility.sbatch
# Installs RoboTwin next to the repo ($PROJ/RoboTwin, its own Python 3.10 venv), downloads its
# assets, then renders a test scene. Look for "RENDER OK" / "RENDER FAILED" at the end of the log.

source ~/.openpi_env
set -x
RT=$PROJ/RoboTwin

# 1. Code (no submodules yet: XPolicyLab is only needed later for data conversion and eval)
[ -d $RT ] || git clone --depth 1 https://github.com/RoboTwin-Platform/RoboTwin.git $RT
cd $RT

# 2. Python env with RoboTwin's requirements (pytorch3d and CuRobo skipped: only needed for data collection)
[ -x .venv/bin/python ] || uv venv --python 3.10 .venv
VIRTUAL_ENV=$RT/.venv uv pip install -r scripts/requirements.txt
PY=$RT/.venv/bin/python
# RoboTwin's two upstream source patches (from scripts/_install.sh)
SAPIEN_DIR=$($PY -c "import sapien, os; print(os.path.dirname(sapien.__file__))")
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' $SAPIEN_DIR/wrapper/urdf_loader.py
MPLIB_DIR=$($PY -c "import mplib, os; print(os.path.dirname(mplib.__file__))")
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $MPLIB_DIR/planner.py

# 3. Assets (robots, objects, textures) from Hugging Face; skipped if already unpacked
if [ ! -d assets/embodiments ]; then
  (cd assets && $PY _download.py && for z in background_texture embodiments objects; do unzip -q -o $z.zip && rm -f $z.zip; done)
  $PY scripts/update_embodiment_config_path.py
fi
ls assets

# 4. What Vulkan support does this node have? (no sudo on PACE, so it must come with the NVIDIA driver)
nvidia-smi --query-gpu=name,driver_version --format=csv
ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>/dev/null
ldconfig -p | grep -i libvulkan || echo "no system libvulkan loader"

# 5. Render test: build a scene with a camera, take one picture, save it to logs/ to look at
set +x
$PY - <<'EOF'
import traceback
try:
    import numpy as np, sapien
    from PIL import Image
    scene = sapien.Scene()                       # default (rasterized) renderer, as RoboTwin uses for observations
    scene.add_ground(0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.8, 0.8, 0.8])
    builder = scene.create_actor_builder()
    builder.add_box_visual(half_size=[0.05, 0.05, 0.05], material=[0.8, 0.3, 0.1])
    builder.build_static().set_pose(sapien.Pose([0, 0, 0.05]))
    cam = scene.add_camera("cam", 320, 240, 1.0, 0.01, 10)
    cam.set_pose(sapien.Pose([-0.5, 0, 0.3], [0.97, 0, 0.24, 0]))
    scene.update_render(); cam.take_picture()
    rgb = (cam.get_picture("Color")[..., :3] * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(rgb).save("logs_render_test.png")
    print(f"RENDER OK: image {rgb.shape}, mean pixel {rgb.mean():.1f} (0 = black, i.e. not really rendering)")
except Exception:
    traceback.print_exc()
    print("RENDER FAILED")
EOF
# RoboTwin's own render check (ray-traced shader); prints "Render Well" or "Render Error"
$PY scripts/test_render.py
mkdir -p $PROJ/openpi/logs && mv -f logs_render_test.png $PROJ/openpi/logs/robotwin_render_test.png 2>/dev/null
echo "Done."
