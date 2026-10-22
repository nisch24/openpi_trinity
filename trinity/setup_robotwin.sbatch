#!/bin/bash
#SBATCH -J robotwin-setup
#SBATCH -A gts-szonouz6
#SBATCH -q inferno
#SBATCH -p gpu-l40s
#SBATCH --gres=gpu:l40s:1             # same GPU type the evaluations run on
#SBATCH --cpus-per-task=4             # L40S nodes allow at most 4 CPUs per GPU
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -o logs/robotwin-setup-%j.out
# One-time, after trinity/robotwin_feasibility.sbatch: finishes RoboTwin's venv for evaluation.
# Usage (from the openpi repo root):  sbatch trinity/setup_robotwin.sbatch
#   1. builds CuRobo (RoboTwin's robot class always creates a CuRobo planner, even for evaluation)
#   2. adds openpi-client + tyro so trinity/eval_robotwin.py can talk to our policy server
#   3. smoke test: RoboTwin's scripted expert solves click_bell once on each of our 3 arms
# Look for "EXPERT OK" / "EXPERT FAILED" at the end of the log.

source ~/.openpi_env
set -ex
RT=$PROJ/RoboTwin
PY=$RT/.venv/bin/python
export VIRTUAL_ENV=$RT/.venv

# 1. CuRobo v0.7.8, compiled against the venv's torch (as in RoboTwin's scripts/_install.sh)
module load cuda || module avail cuda      # nvcc is needed to compile CuRobo's CUDA kernels
nvcc --version
export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"  # A100, L40S, H100
if ! $PY -c "import curobo" 2>/dev/null; then
  [ -d $RT/envs/curobo ] || git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git $RT/envs/curobo
  uv pip install --no-build-isolation -e $RT/envs/curobo
fi
uv pip install warp-lang==1.12.0 setuptools==69.5.1   # RoboTwin's pins (69.5.1 still has pkg_resources)

# 2. Our eval client's dependencies
uv pip install -e $PROJ/openpi/packages/openpi-client tyro

# 3. Smoke test: the expert must solve one episode per arm (checks planner, assets and arm configs)
cd $RT
set +ex
for ARM in aloha-agilex franka ur5; do
$PY - "$ARM" <<'EOF'
import pathlib, sys, traceback
import yaml
sys.path.insert(0, "scripts")
import eval_policy_xpolicylab as rt
arm = sys.argv[1]
try:
    cfg_dir = pathlib.Path("env_cfg/task_config")
    cfg = yaml.safe_load((cfg_dir / "demo_clean.yml").read_text())
    cfg.update(embodiment=[arm], eval_video_log=False)
    (cfg_dir / f"trinity_{arm}_demo_clean.yml").write_text(yaml.safe_dump(cfg))
    args, _ = rt.load_task_args({"task_name": "click_bell", "task_config": f"trinity_{arm}_demo_clean", "policy_name": "trinity"})
    args["eval_mode"] = True
    env = rt.class_decorator("click_bell")
    ok = False
    for seed in range(100000, 100005):  # a few seeds, in case the first scene is unstable
        try:
            env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
            env.play_once()
            ok = env.plan_success and env.check_success()
            env.close_env()
        except Exception:
            traceback.print_exc()
            rt.safe_close_env(env)
        if ok:
            break
    print(f"EXPERT {'OK' if ok else 'FAILED'}: {arm} click_bell (seed {seed})")
except Exception:
    traceback.print_exc()
    print(f"EXPERT FAILED: {arm} (setup error)")
EOF
done
$PY -c "import openpi_client, tyro, curobo; print('Client deps OK')"
echo "Done."
