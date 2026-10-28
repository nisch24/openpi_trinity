"""Success-rate evaluation on RoboTwin 2.0 for one arm, the RoboTwin counterpart to LIBERO's eval.

Connects to a running policy server (scripts/serve_policy.py) and, for each task, runs N episodes
in RoboTwin's simulator. Follows RoboTwin's own protocol (scripts/eval_policy_xpolicylab.py):
  - seeds start at 100000 * (1 + seed); a seed is used only if RoboTwin's scripted expert solves it
  - the instruction comes from RoboTwin's generator, as in its leaderboard runs
  - success = the task's own check (task_env.eval_success), within the task's step limit
Prints per-task results, then "Total episodes" / "Total success rate" in examples/libero/main.py's
format, so trinity/collect_results.py picks the result up.

Runs in RoboTwin's venv with RoboTwin as the working directory. Usually launched by trinity/eval_robotwin.sbatch.
"""
import dataclasses
import functools
import inspect
import logging
import pathlib
import sys

import numpy as np
from openpi_client import websocket_client_policy
import tyro
import websockets.sync.client
import yaml

# RoboTwin's venv has websockets 15+, whose sync client pings every 20 s and drops the connection if no
# pong comes back in 20 s. openpi's server can't answer while it runs inference, and the first call
# (JIT compile) takes longer than that, so turn client keepalive off (older versions have no such option).
if "ping_interval" in inspect.signature(websockets.sync.client.connect).parameters:
    websockets.sync.client.connect = functools.partial(websockets.sync.client.connect, ping_interval=None)

ROBOTWIN = pathlib.Path.cwd()  # the job cds into RoboTwin
sys.path.insert(0, str(ROBOTWIN / "scripts"))
import eval_policy_xpolicylab as rt  # noqa: E402  (RoboTwin's eval helpers; also puts envs/ on the path)

TASKS = ("beat_block_hammer", "click_bell", "move_can_pot", "place_empty_cup", "press_stapler")
# Dataset arm name (as in <arm>_clean_50.zip and --arm) -> RoboTwin "embodiment" list. Two-arm robots are
# [name]; single-arm robots are paired as [left, right, spacing in m]. 0.8 m matches the demonstrations
# (left and right end-effectors start 0.80 m apart in the franka and ur5 clean_50 data).
SIM_ARM = {
    "aloha-agilex": ["aloha-agilex"],
    "arx-x5": ["ARX-X5"],
    "franka": ["franka-panda", "franka-panda", 0.8],
    "ur5": ["ur5-wsg", "ur5-wsg", 0.8],
}
CAMERAS = {"head_camera": "observation/image", "left_camera": "observation/left_wrist_image",
           "right_camera": "observation/right_wrist_image"}


@dataclasses.dataclass
class Args:
    arm: str = "aloha-agilex"  # RoboTwin embodiment the model was trained on
    tasks: tuple[str, ...] = TASKS
    port: int = 8000
    host: str = "0.0.0.0"
    num_episodes: int = 20  # per task
    replan_steps: int = 10  # actions executed from each 15-step chunk before asking again
    seed: int = 0
    base_config: str = "demo_clean"  # RoboTwin "Easy" setting (clean table, fixed lighting)


def write_task_config(arm: str, base: str) -> str:
    """RoboTwin picks the arm from a task-config yml: copy the base one with our arm, no video logging."""
    cfg_dir = ROBOTWIN / "env_cfg" / "task_config"
    cfg = yaml.safe_load((cfg_dir / f"{base}.yml").read_text())
    cfg["embodiment"] = SIM_ARM.get(arm, [arm])
    cfg["eval_video_log"] = False  # RoboTwin writes videos through an ffmpeg binary PACE doesn't have
    name = f"trinity_{arm}_{base}"
    (cfg_dir / f"{name}.yml").write_text(yaml.safe_dump(cfg))
    return name


def to_observation(obs: dict, prompt: str) -> dict:
    """RoboTwin observation -> the keys the pi05_robotwin_* configs expect (same as the converted data)."""
    element = {key: np.asarray(obs["observation"][cam]["rgb"]) for cam, key in CAMERAS.items()}
    element["observation/state"] = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
    element["prompt"] = prompt
    return element


def run_task(task: str, task_config: str, client, args: Args) -> int:
    """Run args.num_episodes episodes of one task; return the number of successes."""
    cfg, _ = rt.load_task_args({"task_name": task, "task_config": task_config, "policy_name": "trinity"})
    cfg["eval_mode"] = True
    instruction_type = cfg["eval_instruction"]
    env = rt.class_decorator(task)
    seed, done, successes = 100000 * (1 + args.seed), 0, 0
    while done < args.num_episodes:
        # Keep only seeds RoboTwin's scripted expert can solve (its standard protocol)
        try:
            env.setup_demo(now_ep_num=done, seed=seed, is_test=True, **cfg)
            info = env.play_once()
            env.close_env()
        except Exception as e:  # unstable scene, or the expert itself errors (e.g. no grasp pose): skip the seed, as RoboTwin does
            logging.info(f"{task} seed {seed} skipped in expert check: {type(e).__name__}: {e}")
            rt.safe_close_env(env)
            seed += 1
            continue
        if not (env.plan_success and env.check_success()):
            seed += 1
            continue

        try:
            env.setup_demo(now_ep_num=done, seed=seed, is_test=True, **cfg)
        except Exception as e:  # scene setup failed this time: skip the seed, as RoboTwin does
            logging.info(f"{task} seed {seed} skipped in setup: {type(e).__name__}: {e}")
            rt.safe_close_env(env)
            seed += 1
            continue
        prompt = rt.build_instruction(cfg, info, instruction_type, args.num_episodes, seed)
        env.set_instruction(instruction=prompt)
        success = False
        while not rt.is_episode_end(env):
            chunk = client.infer(to_observation(env.get_obs(), prompt))["actions"][: args.replan_steps]
            for action in chunk:  # absolute joint targets: left arm, left gripper, right arm, right gripper
                env.take_action(np.asarray(action, dtype=np.float32), action_type="qpos")
                if env.eval_success:
                    success = True
                    break
                if rt.is_episode_end(env):
                    break
            if success:
                break
        env.close_env(clear_cache=(done + 1) % cfg["clear_cache_freq"] == 0)
        done += 1
        successes += success
        logging.info(f"{task} episode {done} (seed {seed}): {'success' if success else 'failure'}, "
                     f"{successes}/{done} so far, prompt: {prompt!r}")
        seed += 1
    return successes


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    task_config = write_task_config(args.arm, args.base_config)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    results = {}
    for task in args.tasks:
        results[task] = run_task(task, task_config, client, args)
        logging.info(f"Task {task}: {results[task]}/{args.num_episodes}")
    total = len(args.tasks) * args.num_episodes
    print("Per task: " + ", ".join(f"{t} {s}/{args.num_episodes}" for t, s in results.items()))
    print(f"Total episodes: {total}")
    print(f"Total success rate: {sum(results.values()) / total}")


if __name__ == "__main__":
    main(tyro.cli(Args))
