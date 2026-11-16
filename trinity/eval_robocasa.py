"""Success-rate evaluation for the two RoboCasa-based robots: the Panda on an Omron base (kitchen
tasks) and the GR-1 humanoid upper body (tabletop tasks).

Connects to a running policy server (scripts/serve_policy.py) and follows NVIDIA's GR00T protocol for
these benchmarks: 16-step chunks with 8 executed before asking again, a 720-step limit, and an episode
counts as a success as soon as the simulator reports the task solved (info["success"]). Kitchen
episodes use RoboCasa's evaluation scenes and held-out objects, which its gym wrapper selects itself.
Prints "Total episodes" / "Total success rate" in examples/libero/main.py's format for
trinity/collect_results.py.

Runs in that robot's simulator venv (trinity/setup_robocasa.sbatch); usually launched by
trinity/eval_robocasa.sbatch. With --check it only builds one task, compares the simulator's starting
state with the demos' (from trinity/convert_groot_sim.py) and steps a few "stay still" actions.
"""
import dataclasses
import json
import logging
import pathlib
from typing import Literal

import gymnasium
import imageio
import numpy as np
import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401  (registers the environments)
import tyro

import sim_robots


@dataclasses.dataclass
class Args:
    robot: Literal["robocasa", "gr1"]
    port: int = 8000
    host: str = "0.0.0.0"
    episodes_per_task: int = 20  # x 5 tasks = 100 episodes, as on RoboTwin
    max_steps: int = 720  # GR00T's limit for both benchmarks
    replan_steps: int = 8  # actions executed from each 16-step chunk before asking again
    seed: int = 0  # episode i of every task uses seed + i, so every model sees the same scenes
    video_out_path: str = "logs/videos/robocasa"
    save_videos: int = 1  # episodes per task saved as mp4
    check: bool = False  # setup check only: no policy server needed
    reference: str = ""  # <robot>_reference.json from the converter, for --check


def to_observation(obs: dict, spec: dict) -> dict:
    """Simulator observation -> the keys the policy server expects (same layout as the converted demos)."""
    out = {"observation/state": np.concatenate([np.ravel(obs[key]) for key, _, _ in spec["state"]]).astype(np.float32)}
    for feature, key in spec["cameras"].values():
        name = "observation/image" if feature == "image" else f"observation/{feature}"
        out[name] = np.ascontiguousarray(obs[key])
    # GR-1 envs put the instruction under "coarse_action" (prefixed "unlocked_waist: "), Panda under "task_description"
    text = obs.get("annotation.human.action.task_description", obs.get("annotation.human.coarse_action"))
    out["prompt"] = str(text).removeprefix(sim_robots.PROMPT_PREFIX).strip()
    return out


def to_action(vector: np.ndarray, spec: dict) -> dict:
    """Flat action in the dataset's (reordered) layout -> the simulator's named parts."""
    action, offset = {}, 0
    for key, start, end in spec["action"]:
        part = np.asarray(vector[offset : offset + end - start], dtype=np.float32)
        action[key] = float(part[0]) if end - start == 1 else part  # the 0/1 switches are plain numbers
        offset += end - start
    return action


def check(args: Args, spec: dict) -> None:
    """Build the first task, compare its starting state with the demos', and hold still for a few steps."""
    task = spec["tasks"][0]
    env = gymnasium.make(spec["env"].format(task=task), enable_render=True)
    obs, _ = env.reset(seed=args.seed)
    ours = to_observation(obs, spec)
    state = ours["observation/state"]
    shapes = {key: value.shape for key, value in ours.items() if key.endswith("image")}
    print(f"{task}: state {state.shape}, images {shapes}, prompt {ours['prompt']!r}")
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print(f"simulator start state: {state}")
    if args.reference:
        demo = json.loads(pathlib.Path(args.reference).read_text())[task]
        print(f"demo first state:      {np.asarray(demo['state'])}")
        print(f"demo prompt: {demo['prompt']!r}")
        if len(demo["state"]) != len(state):
            raise SystemExit(f"SIM CHECK BROKEN: state has {len(state)} numbers, demos have {len(demo['state'])}")
    out = pathlib.Path("logs")
    out.mkdir(exist_ok=True)
    for key, image in ours.items():  # one frame per camera, to compare by eye with the demos' videos
        if key.endswith("image"):
            imageio.imwrite(out / f"{args.robot}-check-{key.split('/')[-1]}.png", image)
    # "Stay still": zeros where actions are relative moves (robocasa), the current joints where they are
    # absolute (gr1, whose state and action layouts are the same 29 slots).
    hold = np.zeros(sim_robots.dims(spec["action"]), np.float32) if spec["hold"] == "zero" else state
    for _ in range(5):
        obs, _, _, _, info = env.step(to_action(hold, spec))
    moved = np.abs(to_observation(obs, spec)["observation/state"] - state).max()
    print(f"after 5 hold-still steps: largest state change {moved:.4f}, success {bool(info['success'])}")
    print("SIM CHECK OK: environment builds, renders and accepts actions")


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    spec = sim_robots.GROOT_ROBOTS[args.robot]
    if args.check:
        return check(args, spec)
    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    out = pathlib.Path(args.video_out_path)
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    for task in spec["tasks"]:
        env = gymnasium.make(spec["env"].format(task=task), enable_render=True)
        successes = 0
        for ep in range(args.episodes_per_task):
            obs, _ = env.reset(seed=args.seed + ep)
            frames, plan, success = [], [], False
            for step in range(args.max_steps):
                if not plan:  # ask the policy for a new chunk and keep its first replan_steps actions
                    plan = list(client.infer(to_observation(obs, spec))["actions"][: args.replan_steps])
                obs, _, terminated, truncated, info = env.step(to_action(plan.pop(0), spec))
                success = bool(info["success"])
                if ep < args.save_videos and step % 2 == 0:  # every other frame keeps the files small
                    frames.append(env.render())
                if success or terminated or truncated:  # stop at the first success, as GR00T's rollout does
                    break
            successes += success
            if frames:
                imageio.mimwrite(out / f"{task}_ep{ep:02d}_{'success' if success else 'fail'}.mp4", frames, fps=10)
            logging.info(f"{task} episode {ep + 1}/{args.episodes_per_task}: {'success' if success else 'failure'}")
        env.close()
        total += successes
        print(f"Task {task}: {successes}/{args.episodes_per_task}")

    episodes = args.episodes_per_task * len(spec["tasks"])
    print(f"Total episodes: {episodes}")
    print(f"Total success rate: {total / episodes}")


if __name__ == "__main__":
    main(tyro.cli(Args))
