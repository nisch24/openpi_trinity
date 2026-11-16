"""Success-rate evaluation on BiGym (Unitree H1 humanoid with two grippers), 5 tasks.

Connects to a running policy server (scripts/serve_policy.py), runs N episodes per task from fixed
seeds, and counts an episode as a success when BiGym reports the task solved (info["task_success"]).
BiGym sets no time limit, so each task's limit is 1.5x its longest training demo (from meta.json,
written by trinity/convert_bigym.py). Prints "Total episodes" / "Total success rate" in
examples/libero/main.py's format for trinity/collect_results.py.

Runs in the BiGym venv (trinity/setup_bigym.sbatch); usually launched by trinity/eval_bigym.sbatch.
"""
import dataclasses
import json
import logging
import pathlib

import imageio
import numpy as np
from openpi_client import websocket_client_policy
import tyro

from convert_bigym import images
from convert_bigym import make_env
from convert_bigym import robot_state
import sim_robots

SPEC = sim_robots.BIGYM


@dataclasses.dataclass
class Args:
    port: int = 8000
    host: str = "0.0.0.0"
    src_root: str = "bigym_data"  # holds meta.json (prompts and demo lengths)
    episodes_per_task: int = 20  # x 5 tasks = 100 episodes, as on RoboTwin
    step_limit_factor: float = 1.5  # episode limit = this x the task's longest training demo
    replan_steps: int = 25  # actions executed from each 50-step chunk before asking again (0.5 s)
    seed: int = 0  # episode i of every task uses seed + i, so every model sees the same scenes
    video_out_path: str = "logs/videos/bigym"
    save_videos: int = 1  # episodes per task saved as mp4 (head camera)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    meta = json.loads((pathlib.Path(args.src_root) / "meta.json").read_text())
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    out = pathlib.Path(args.video_out_path)
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    for task in SPEC["tasks"]:
        env = make_env(task)
        low, high = env.action_space.low, env.action_space.high
        max_steps = int(args.step_limit_factor * meta[task]["max_length"])
        successes = 0
        for ep in range(args.episodes_per_task):
            obs, _ = env.reset(seed=args.seed + ep)
            frames, plan, success = [], [], False
            for step in range(max_steps):
                if not plan:  # ask the policy for a new chunk and keep its first replan_steps actions
                    request = {f"observation/{feature}": image for feature, image in images(obs).items()}
                    request |= {"observation/state": robot_state(env), "prompt": meta[task]["prompt"]}
                    plan = list(client.infer(request)["actions"][: args.replan_steps])
                # Keep commands inside the simulator's joint and base-step limits
                action = np.clip(np.asarray(plan.pop(0), dtype=np.float32), low, high)
                obs, _, terminated, truncated, info = env.step(action)
                success = bool(info["task_success"])
                if ep < args.save_videos and step % 5 == 0:  # 10 frames per second of simulated time
                    frames.append(images(obs)["image"])
                if success or terminated or truncated:  # BiGym ends the episode on success or a broken state
                    break
            successes += success
            if frames:
                imageio.mimwrite(out / f"{task}_ep{ep:02d}_{'success' if success else 'fail'}.mp4", frames, fps=10)
            logging.info(f"{task} episode {ep + 1}/{args.episodes_per_task}: {'success' if success else 'failure'}")
        env.close()
        total += successes
        print(f"Task {task}: {successes}/{args.episodes_per_task} (limit {max_steps} steps)")

    episodes = args.episodes_per_task * len(SPEC["tasks"])
    print(f"Total episodes: {episodes}")
    print(f"Total success rate: {total / episodes}")


if __name__ == "__main__":
    main(tyro.cli(Args))
