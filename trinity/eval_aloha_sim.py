"""Success-rate evaluation on ALOHA sim (gym-aloha), the bimanual counterpart to LIBERO's eval.

Connects to a running policy server (scripts/serve_policy.py), runs N episodes of the transfer-cube
task, and counts an episode as a success when the environment reports the task completed
(gym-aloha's max reward). Prints "Total episodes" and "Total success rate" in the same format as
examples/libero/main.py, so trinity/collect_results.py picks the result up.

Runs in the ALOHA client venv (trinity/setup_aloha_client.sh). Usually launched by trinity/eval_aloha.sbatch.
"""
import dataclasses
import logging
import pathlib

import gym_aloha  # noqa: F401  (registers the gym_aloha/* environments)
import gymnasium
import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import tyro


@dataclasses.dataclass
class Args:
    port: int = 8000
    host: str = "0.0.0.0"
    task: str = "gym_aloha/AlohaTransferCube-v0"
    num_episodes: int = 50
    max_steps: int = 400  # episode cap = length of the human demos (8 s at 50 Hz); gym-aloha registers 300
    replan_steps: int = 10  # actions executed from each predicted chunk before asking again (openpi's sim example)
    seed: int = 0  # episode i uses seed + i, so every model sees the same 50 starting scenes
    video_out_path: str = "logs/videos/aloha"
    save_videos: int = 5  # save the first N episodes as mp4 for a quick visual check


def to_observation(gym_obs: dict) -> dict:
    """gym-aloha observation -> the dict openpi's ALOHA policy expects (same as examples/aloha_sim/env.py)."""
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(gym_obs["pixels"]["top"], 224, 224))
    return {
        "state": gym_obs["agent_pos"],
        "images": {"cam_high": np.transpose(img, (2, 0, 1))},  # HWC -> CHW
    }


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    # gym-aloha never sets `truncated` itself, so this time limit is what ends failed episodes
    env = gymnasium.make(args.task, obs_type="pixels_agent_pos", max_episode_steps=args.max_steps)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    out = pathlib.Path(args.video_out_path)
    out.mkdir(parents=True, exist_ok=True)

    successes = 0
    for ep in range(args.num_episodes):
        gym_obs, _ = env.reset(seed=args.seed + ep)
        frames, plan, success, done = [], [], False, False
        while not done:
            if not plan:  # ask the policy for a new chunk and keep its first replan_steps actions
                plan = list(client.infer(to_observation(gym_obs))["actions"][: args.replan_steps])
            gym_obs, reward, terminated, truncated, info = env.step(plan.pop(0))
            # Success = task completed (gym-aloha reports it in info; max reward is 4 for both sim tasks)
            success = success or bool(info.get("is_success", reward >= 4))
            done = terminated or truncated
            if ep < args.save_videos:
                frames.append(gym_obs["pixels"]["top"])
        successes += success
        if frames:
            imageio.mimwrite(out / f"episode{ep:03d}_{'success' if success else 'fail'}.mp4", frames, fps=50)
        logging.info(f"Episode {ep + 1}/{args.num_episodes}: {'success' if success else 'failure'} "
                     f"(running success rate {successes / (ep + 1):.2f})")

    print(f"Total episodes: {args.num_episodes}")
    print(f"Total success rate: {successes / args.num_episodes}")


if __name__ == "__main__":
    main(tyro.cli(Args))
