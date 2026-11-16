"""Success-rate evaluation on the simulated xArm lifting a cube (gym-xarm Lift), like eval_aloha_sim.py.

Connects to a running policy server (scripts/serve_policy.py), runs N episodes, and counts an episode as a
success when the cube rises 0.15 m above where it rested at the start. gym-xarm's own is_success compares
a height measured from the table centre with a target measured from the floor, so it never fires
(trinity/check_xarm.py shows this); the 0.15 m threshold is the one its Lift task intends. Prints
"Total episodes" and "Total success rate" in examples/libero/main.py's format for trinity/collect_results.py.

Runs in the xArm client venv (trinity/setup_xarm.sbatch). Usually launched by trinity/eval_xarm.sbatch.
"""
import dataclasses
import logging
import pathlib

import gym_xarm  # noqa: F401  (registers gym_xarm/XarmLift-v0)
import gymnasium
import imageio
import numpy as np
from openpi_client import websocket_client_policy
import tyro

LIFT = 0.15  # metres the cube must rise for success (gym-xarm Lift's intended threshold)
PROMPT = "Pick up the cube and lift it."  # the dataset's task text, which training used as the prompt


@dataclasses.dataclass
class Args:
    port: int = 8000
    host: str = "0.0.0.0"
    task: str = "gym_xarm/XarmLift-v0"
    num_episodes: int = 100
    max_steps: int = 50  # the Lift task's own episode_length; gym-xarm registers a looser 300
    replan_steps: int = 5  # actions executed from each 10-step chunk before asking again (LIBERO's setting)
    seed: int = 0  # episode i uses seed + i, so every model sees the same starting scenes
    video_out_path: str = "logs/videos/xarm"
    save_videos: int = 5  # save the first N episodes as mp4 for a quick visual check


def to_observation(gym_obs: dict) -> dict:
    """gym-xarm observation -> the keys xarm_policy.XarmInputs expects (raw 84x84 image, as in training)."""
    return {
        "observation/image": np.ascontiguousarray(gym_obs["pixels"]),
        "observation/state": np.asarray(gym_obs["agent_pos"], dtype=np.float32),
        "prompt": PROMPT,
    }


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    env = gymnasium.make(args.task, obs_type="pixels_agent_pos", max_episode_steps=args.max_steps)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    out = pathlib.Path(args.video_out_path)
    out.mkdir(parents=True, exist_ok=True)

    successes = 0
    for ep in range(args.num_episodes):
        gym_obs, _ = env.reset(seed=args.seed + ep)
        frames, plan, success, done = [], [], False, False
        start_z = env.unwrapped.obj[2]  # the cube's resting height; obj is in table-centre coordinates
        while not done:
            if not plan:  # ask the policy for a new chunk and keep its first replan_steps actions
                plan = list(client.infer(to_observation(gym_obs))["actions"][: args.replan_steps])
            # gym-xarm rejects actions outside [-1, 1], so clip the model's output into range
            action = np.clip(np.asarray(plan.pop(0), dtype=np.float32), -1.0, 1.0)
            gym_obs, _, terminated, truncated, info = env.step(action)
            success = success or env.unwrapped.obj[2] - start_z >= LIFT
            done = terminated or truncated
            if ep < args.save_videos:
                frames.append(env.render())
        successes += success
        if frames:
            imageio.mimwrite(out / f"episode{ep:03d}_{'success' if success else 'fail'}.mp4", frames, fps=25)
        logging.info(f"Episode {ep + 1}/{args.num_episodes}: {'success' if success else 'failure'} "
                     f"(running success rate {successes / (ep + 1):.2f})")

    print(f"Total episodes: {args.num_episodes}")
    print(f"Total success rate: {successes / args.num_episodes}")


if __name__ == "__main__":
    main(tyro.cli(Args))
