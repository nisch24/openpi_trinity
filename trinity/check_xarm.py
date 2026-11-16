"""Checks to run before training on the simulated xArm (usually via trinity/setup_xarm.sbatch).

Two steps, because each needs a different venv:
  sim   (xArm client venv) can a lift count as success at all? gym-xarm's Lift compares the cube's height
        measured from the table centre with a target measured from the floor, which may never be reachable.
        Raises the cube 0.2 m by hand and reports whether is_success turns on. Also renders one frame.
  data  (openpi venv) how good are the demos? They are reinforcement-learning rollouts, not human demos,
        so print how many episodes ever reach a high reward.
Look for "SUCCESS CHECK OK" / "SUCCESS CHECK BROKEN" and the reward summary in the log.
"""
import dataclasses
from typing import Literal

import numpy as np
import tyro

RAISE = 0.2  # metres; above the task's 0.15 m threshold


@dataclasses.dataclass
class Args:
    step: Literal["sim", "data"]
    repo_id: str = "lerobot/xarm_lift_medium"


def check_sim() -> None:
    import gym_xarm  # noqa: F401  (registers the environment)
    import gymnasium
    import mujoco

    env = gymnasium.make("gym_xarm/XarmLift-v0", obs_type="pixels_agent_pos")
    obs, _ = env.reset(seed=0)
    print(f"rendered frame {obs['pixels'].shape}, state {np.round(obs['agent_pos'], 3)}")
    u = env.unwrapped
    print(f"at rest: cube height {u.obj[2]:.3f}, target {u.z_target:.3f}, success {bool(u.is_success())}")

    # Lift the cube by hand, well past the threshold, and ask the task again
    qpos = u._utils.get_joint_qpos(u.model, u.data, "object_joint0").copy()
    qpos[2] += RAISE
    u._utils.set_joint_qpos(u.model, u.data, "object_joint0", qpos)
    mujoco.mj_forward(u.model, u.data)
    lifted = bool(u.is_success())
    print(f"raised {RAISE} m: cube height {u.obj[2]:.3f}, target {u.z_target:.3f}, success {lifted}")
    print("SUCCESS CHECK OK: a lift counts as success" if lifted
          else "SUCCESS CHECK BROKEN: even a 0.2 m lift does not count as success")


def check_data(repo_id: str) -> None:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    df = LeRobotDataset(repo_id, video_backend="pyav").hf_dataset.to_pandas()
    best = df.groupby("episode_index")["next.reward"].max()  # best reward each episode reached
    print(f"{len(best)} episodes; best reward per episode:")
    print(best.describe().round(3).to_string())
    for cut in (0.5, 1.0, 1.2):
        print(f"  episodes reaching reward >= {cut}: {(best >= cut).sum()}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    check_sim() if args.step == "sim" else check_data(args.repo_id)
