"""Turn BiGym's human demos (Unitree H1 humanoid with two grippers) into a LeRobot v2.1 dataset.

BiGym stores demos as recorded actions plus a scene seed, without images. So this runs in two steps,
because each needs a different venv:
  dump   (BiGym venv, GPU for rendering) replays every demo of the 5 tasks in the simulator at 50 Hz,
         recording 3 cameras and the robot state, and keeps only the replays that solve the task
         (BiGym warns that not every demo does). Writes <src_root>/<task>/<n>.npz and meta.json.
  build  (openpi venv) writes those into the dataset trinity/bigym_h1.

Layout (trinity/sim_robots.py holds tasks, prompts and cameras):
  state    16 numbers: base x, y, height, turn; 5 left-arm and 5 right-arm joints; 2 grippers
  actions  16 numbers in the same slots: base moves are relative steps, arm joints and grippers are
           absolute targets (BiGym's JointPositionActionMode, absolute, floating base)
meta.json also records each task's longest kept demo; trinity/eval_bigym.py sets its step limit from it.

Run from the openpi repo root (usually via trinity/setup_bigym.sbatch):
  MUJOCO_GL=egl examples/bigym/.venv/bin/python trinity/convert_bigym.py --step dump --src-root $PROJ/bigym_data
  uv run python trinity/convert_bigym.py --step build --src-root $PROJ/bigym_data
"""
import dataclasses
import json
import logging
import pathlib
import shutil
from typing import Literal

import numpy as np
import tyro

import sim_robots

SPEC = sim_robots.BIGYM


@dataclasses.dataclass
class Args:
    step: Literal["dump", "build"]
    src_root: str = "bigym_data"  # replayed demos (npz) and meta.json
    max_demos: int = 50  # successful demos kept per task, like RoboTwin's 50
    overwrite: bool = True


def make_env(task: str):
    """One BiGym task with the action mode its demos were recorded for, and 3 cameras at 224x224."""
    import importlib

    from bigym.action_modes import JointPositionActionMode
    from bigym.action_modes import PelvisDof
    from bigym.utils.observation_config import CameraConfig
    from bigym.utils.observation_config import ObservationConfig

    module, _ = SPEC["tasks"][task]
    size = (SPEC["image_size"], SPEC["image_size"])
    return getattr(importlib.import_module(f"bigym.envs.{module}"), task)(
        action_mode=JointPositionActionMode(
            floating_base=True,
            absolute=True,
            floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],  # the demos move all four
        ),
        control_frequency=SPEC["control_hz"],
        observation_config=ObservationConfig(cameras=[CameraConfig(name, resolution=size) for name in SPEC["cameras"]]),
        render_mode=None,
    )


def robot_state(env) -> np.ndarray:
    """Base position, arm joints and grippers, in the same 16 slots as the actions."""
    return np.asarray(env.unwrapped.robot.qpos_actuated, dtype=np.float32)


def images(obs: dict) -> dict:
    """BiGym returns each camera as channels-first; store height x width x channel."""
    return {feature: np.moveaxis(obs[f"rgb_{name}"], 0, -1) for name, feature in SPEC["cameras"].items()}


def dump(args: Args) -> None:
    import copy

    from demonstrations.demo_store import DemoStore
    from demonstrations.utils import Metadata
    from demonstrations.utils import ObservationMode

    root = pathlib.Path(args.src_root)
    store = DemoStore(cache_root=root / "cache")  # BiGym's default cache is the home folder, which is small on PACE
    meta = {}
    for task, (_, prompt) in SPEC["tasks"].items():
        env = make_env(task)
        # Ask for the stored (image-free) demos, resampled from 500 Hz to our control rate
        metadata = copy.deepcopy(Metadata.from_env(env))
        metadata.observation_mode = ObservationMode.Lightweight
        demos = store.get_demos(metadata, amount=-1, frequency=SPEC["control_hz"])
        demos = sorted(demos, key=lambda d: d.metadata.uuid)  # BiGym shuffles them; fix the order so reruns match
        out = root / task
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        lengths = []
        for demo in demos:
            if len(lengths) == args.max_demos:
                break
            obs, _ = env.reset(seed=demo.seed)  # the seed rebuilds the demo's scene
            frames, states, actions, solved = {f: [] for f in SPEC["cameras"].values()}, [], [], False
            for timestep in demo.timesteps:
                action = np.asarray(timestep.executed_action, dtype=np.float32)
                for feature, image in images(obs).items():
                    frames[feature].append(image)
                states.append(robot_state(env))
                actions.append(action)
                obs, _, terminated, truncated, info = env.step(action)
                solved = solved or bool(info["task_success"])
                if terminated or truncated:
                    break
            if not solved:
                continue  # the replay did not solve the task, so it is not a usable demo
            if len(actions[0]) != SPEC["action_dim"] or len(states[0]) != SPEC["action_dim"]:
                raise SystemExit(f"BIGYM CHECK BROKEN: {task} has {len(actions[0])} action and "
                                 f"{len(states[0])} state numbers, expected {SPEC['action_dim']}")
            np.savez_compressed(out / f"{len(lengths):03d}.npz", state=np.stack(states), actions=np.stack(actions),
                                **{feature: np.stack(f) for feature, f in frames.items()})
            lengths.append(len(actions))
        env.close()
        meta[task] = {"prompt": prompt, "demos": len(lengths), "available": len(demos),
                      "max_length": max(lengths, default=0), "mean_length": float(np.mean(lengths)) if lengths else 0}
        print(f"{task}: kept {len(lengths)} of {len(demos)} demos that solve the task on replay; "
              f"steps per demo {min(lengths, default=0)}-{max(lengths, default=0)}")
        if len(lengths) < 20:
            print(f"BIGYM CHECK WEAK: only {len(lengths)} usable demos for {task}")
    (root / "meta.json").write_text(json.dumps(meta, indent=1))
    print("BIGYM DUMP DONE")


def build(args: Args) -> None:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    root = pathlib.Path(args.src_root)
    meta = json.loads((root / "meta.json").read_text())
    path = HF_LEROBOT_HOME / SPEC["repo_id"]
    if path.exists():
        if not args.overwrite:
            raise FileExistsError(path)
        shutil.rmtree(path)
    size, dim = SPEC["image_size"], SPEC["action_dim"]
    image_feature = {"dtype": "image", "shape": (size, size, 3), "names": ["height", "width", "channel"]}
    names = ["base_x", "base_y", "base_z", "base_turn"] + [f"left_arm_{i}" for i in range(5)]
    names += [f"right_arm_{i}" for i in range(5)] + ["left_gripper", "right_gripper"]
    dataset = LeRobotDataset.create(
        repo_id=SPEC["repo_id"],
        robot_type="unitree_h1",
        fps=SPEC["control_hz"],
        features={
            **{feature: image_feature for feature in SPEC["cameras"].values()},
            "state": {"dtype": "float32", "shape": (dim,), "names": names},
            "actions": {"dtype": "float32", "shape": (dim,), "names": names},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    for task, info in meta.items():
        for file in sorted((root / task).glob("*.npz")):
            demo = dict(np.load(file))  # read each array once (indexing the file re-reads it every time)
            # Frame t holds the observation before action t, so they are already paired
            for t in range(len(demo["actions"])):
                dataset.add_frame(
                    {
                        **{feature: demo[feature][t] for feature in SPEC["cameras"].values()},
                        "state": demo["state"][t],
                        "actions": demo["actions"][t],
                        "task": info["prompt"],
                    }
                )
            dataset.save_episode()
    logging.info(f"Converted { {task: info['demos'] for task, info in meta.items()} } to {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = tyro.cli(Args)
    dump(args) if args.step == "dump" else build(args)
