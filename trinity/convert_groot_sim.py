"""Convert NVIDIA's simulated demos for one robot into a LeRobot v2.1 dataset openpi trains on.

Source: nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim (LeRobot v2.0 layout, which openpi's pinned
LeRobot cannot read). Each task folder holds data/chunk-000/episode_<i>.parquet (state, action, labels)
and videos/chunk-000/<camera>/episode_<i>.mp4 (256x256, 20 fps). Robots, tasks and layouts are in
trinity/sim_robots.py:
  robocasa  Panda on an Omron base, kitchen tasks   -> trinity/robocasa_panda (3 cameras, 16 state, 12 actions)
  gr1       GR-1 upper body with hands, tabletop    -> trinity/gr1_tabletop   (1 camera, 29 state, 29 actions)

Only the first --episodes demos per task are downloaded, so the 1.9 TB source is never fetched whole.
Output features match the other Trinity datasets (image[, left_wrist_image, right_wrist_image], state,
actions). Also writes <src_root>/<robot>_reference.json (first state and prompt per task) for the
simulator check in trinity/eval_robocasa.py --check.

Run from the openpi repo root on PACE:
  uv run python trinity/convert_groot_sim.py --robot robocasa
Quick test:
  uv run python trinity/convert_groot_sim.py --robot gr1 --episodes 2
"""
import dataclasses
import json
import logging
import pathlib
import shutil
import time
from typing import Literal

import av
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pandas as pd
import tqdm
import tyro

import sim_robots


@dataclasses.dataclass
class Args:
    robot: Literal["robocasa", "gr1"]
    episodes: int = 100  # demos per task (the source has 1,000+; chunk-000 holds the first 1,000)
    src_root: str = "groot_sim_data"  # where the source files are downloaded
    overwrite: bool = True


def download(path: str, src_root: pathlib.Path) -> None:
    """Fetch one file of the source dataset, waiting and retrying when Hugging Face rate-limits us."""
    from huggingface_hub import hf_hub_download  # uses HF_TOKEN from the environment

    if (src_root / path).exists():  # kept from an earlier run
        return
    for attempt in range(8):
        try:
            hf_hub_download(sim_robots.GROOT_SIM_REPO, path, repo_type="dataset", local_dir=src_root)
            return
        except Exception as error:  # noqa: BLE001
            if "429" not in str(error) or attempt == 7:
                raise
            wait = 60 * (attempt + 1)
            logging.warning(f"Rate-limited on {path}; retrying in {wait} s")
            time.sleep(wait)


def fetch(src_root: pathlib.Path, subset: str, episodes: int, cameras) -> pathlib.Path:
    """Download one task's labels and its first `episodes` demos; return the task folder.

    Files are requested by exact path. Asking for them by pattern makes Hugging Face list all of this
    repo's 600,000+ files first, which uses up the hourly request limit before any download starts.
    """
    paths = [f"{subset}/meta/tasks.jsonl"]
    for i in range(episodes):
        paths.append(f"{subset}/data/chunk-000/episode_{i:06d}.parquet")
        paths += [f"{subset}/videos/chunk-000/observation.images.{name}/episode_{i:06d}.mp4" for name in cameras]
    for path in tqdm.tqdm(paths, desc=f"download {subset}"):
        download(path, src_root)
    return src_root / subset


def read_video(path: pathlib.Path) -> list[np.ndarray]:
    """All frames of an mp4 as RGB uint8 arrays (PyAV bundles its own FFmpeg, which PACE lacks)."""
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def select(vectors: np.ndarray, parts) -> np.ndarray:
    """Keep and reorder the listed (name, start, end) slices of each row."""
    return np.concatenate([vectors[:, start:end] for _, start, end in parts], axis=1).astype(np.float32)


def episode_prompt(df: pd.DataFrame, labels: dict[int, str]) -> str | None:
    """The episode's instruction, or None if the dataset marks the episode as not valid."""
    if "annotation.human.validity" in df and labels[int(df["annotation.human.validity"].iloc[0])].lower() != "valid":
        return None
    # RoboCasa keeps the instruction in its own column; GR-1 uses LeRobot's task_index
    column = "annotation.human.action.task_description"
    prompt = labels[int(df[column if column in df else "task_index"].iloc[0])]
    return prompt.removeprefix(sim_robots.PROMPT_PREFIX).strip() or None


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    spec = sim_robots.GROOT_ROBOTS[args.robot]
    src_root = pathlib.Path(args.src_root)
    state_dim, action_dim = sim_robots.dims(spec["state"]), sim_robots.dims(spec["action"])
    features = {name: feature for name, (feature, _) in spec["cameras"].items()}  # dataset camera -> output name

    path = HF_LEROBOT_HOME / spec["repo_id"]
    if path.exists():
        if not args.overwrite:
            raise FileExistsError(path)
        shutil.rmtree(path)
    image_feature = {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]}
    dataset = LeRobotDataset.create(
        repo_id=spec["repo_id"],
        robot_type=args.robot,
        fps=20,
        features={
            **{feature: image_feature for feature in features.values()},
            "state": {"dtype": "float32", "shape": (state_dim,), "names": [f"state_{i}" for i in range(state_dim)]},
            "actions": {"dtype": "float32", "shape": (action_dim,), "names": [f"action_{i}" for i in range(action_dim)]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    counts, reference = {}, {}
    for task in spec["tasks"]:
        folder = fetch(src_root, spec["subset"].format(task=task), args.episodes, spec["cameras"])
        labels = {}  # index -> text, for instructions and validity labels alike
        for line in (folder / "meta/tasks.jsonl").read_text().splitlines():
            labels[json.loads(line)["task_index"]] = json.loads(line)["task"]
        kept = 0
        for i in tqdm.trange(args.episodes, desc=f"{args.robot}/{task}"):
            df = pd.read_parquet(folder / f"data/chunk-000/episode_{i:06d}.parquet")
            prompt = episode_prompt(df, labels)
            if prompt is None:
                logging.info(f"  skipping {task} episode {i}: marked not valid or no instruction")
                continue
            state = select(np.stack(df["observation.state"].to_numpy()), spec["state"])
            actions = select(np.stack(df["action"].to_numpy()), spec["action"])
            frames = {
                feature: read_video(folder / f"videos/chunk-000/observation.images.{name}/episode_{i:06d}.mp4")
                for name, feature in features.items()
            }
            # A video can differ from its table by a frame; keep the part every stream covers
            length = min(len(df), *(len(f) for f in frames.values()))
            # Row t pairs the observation at t with the action taken at t, so no shifting is needed
            for t in range(length):
                dataset.add_frame(
                    {
                        **{feature: frames[feature][t] for feature in features.values()},
                        "state": state[t],
                        "actions": actions[t],
                        "task": prompt,
                    }
                )
            dataset.save_episode()
            if kept == 0:  # one example per task for the simulator check
                reference[task] = {"prompt": prompt, "state": state[0].tolist(), "frames": length}
            kept += 1
        counts[task] = kept

    (src_root / f"{args.robot}_reference.json").write_text(json.dumps(reference, indent=1))
    logging.info(f"Converted {counts} to {path} ({state_dim} state, {action_dim} action numbers)")
    logging.info(f"Train on it with repo_id='{spec['repo_id']}' (see the pi05_{args.robot}_* configs)")


if __name__ == "__main__":
    main(tyro.cli(Args))
