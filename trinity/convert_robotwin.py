"""Convert RoboTwin 2.0 demonstrations (HDF5) for one arm into a LeRobot v2.1 dataset openpi trains on.

RoboTwin publishes 50 clean demonstrations per task and arm as TianxingChen/RoboTwin2.0
dataset/<task>/<arm>_clean_50.zip (data/episode<i>.hdf5 + instructions/episode<i>.json).
This script downloads the chosen tasks for one arm and writes them into one dataset, so a
single adapter per arm learns all of its tasks.

Output features (openpi naming, like its LIBERO example):
  image              head camera          -> pi0.5 slot base_0_rgb
  left_wrist_image   left wrist camera    -> pi0.5 slot left_wrist_0_rgb
  right_wrist_image  right wrist camera   -> pi0.5 slot right_wrist_0_rgb
  state              joint_action/vector at frame t  (left arm, left gripper, right arm, right gripper)
  actions            joint_action/vector at frame t+1 (the next commanded position, absolute)
Images keep RoboTwin's 320x240; pi0.5 resizes them itself, exactly as it does at evaluation.
Each episode's prompt is one of its "seen" instructions, picked with a fixed seed.

Run from the openpi repo root on PACE (CPU job):
  uv run python trinity/convert_robotwin.py --arm aloha-agilex
Quick test:
  uv run python trinity/convert_robotwin.py --arm franka --tasks click_bell --max-episodes 3
"""
import dataclasses
import io
import json
import logging
import pathlib
import shutil
import zipfile

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
import tqdm
import tyro

# The 5 tasks chosen for Trinity: expert success >= 64% on aloha-agilex, franka and ur5 in the
# RoboTwin 2.0 paper, and mid-range pi0 success (37-62%), so the adapter variants can separate.
TASKS = ("beat_block_hammer", "click_bell", "move_can_pot", "place_empty_cup", "press_stapler")
# RoboTwin camera group -> output feature name
CAMERAS = {"head_camera": "image", "left_camera": "left_wrist_image", "right_camera": "right_wrist_image"}
# Marker written into JPEGs by RoboTwin's newer encoder; without it the bytes hold BGR-ordered data
_RGB_MARKER = b"XPL-RGB1"


@dataclasses.dataclass
class Args:
    arm: str = "aloha-agilex"  # RoboTwin embodiment: aloha-agilex, franka, ur5, arx-x5, piper
    tasks: tuple[str, ...] = TASKS
    setting: str = "clean_50"  # which RoboTwin demo set (clean_50 = 50 demos, no clutter)
    src_root: str = "robotwin_data"  # where zips are downloaded and unpacked (relative to $PROJ is fine)
    out_repo_id: str | None = None  # default: trinity/robotwin_<arm>
    max_episodes: int | None = None  # per task, for testing
    seed: int = 0  # picks each episode's instruction
    overwrite: bool = True


def decode_jpeg(buf: bytes) -> np.ndarray:
    """RoboTwin camera frame -> RGB uint8 HxWx3.

    Older RoboTwin data was written by cv2.imencode on an RGB array, so a standard decoder (PIL)
    returns the channels reversed; data with the RGB marker is a normal JPEG.
    """
    buf = bytes(buf).rstrip(b"\0")  # fixed-width HDF5 strings are NUL-padded
    image = Image.open(io.BytesIO(buf))
    rgb = np.asarray(image.convert("RGB"))
    return rgb if _RGB_MARKER in image.info.get("comment", b"") else rgb[..., ::-1].copy()


def fetch_task(src_root: pathlib.Path, task: str, arm: str, setting: str) -> pathlib.Path:
    """Download and unpack dataset/<task>/<arm>_<setting>.zip once; return the unpacked folder."""
    folder = src_root / task / f"{arm}_{setting}"
    if not (folder / "data").is_dir():
        from huggingface_hub import hf_hub_download  # uses HF_TOKEN from the environment

        zip_path = hf_hub_download(
            "TianxingChen/RoboTwin2.0", f"dataset/{task}/{arm}_{setting}.zip", repo_type="dataset"
        )
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(src_root / task)
    return folder


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    src_root = pathlib.Path(args.src_root)
    folders = {task: fetch_task(src_root, task, args.arm, args.setting) for task in args.tasks}

    # Action layout from the first episode: [left arm n, left gripper, right arm n, right gripper]
    with h5py.File(next(iter(folders.values())) / "data/episode0.hdf5", "r") as f:
        dim = f["joint_action/vector"].shape[1]
        arm_dof = f["joint_action/left_arm"].shape[1]
        height, width = decode_jpeg(f["observation/head_camera/rgb"][0]).shape[:2]
    if dim != 2 * (arm_dof + 1):
        raise ValueError(f"unexpected joint vector size {dim} for {arm_dof}-joint arms")
    names = [f"left_arm_{i}" for i in range(arm_dof)] + ["left_gripper"]
    names += [f"right_arm_{i}" for i in range(arm_dof)] + ["right_gripper"]
    logging.info(f"{args.arm}: {arm_dof} joints per arm, {dim}-dim state/actions, images {width}x{height}")

    out_repo_id = args.out_repo_id or f"trinity/robotwin_{args.arm}"
    path = HF_LEROBOT_HOME / out_repo_id
    if path.exists():
        if not args.overwrite:
            raise FileExistsError(path)
        shutil.rmtree(path)
    image_feature = {"dtype": "image", "shape": (height, width, 3), "names": ["height", "width", "channel"]}
    dataset = LeRobotDataset.create(
        repo_id=out_repo_id,
        robot_type=args.arm,
        fps=15,  # RoboTwin saves demonstrations at 15 frames per second
        features={
            **{name: image_feature for name in CAMERAS.values()},
            "state": {"dtype": "float32", "shape": (dim,), "names": names},
            "actions": {"dtype": "float32", "shape": (dim,), "names": names},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    rng = np.random.default_rng(args.seed)
    counts = {}
    for task, folder in folders.items():
        episodes = sorted((folder / "data").glob("episode*.hdf5"), key=lambda p: int(p.stem[len("episode") :]))
        episodes = episodes[: args.max_episodes] if args.max_episodes else episodes
        for ep_path in tqdm.tqdm(episodes, desc=f"{args.arm}/{task}"):
            instructions = json.loads((folder / "instructions" / f"{ep_path.stem}.json").read_text())["seen"]
            prompt = str(rng.choice(instructions))
            with h5py.File(ep_path, "r") as f:
                vector = f["joint_action/vector"][()].astype(np.float32)
                frames = {name: f[f"observation/{cam}/rgb"][()] for cam, name in CAMERAS.items()}
            # Frame t's observation is paired with the position commanded next (frame t+1)
            for t in range(len(vector) - 1):
                dataset.add_frame(
                    {
                        **{name: decode_jpeg(frames[name][t]) for name in CAMERAS.values()},
                        "state": vector[t],
                        "actions": vector[t + 1],
                        "task": prompt,
                    }
                )
            dataset.save_episode()
        counts[task] = len(episodes)

    logging.info(f"Converted {counts} to {path}")
    logging.info(f"Train on it with repo_id='{out_repo_id}' (see the pi05_robotwin_* configs)")


if __name__ == "__main__":
    main(tyro.cli(Args))
