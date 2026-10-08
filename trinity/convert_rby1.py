"""Convert a Rainbow Robotics RB-Y1 dataset from LeRobot v3.0 to the v2.1 format openpi trains on.

Rainbow publishes its RB-Y1 datasets in LeRobot v3.0, which packs many episodes into each data
file and each camera video. openpi pins an older LeRobot that only reads v2.1 (one episode per
file), so this script unpacks every episode and rewrites it with openpi's own LeRobot.

Output features (openpi naming, like its LIBERO example):
  image              front camera     -> pi0.5 slot base_0_rgb
  left_wrist_image   left camera      -> pi0.5 slot left_wrist_0_rgb
  right_wrist_image  right camera     -> pi0.5 slot right_wrist_0_rgb
  state, actions     16 numbers: right arm 7, left arm 7, right gripper, left gripper
Images are stored resized-with-padding to 224x224, the size pi0.5 uses, to keep the output small.

Two datasets are written: trinity/<name> for training, and trinity/<name>_heldout with every
20th episode (about 5%), which training never sees. RB-Y1 has no simulator, so models are scored
by how closely their predicted actions match the recorded ones on held-out episodes (eval_rby1.py).

Run from the openpi repo root on PACE (CPU job; needs HF_TOKEN for the download):
  uv run python trinity/convert_rby1.py --repo-id rainbowrobotics/simtos_0412_mirror_v2
Quick test on a few episodes first:
  uv run python trinity/convert_rby1.py --repo-id rainbowrobotics/simtos_0412_mirror_v2 --max-episodes 5
"""
import dataclasses
import json
import logging
import pathlib
import shutil

import av
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from openpi_client import image_tools
import pandas as pd
import tqdm
import tyro

# Rainbow's 16-dim joint order, identical for observation.state and action
JOINTS = [f"right_arm_{i}" for i in range(7)] + [f"left_arm_{i}" for i in range(7)] + ["right_gripper_0", "left_gripper_0"]
# Rainbow camera key -> output feature name
CAMERAS = {
    "observation.images.front": "image",
    "observation.images.left": "left_wrist_image",
    "observation.images.right": "right_wrist_image",
}


@dataclasses.dataclass
class Args:
    repo_id: str = "rainbowrobotics/simtos_0412_mirror_v2"  # Rainbow dataset to convert
    src_root: str | None = None  # use an already-downloaded v3.0 copy instead of downloading
    out_repo_id: str | None = None  # name of the converted dataset (default: trinity/<dataset name>)
    image_size: int = 224  # stored image size; pi0.5 resizes to 224 anyway
    max_episodes: int | None = None  # convert only the first N episodes (for testing)
    overwrite: bool = True  # replace an existing converted copy
    heldout_every: int = 20  # episodes whose index is a multiple of this go to the held-out set (0 = none)


class EpisodeVideoReader:
    """Decodes one camera's v3.0 video files, returning the frames that belong to one episode.

    Episodes sit back to back inside each file, so reading them in order decodes each file once.
    """

    def __init__(self, root: pathlib.Path, video_path_template: str, key: str, fps: int):
        self.root, self.template, self.key, self.fps = root, video_path_template, key, fps
        self.half_frame = 0.5 / fps  # tolerance when matching frame times to an episode's start
        self.path, self.frames, self.last_ts = None, None, -1.0

    def _open(self, path: pathlib.Path, start: float):
        container = av.open(str(path))
        stream = container.streams.video[0]
        if start > 0:  # jump close to the episode start instead of decoding from the beginning
            container.seek(int(start / stream.time_base), stream=stream, backward=True)
        self.path, self.frames = path, container.decode(stream)

    def read(self, chunk: int, file: int, start: float, n: int) -> list[np.ndarray]:
        path = self.root / self.template.format(video_key=self.key, chunk_index=chunk, file_index=file)
        if path != self.path or start < self.last_ts:  # new file, or going backwards: reopen
            self._open(path, start)
        out = []
        for frame in self.frames:
            if frame.time < start - self.half_frame:
                continue  # still before this episode (left over from seeking)
            out.append(frame.to_ndarray(format="rgb24"))
            if len(out) == n:
                break
        if len(out) != n:
            raise RuntimeError(f"{self.key}: expected {n} frames from {start:.3f}s in {path.name}, got {len(out)}")
        self.last_ts = start + (n - 1) / self.fps  # time of the last frame handed out
        return out


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 1. Locate (or download) the v3.0 source dataset
    if args.src_root:
        src = pathlib.Path(args.src_root)
    else:
        from huggingface_hub import snapshot_download  # uses HF_TOKEN from the environment

        src = pathlib.Path(snapshot_download(args.repo_id, repo_type="dataset"))
    info = json.loads((src / "meta/info.json").read_text())
    if not info["codebase_version"].startswith("v3"):
        raise ValueError(f"expected a v3.x dataset, got {info['codebase_version']}")
    for key in ("observation.state", "action"):
        names = info["features"][key]["names"]
        if names != JOINTS:
            raise ValueError(f"{key} layout differs from the expected 16 RB-Y1 joints: {names}")
    for key in CAMERAS:
        if key not in info["features"]:
            raise ValueError(f"camera {key} missing; available: {list(info['features'])}")
    fps = int(info["fps"])

    # 2. Read the episode index: where each episode's rows and video segments live
    episodes = pd.concat(
        [pd.read_parquet(p) for p in sorted((src / "meta/episodes").glob("*/*.parquet"))], ignore_index=True
    ).sort_values("episode_index")
    if args.max_episodes:
        episodes = episodes.head(args.max_episodes)

    # 3. Create the v2.1 output datasets (train + held-out) in openpi's LeRobot home (HF_LEROBOT_HOME)
    out_repo_id = args.out_repo_id or f"trinity/{args.repo_id.split('/')[-1]}"
    s = args.image_size
    image_feature = {"dtype": "image", "shape": (s, s, 3), "names": ["height", "width", "channel"]}

    def create(repo_id: str) -> LeRobotDataset:
        path = HF_LEROBOT_HOME / repo_id
        if path.exists():
            if not args.overwrite:
                raise FileExistsError(path)
            shutil.rmtree(path)
        return LeRobotDataset.create(
            repo_id=repo_id,
            robot_type=info.get("robot_type", "rby1"),
            fps=fps,
            features={
                **{name: image_feature for name in CAMERAS.values()},
                "state": {"dtype": "float32", "shape": (16,), "names": JOINTS},
                "actions": {"dtype": "float32", "shape": (16,), "names": JOINTS},
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )

    train = create(out_repo_id)
    heldout = create(out_repo_id + "_heldout") if args.heldout_every else None
    counts = {"train": 0, "heldout": 0}

    # 4. Copy every episode: rows from the data parquet, frames from each camera's video
    readers = {k: EpisodeVideoReader(src, info["video_path"], k, fps) for k in CAMERAS}
    data_cache: dict[pathlib.Path, pd.DataFrame] = {}
    for _, ep in tqdm.tqdm(episodes.iterrows(), total=len(episodes), desc="Converting episodes"):
        data_file = src / info["data_path"].format(
            chunk_index=int(ep["data/chunk_index"]), file_index=int(ep["data/file_index"])
        )
        if data_file not in data_cache:
            data_cache.clear()  # keep one data file in memory at a time
            data_cache[data_file] = pd.read_parquet(data_file, columns=["episode_index", "frame_index", "observation.state", "action"])
        rows = data_cache[data_file]
        rows = rows[rows["episode_index"] == ep["episode_index"]].sort_values("frame_index")
        n = len(rows)
        if n != int(ep["length"]):
            raise RuntimeError(f"episode {ep['episode_index']}: {n} rows but length {ep['length']}")

        frames = {
            name: readers[key].read(
                int(ep[f"videos/{key}/chunk_index"]),
                int(ep[f"videos/{key}/file_index"]),
                float(ep[f"videos/{key}/from_timestamp"]),
                n,
            )
            for key, name in CAMERAS.items()
        }
        task = str(ep["tasks"][0])  # instruction text for this episode
        # Fixed rule (not random) so the split is identical on every run
        is_heldout = heldout is not None and int(ep["episode_index"]) % args.heldout_every == 0
        out = heldout if is_heldout else train
        counts["heldout" if is_heldout else "train"] += 1
        states = np.stack(rows["observation.state"].to_numpy()).astype(np.float32)
        actions = np.stack(rows["action"].to_numpy()).astype(np.float32)
        for i in range(n):
            out.add_frame(
                {
                    # resize_with_pad keeps the aspect ratio (front is landscape, wrists are portrait)
                    **{name: image_tools.resize_with_pad(frames[name][i], s, s) for name in CAMERAS.values()},
                    "state": states[i],
                    "actions": actions[i],
                    "task": task,
                }
            )
        out.save_episode()

    logging.info(f"Converted {counts['train']} train episodes to {HF_LEROBOT_HOME / out_repo_id}")
    if heldout is not None:
        logging.info(f"and {counts['heldout']} held-out episodes to {HF_LEROBOT_HOME / (out_repo_id + '_heldout')}")
    logging.info(f"Train on it with repo_id='{out_repo_id}' (see the pi05_rby1_* configs)")


if __name__ == "__main__":
    main(tyro.cli(Args))
