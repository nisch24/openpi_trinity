"""Sanity-check a converted RB-Y1 dataset against Rainbow's original before training on it.

Checks, for a few converted episodes:
  1. state/actions equal the source rows exactly (nothing reordered or corrupted)
  2. images are real pictures (not blank) and change over time (video frames advance)
  3. actions look like targets just ahead of the state (action[t] closest to state[t+1])
and saves start/middle/end frames of every camera to logs/rby1_check/ to look at by eye
(are left/right really wrist views? does the scene change between episodes?).

Run from the openpi repo root after convert_rby1.py (CPU is fine):
  uv run python trinity/check_rby1.py
After a full conversion, also check a mirrored copy (source episode 569+) next to its original:
  uv run python trinity/check_rby1.py --first 545 --episodes 2
"""
import dataclasses
import pathlib

from huggingface_hub import snapshot_download
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pandas as pd
from PIL import Image
import tyro

CAMERAS = ["image", "left_wrist_image", "right_wrist_image"]


@dataclasses.dataclass
class Args:
    repo_id: str = "rainbowrobotics/simtos_0412_mirror_v2"  # Rainbow source dataset
    converted: str = "trinity/simtos_0412_mirror_v2"  # converted train set
    episodes: int = 3  # how many converted episodes to check
    first: int = 0  # first converted episode to check (mirrored copies start near 540 after a full conversion)
    out_dir: str = "logs/rby1_check"


def to_hwc_uint8(img) -> np.ndarray:
    """LeRobot returns float CHW in [0, 1]; turn it back into a viewable uint8 HWC image."""
    a = np.asarray(img)
    if a.shape[0] == 3:
        a = a.transpose(1, 2, 0)
    return (a * 255).clip(0, 255).astype(np.uint8) if a.dtype != np.uint8 else a


def main(args: Args) -> None:
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ok = True

    # Source rows (already in the HF cache from the conversion, so this is fast)
    print("Loading source and converted data (about 1-2 minutes)...", flush=True)
    src = pathlib.Path(snapshot_download(args.repo_id, repo_type="dataset", allow_patterns=["data/*", "meta/*"]))
    rows = pd.concat([pd.read_parquet(p, columns=["episode_index", "frame_index", "observation.state", "action"])
                      for p in sorted((src / "data").glob("*/*.parquet"))])
    by_ep = {e: g.sort_values("frame_index") for e, g in rows.groupby("episode_index")}

    for name in (args.converted, args.converted + "_heldout"):
        d = LeRobotDataset(name)
        print(f"{name}: {d.num_episodes} episodes, {d.num_frames} frames")
    ds = LeRobotDataset(args.converted)
    # Number columns only, so reading them never decodes the images (much faster)
    nums = ds.hf_dataset.select_columns(["episode_index", "state", "actions"]).with_format("numpy")
    ep_col = np.asarray(nums["episode_index"])
    all_state, all_actions = np.stack(nums["state"]), np.stack(nums["actions"])
    print("Checking episodes...", flush=True)

    for k in range(args.first, min(args.first + args.episodes, ds.num_episodes)):
        idx = np.flatnonzero(ep_col == k)
        state, actions = all_state[idx], all_actions[idx]

        # 1. Find the source episode with the same length and first state, then compare every value
        match = [e for e, g in by_ep.items() if len(g) == len(idx) and np.allclose(g["observation.state"].iloc[0], state[0])]
        if len(match) != 1:
            print(f"  episode {k}: FAIL, {len(match)} source episodes match its length and first state")
            ok = False
            continue
        g = by_ep[match[0]]
        same = np.array_equal(np.stack(g["observation.state"].to_numpy()).astype(np.float32), state) and \
            np.array_equal(np.stack(g["action"].to_numpy()).astype(np.float32), actions)
        ok &= same
        print(f"  episode {k} (source {match[0]}, {len(idx)} frames): state/actions identical to source: {same}")

        # 3. A target just ahead of the state should sit closer to the next state than to the current one
        now, ahead = np.abs(actions[:-1, :14] - state[:-1, :14]).mean(), np.abs(actions[:-1, :14] - state[1:, :14]).mean()
        print(f"    mean |action - state[t]| = {now:.4f}, |action - state[t+1]| = {ahead:.4f}"
              f" ({'action leads state, as expected' if ahead <= now else 'CHECK: action does not lead state'})")

        # 2. Images: not blank, and frames change over time; save start/middle/end for viewing
        picks = [int(idx[0]), int(idx[len(idx) // 2]), int(idx[-1])]
        frames = {cam: [to_hwc_uint8(ds[i][cam]) for i in picks] for cam in CAMERAS}
        for cam, imgs in frames.items():
            blank = min(float(im.std()) for im in imgs) < 3
            moves = float(np.abs(imgs[0].astype(int) - imgs[-1].astype(int)).mean())
            ok &= not blank and moves > 1
            print(f"    {cam}: pixel std {min(float(im.std()) for im in imgs):.1f}, start-vs-end change {moves:.1f}"
                  f"{'  FAIL: blank or frozen' if blank or moves <= 1 else ''}")
        grid = np.concatenate([np.concatenate(frames[cam], axis=1) for cam in CAMERAS], axis=0)
        Image.fromarray(grid).save(out / f"episode{k}_rows-front-left-right_cols-start-mid-end.png")

    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}; frames saved in {out}/ (rows: front, left, right)")


if __name__ == "__main__":
    main(tyro.cli(Args))
