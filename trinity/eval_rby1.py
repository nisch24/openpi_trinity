"""Score a trained RB-Y1 checkpoint on held-out episodes (RB-Y1 has no simulator, unlike LIBERO).

For sampled frames of the held-out set (trinity/<name>_heldout, written by convert_rby1.py and never
trained on), the policy sees the 3 camera images, the joint state and the task text, predicts the next
15 actions, and we measure how far those are from what was actually recorded. Lower is better.

Reported (absolute joint targets, the same units as the dataset; arm joints are radians):
  arm_mae          mean |predicted - recorded| over the 14 arm joints and all 15 steps
  arm_mae_first    same, first step only (the action the robot would execute next)
  arm_mae_last     same, last step only (1 second ahead; errors grow with distance)
  grip_mae         mean error of the 2 gripper values
  hold_*           the same metrics for "don't move" (repeat the current state). A trained model
                   should beat this easily; it is a sanity floor, not a competitor.

Run from the openpi repo root on a GPU node:
  uv run python trinity/eval_rby1.py --config pi05_rby1_lora --checkpoint checkpoints/pi05_rby1_lora/run1/29999
Results are appended to logs/rby1_results.csv; per-joint errors go to logs/rby1_eval/<config>_<exp>_<step>.json.
"""
import csv
import dataclasses
import json
import logging
import pathlib

import numpy as np
import tqdm
import tyro

from openpi.policies import policy_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

N_ARM = 14  # first 14 action dims are arm joints, the last 2 are grippers


@dataclasses.dataclass
class Args:
    config: str  # training config the checkpoint came from, e.g. pi05_rby1_lora
    checkpoint: str  # checkpoint step directory, e.g. checkpoints/pi05_rby1_lora/run1/29999
    heldout_repo_id: str | None = None  # default: the config's dataset name + "_heldout"
    stride: int = 5  # score every Nth frame (neighbouring frames are near-duplicates); 1 = all
    max_samples: int | None = None  # cap the number of scored frames (for a quick check)
    out_dir: str = "logs"


def score(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict:
    """Absolute errors for one frame. pred/truth: (horizon, 16); valid: (horizon,) False past the episode end."""
    err = np.abs(pred - truth)[valid]  # (valid steps, 16)
    return {
        "arm": err[:, :N_ARM].mean(),
        "arm_first": err[0, :N_ARM].mean(),
        "arm_last": err[-1, :N_ARM].mean() if valid[-1] else np.nan,  # only when a full chunk exists
        "grip": err[:, N_ARM:].mean(),
        "per_joint": err.mean(axis=0),
    }


def summarize(rows: list[dict], prefix: str) -> dict:
    """Average per-frame scores into one number per metric (nan-safe for arm_last)."""
    return {
        f"{prefix}arm_mae": float(np.mean([r["arm"] for r in rows])),
        f"{prefix}arm_mae_first": float(np.mean([r["arm_first"] for r in rows])),
        f"{prefix}arm_mae_last": float(np.nanmean([r["arm_last"] for r in rows])),
        f"{prefix}grip_mae": float(np.mean([r["grip"] for r in rows])),
    }


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train_config = _config.get_config(args.config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    heldout_repo = args.heldout_repo_id or f"{data_config.repo_id}_heldout"

    # Held-out frames, loaded exactly like training data: each item carries a 15-step action chunk
    ds = _data_loader.create_torch_dataset(
        dataclasses.replace(data_config, repo_id=heldout_repo), train_config.model.action_horizon, train_config.model
    )
    indices = list(range(0, len(ds), args.stride))[: args.max_samples]
    logging.info(f"Scoring {len(indices)} of {len(ds)} held-out frames from {heldout_repo}")

    # Policy = checkpoint weights + the norm stats saved with it + the same transforms as training.
    # Its output is converted back to absolute joint targets (relative arm moves are undone).
    policy = policy_config.create_trained_policy(train_config, args.checkpoint)
    repack = data_config.repack_transforms.inputs  # dataset keys -> the keys a robot client sends

    model_rows, hold_rows = [], []
    for i in tqdm.tqdm(indices, desc="Evaluating"):
        sample = ds[i]
        valid = ~np.asarray(sample["actions_is_pad"], dtype=bool)  # chunk steps past the episode end are padding
        obs = sample
        for t in repack:
            obs = t(obs)
        truth = np.asarray(obs.pop("actions"), dtype=np.float32)  # hide the answer from the policy
        pred = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)
        model_rows.append(score(pred, truth, valid))
        # Baseline: hold the current pose for the whole chunk
        hold = np.repeat(np.asarray(obs["observation/state"], dtype=np.float32)[None], len(truth), axis=0)
        hold_rows.append(score(hold, truth, valid))

    ckpt = pathlib.Path(args.checkpoint)
    result = {
        "config": args.config,
        "checkpoint": "/".join(ckpt.parts[-2:]),  # <exp>/<step>
        "frames": len(indices),
        **summarize(model_rows, ""),
        **summarize(hold_rows, "hold_"),
    }
    for k, v in result.items():
        logging.info(f"{k:>20}: {v:.4f}" if isinstance(v, float) else f"{k:>20}: {v}")

    # One row per evaluated checkpoint, so the variants line up in one table
    out = pathlib.Path(args.out_dir)
    (out / "rby1_eval").mkdir(parents=True, exist_ok=True)
    csv_path = out / "rby1_results.csv"
    new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result))
        if new:
            w.writeheader()
        w.writerow(result)
    per_joint = np.mean([r["per_joint"] for r in model_rows], axis=0)
    detail = out / "rby1_eval" / f"{args.config}_{'_'.join(ckpt.parts[-2:])}.json"
    detail.write_text(json.dumps({**result, "per_joint_mae": per_joint.round(5).tolist()}, indent=2))
    logging.info(f"Appended to {csv_path}; per-joint errors in {detail}")


if __name__ == "__main__":
    main(tyro.cli(Args))
