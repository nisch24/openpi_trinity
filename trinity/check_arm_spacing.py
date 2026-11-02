"""Check that RoboTwin's simulator, as we configure it for scoring, starts where the demonstrations started.

Franka and UR5 are single arms that RoboTwin pairs at a chosen spacing. The demos don't state theirs, so
compare the first demo frame's joints and end-effector poses with a freshly reset simulator on the same arm.
Matching poses mean the scoring setup has the demos' geometry. Aloha-AgileX has no spacing: it is the control.

Two steps, because each needs a different venv (usually run by trinity/check_arm_spacing.sbatch):
  demo: in openpi's venv, from the repo root, save the first demo frame to --frame
  sim:  in RoboTwin's venv, from RoboTwin's folder, reset the simulator and compare with --frame
"""
import dataclasses
import pathlib
from typing import Literal

import numpy as np
import tyro

TASK = "click_bell"  # the robot's start doesn't depend on the task; this one is the setup smoke test's
TOLERANCE = 0.01  # largest accepted difference: metres for positions, radians for joints


@dataclasses.dataclass
class Args:
    step: Literal["demo", "sim"]
    arm: str  # dataset arm name: aloha-agilex, franka or ur5
    frame: str  # .npz holding the first demo frame
    src_root: str = "robotwin_data"  # demo: where convert_robotwin.py keeps the unpacked demos


def save_demo_frame(args: Args) -> None:
    """First frame of one demo -> args.frame (downloads the demos if they aren't unpacked)."""
    import convert_robotwin
    import h5py

    folder = convert_robotwin.fetch_task(pathlib.Path(args.src_root), TASK, args.arm, "clean_50")
    with h5py.File(folder / "data/episode0.hdf5", "r") as f:
        np.savez(
            args.frame,
            left=f["endpose/left_endpose"][0],
            right=f["endpose/right_endpose"][0],
            joints=f["joint_action/vector"][0],
        )


def compare_with_sim(args: Args) -> None:
    """Reset the simulator with the embodiment scoring uses and print how far it is from the demo frame."""
    import eval_robotwin
    import yaml

    rt = eval_robotwin.rt
    # Own yml, so the config the scoring jobs use is left alone
    cfg_dir = eval_robotwin.ROBOTWIN / "env_cfg" / "task_config"
    cfg = yaml.safe_load((cfg_dir / "demo_clean.yml").read_text())
    cfg.update(embodiment=eval_robotwin.SIM_ARM[args.arm], eval_video_log=False)
    name = f"trinity_spacing_{args.arm}"
    (cfg_dir / f"{name}.yml").write_text(yaml.safe_dump(cfg))

    task_args, _ = rt.load_task_args({"task_name": TASK, "task_config": name, "policy_name": "trinity"})
    task_args["eval_mode"] = True
    env = rt.class_decorator(TASK)
    obs = None
    for seed in range(100000, 100010):  # a few seeds, in case a scene is unstable
        try:
            env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **task_args)
            obs = env.get_obs()
            break
        except Exception as e:  # noqa: BLE001
            print(f"seed {seed} failed: {type(e).__name__}: {e}")
            rt.safe_close_env(env)
    if obs is None:
        raise SystemExit(f"SPACING UNKNOWN: {args.arm}: no scene could be set up")
    if "endpose" not in obs:
        raise SystemExit(f"SPACING UNKNOWN: {args.arm}: no endpose in the observation, only {list(obs)}")

    demo = np.load(args.frame)
    sim_left = np.asarray(obs["endpose"]["left_endpose"])
    sim_right = np.asarray(obs["endpose"]["right_endpose"])
    sim_joints = np.asarray(obs["joint_action"]["vector"])
    print(f"{args.arm}: embodiment {cfg['embodiment']}")
    print(f"  demo left xyz {demo['left'][:3].round(4)}  right xyz {demo['right'][:3].round(4)}")
    print(f"  sim  left xyz {sim_left[:3].round(4)}  right xyz {sim_right[:3].round(4)}")
    # Positions show the geometry; the joints show both start from the same pose, so the comparison is fair
    pos_diff = max(np.abs(sim_left[:3] - demo["left"][:3]).max(), np.abs(sim_right[:3] - demo["right"][:3]).max())
    joint_diff = np.abs(sim_joints - demo["joints"]).max()
    print(f"  largest position difference {pos_diff:.4f} m, largest joint difference {joint_diff:.4f}")
    if joint_diff > TOLERANCE:
        print(f"SPACING UNKNOWN: {args.arm}: the sim and the demo start from different joints")
    else:
        print(f"SPACING {'OK' if pos_diff <= TOLERANCE else 'DIFFERS'}: {args.arm}")


if __name__ == "__main__":
    cli_args = tyro.cli(Args)
    if cli_args.step == "demo":
        save_demo_frame(cli_args)
    else:
        compare_with_sim(cli_args)
