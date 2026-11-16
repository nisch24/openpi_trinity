"""One table describing Trinity's newer simulated robots, shared by their converters and eval clients.

Pure Python with no dependencies, so it imports in every venv (openpi's and each simulator's).
The converters and eval clients must agree on tasks, camera order and state/action layout; keeping
those here means they cannot drift apart. The matching training configs are in
src/openpi/training/config.py (_SIM_ROBOTS).

  robocasa  Franka Panda on an Omron mobile base, RoboCasa kitchen tasks. 12 action numbers.
  gr1       Fourier GR-1 humanoid upper body with 6-joint hands, RoboCasa GR-1 tabletop tasks. 29 numbers.
  bigym     Unitree H1 humanoid with two grippers, BiGym tasks. 16 numbers.
"""

# NVIDIA's simulated demos (LeRobot v2.0 layout); robocasa and gr1 both read from it
GROOT_SIM_REPO = "nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
# Each robot's evaluation follows NVIDIA's GR00T protocol for these benchmarks:
# 16-step chunks, 8 executed before asking again, 720-step limit, episode ends on success.

ROBOCASA = {
    "repo_id": "trinity/robocasa_panda",  # converted dataset name
    "subset": "single_panda_gripper.{task}",  # folder in GROOT_SIM_REPO
    "env": "robocasa_panda_omron/{task}_PandaOmron_Env",  # gym id registered by robocasa
    # The 5 kitchen tasks GR00T N1.6 solves most often (91-100%), so the variants have room to separate.
    # None is pick-and-place, which every published model finds much harder (19-63%).
    "tasks": ("CloseDrawer", "CloseSingleDoor", "CoffeePressButton", "TurnOffMicrowave", "TurnOffSinkFaucet"),
    # dataset camera -> (output feature, key in the simulator's observation)
    "cameras": {
        "left_view": ("image", "video.res256_image_side_0"),  # left side view -> pi0.5 base slot
        "wrist_view": ("left_wrist_image", "video.res256_image_wrist_0"),  # hand camera -> left wrist slot
        "right_view": ("right_wrist_image", "video.res256_image_side_1"),  # right side view -> right wrist slot
    },
    # State: 16 of the dataset's 53 numbers, as (simulator key, start, end) in the dataset's vector.
    # The rest (absolute hand pose, joint angles, velocities) repeat the same information.
    "state": (
        ("state.end_effector_position_relative", 10, 13),  # hand position relative to the base
        ("state.end_effector_rotation_relative", 17, 21),  # hand rotation relative to the base (quaternion)
        ("state.gripper_qpos", 21, 23),
        ("state.base_position", 0, 3),
        ("state.base_rotation", 3, 7),
    ),
    # Actions: all 12, in the dataset's order. Hand moves are already relative; the last two are 0/1 switches.
    "action": (
        ("action.base_motion", 0, 4),  # base x, y, turn, and torso height
        ("action.control_mode", 4, 5),  # 0 = move the arm, 1 = move the base
        ("action.end_effector_position", 5, 8),
        ("action.end_effector_rotation", 8, 11),
        ("action.gripper_close", 11, 12),
    ),
    "hold": "zero",  # an all-zero action keeps this robot still (used by the setup check)
}

GR1 = {
    "repo_id": "trinity/gr1_tabletop",
    "subset": "gr1_unified.{task}_GR1ArmsAndWaistFourierHands_1000",
    "env": "gr1_unified/{task}_GR1ArmsAndWaistFourierHands_Env",
    # The 5 tabletop tasks GR00T N1.7 solves most often (60-75%); all are pick-and-place.
    "tasks": (
        "PosttrainPnPNovelFromPlateToPlateSplitA",
        "PnPBottleToCabinetClose",
        "PnPCanToDrawerClose",
        "PnPWineToCabinetClose",
        "PosttrainPnPNovelFromTrayToCardboardboxSplitA",
    ),
    # One head camera. The dataset's 256x256 frames are the cropped and padded view GR00T trains on.
    "cameras": {"ego_view": ("image", "video.ego_view_bg_crop_pad_res256_freq20")},
    # State and actions: 29 of the dataset's 44 joint slots (legs and neck stay fixed in this benchmark).
    # Arms are joint angles; hands are 6 finger commands each; all absolute.
    "state": (
        ("state.left_arm", 0, 7),
        ("state.left_hand", 7, 13),
        ("state.right_arm", 22, 29),
        ("state.right_hand", 29, 35),
        ("state.waist", 41, 44),
    ),
    "action": (
        ("action.left_arm", 0, 7),
        ("action.left_hand", 7, 13),
        ("action.right_arm", 22, 29),
        ("action.right_hand", 29, 35),
        ("action.waist", 41, 44),
    ),
    "hold": "state",  # joint targets are absolute, so "stay still" means commanding the current state
}

GROOT_ROBOTS = {"robocasa": ROBOCASA, "gr1": GR1}
PROMPT_PREFIX = "unlocked_waist: "  # tag the GR-1 data and simulator put before the instruction; dropped in both

BIGYM = {
    "repo_id": "trinity/bigym_h1",
    # Task class -> (module in bigym.envs, instruction). BiGym has no language, so these prompts are ours.
    # All five use the same action layout (base x, y, height, turn + 10 arm joints + 2 grippers = 16)
    # and have 50+ human demos; tasks whose demos omit base height (15 numbers) are left out.
    "tasks": {
        "DishwasherClose": ("dishwasher", "close the dishwasher"),
        "DishwasherCloseTrays": ("dishwasher", "push the dishwasher trays back in"),
        "DrawerTopClose": ("cupboards", "close the top drawer"),
        "DrawersAllClose": ("cupboards", "close all the drawers"),
        "WallCupboardClose": ("cupboards", "close the wall cupboard doors"),
    },
    "cameras": {"head": "image", "left_wrist": "left_wrist_image", "right_wrist": "right_wrist_image"},
    "image_size": 224,  # pi0.5's input size, so no resizing happens later
    "control_hz": 50,  # demos are recorded at 500 Hz and replayed at 50 Hz, like ALOHA
    "action_dim": 16,
}


def dims(parts) -> int:
    """Total length of a state or action layout."""
    return sum(end - start for _, start, end in parts)
