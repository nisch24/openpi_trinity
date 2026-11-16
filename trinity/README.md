# Trinity: per-robot LoRA adapters on π0.5 (`lora-adapters` branch)

This branch tests whether one shared π0.5 base can serve many robots, each with a small per-robot
adapter (~0.25 GB) instead of its own fine-tuned copy (~7 GB). It adds LoRA to π0.5's vision
encoder (SigLIP), training configs for several robots, and the scripts to train and score them on
Georgia Tech's PACE cluster.

## The variants

Every variant puts openpi's LoRA on the language model and action expert (`gemma_2b_lora`,
`gemma_300m_lora`). They differ only in how the vision encoder is handled:

| Variant | Config suffix | Vision encoder (SigLIP) |
|---|---|---|
| A | (none) | Fully trained per robot |
| B | `_strict` | Frozen |
| C r16 | `_vision` | Frozen, plus a rank-16 LoRA |
| C r128 | `_vision_r128` | Frozen, plus a rank-128 LoRA |
| C r256 / r512 | `_vision_r256`, `_vision_r512` | Larger LoRA (ALOHA sim only) |
| C r128 plus | `_vision_r128_plus` | r128 with a 4× learning rate, and SigLIP's embeddings, norms and head also trained (ALOHA sim only) |

`trinity/adapter_size.py` prints how many parameters each variant trains per robot.

## Changes to openpi itself

These are the only edits outside `trinity/`. Each is marked with a `Trinity:` comment.

| File | Change |
|---|---|
| `src/openpi/models/siglip.py` | Adds `LoRADotGeneral`, a drop-in matmul that adds a low-rank update. When `lora_rank > 0`, it is applied to every encoder block's q/k/v/out projections and both MLP layers; each layer gets its own adapter weights. The pretrained kernels keep their names, so base weights load unchanged. |
| `src/openpi/models/pi0_config.py` | New `vision_lora_rank` field (default 0, which leaves the original model unchanged). JAX model only. |
| `src/openpi/models/pi0.py` | Passes `vision_lora_rank` into SigLIP. |
| `src/openpi/training/config.py` | All Trinity configs (below), plus the data configs for RoboTwin, RB-Y1, xArm and `LeRobotSimRobotDataConfig` (RoboCasa, GR-1, BiGym). |
| `src/openpi/policies/robotwin_policy.py` | New: RoboTwin input/output transforms. Three cameras map to π0.5's image slots; outputs are cut back to the arm's 14 or 16 action dimensions. |
| `src/openpi/policies/rby1_policy.py` | New: RB-Y1 input/output transforms. |
| `src/openpi/policies/xarm_policy.py` | New: xArm input/output transforms. Its one camera fills the base image slot; outputs are cut back to 4 actions. |
| `src/openpi/training/data_loader.py` | Decodes video datasets with PyAV, because PACE nodes lack the FFmpeg libraries LeRobot's default decoder needs. Image datasets are unaffected. |
| `src/openpi/serving/websocket_policy_server.py` | Turns off websocket keepalive pings. A simulator client can block for more than 20 s between requests (RoboTwin's planner) and would otherwise be disconnected mid-evaluation. |

With `vision_lora_rank=0`, every existing openpi config behaves as before.

## Configs added

| Robot | Configs | Data | Default steps |
|---|---|---|---|
| LIBERO (1 Franka arm) | `pi05_libero_lora{,_strict,_vision,_vision_r128}` | `physical-intelligence/libero`, norm stats from the published `pi05_libero` | 30k (runs stopped at 5k) |
| ALOHA sim (2 arms, transfer cube) | `pi05_aloha_sim_lora{,_strict,_vision,_vision_r128,_vision_r256,_vision_r512,_vision_r128_plus}` | `lerobot/aloha_sim_transfer_cube_human` (50 human demos) | 20k |
| RoboTwin 2.0 (Aloha-AgileX, Franka, UR5) | `pi05_robotwin_{aloha,franka,ur5}_lora{,_strict,_vision,_vision_r128}` | `trinity/robotwin_<arm>`, from `convert_robotwin.py` | 20k (runs stopped at 5k) |
| xArm (1 arm, lift a cube) | `pi05_xarm_lora{,_vision_r128}` | `lerobot/xarm_lift_medium` (800 reinforcement-learning episodes, read as v2.1) | 30k |
| RoboCasa (Panda arm on an Omron mobile base, 12 actions) | `pi05_robocasa_lora{,_vision_r128}` | `trinity/robocasa_panda`: 5 kitchen tasks x 100 demos from NVIDIA's GR00T simulation dataset | 30k |
| BiGym (Unitree H1 humanoid with two grippers, 16 actions) | `pi05_bigym_lora{,_vision_r128}` | `trinity/bigym_h1`: 5 tasks x up to 50 human demos, replayed in the simulator at 50 Hz | 30k |
| GR-1 tabletop (Fourier GR-1 upper body with 6-joint hands, 29 actions) | `pi05_gr1_lora{,_vision_r128}` | `trinity/gr1_tabletop`: 5 tabletop tasks x 100 demos from the same NVIDIA dataset | 30k |
| RB-Y1 (real robot, no simulator) | `pi05_rby1`, `pi05_rby1_lora{,_strict,_vision,_vision_r64,_vision_r128}` | `trinity/<Rainbow dataset>`, from `convert_rby1.py` | |

How RoboTwin is set up:
- **State and action:** the state is the joint vector at frame t; the action is the joint vector at frame t+1.
- **Action encoding:** arm joints are learned as deltas from the current state; grippers are absolute (as in openpi's ALOHA config).
- **Chunks:** 15-step action chunks (1 s at 15 fps).
- **Data:** five tasks with 50 clean demos each: beat_block_hammer, click_bell, move_can_pot, place_empty_cup, press_stapler.
- **Images:** the converter undoes the BGR channel order in RoboTwin's older JPEGs.

How the three newest robots are set up (`trinity/sim_robots.py` holds their tasks, cameras and layouts):
- **RoboCasa:** tasks CloseDrawer, CloseSingleDoor, CoffeePressButton, TurnOffMicrowave, TurnOffSinkFaucet. Three cameras (left, right, wrist). State is 16 of the dataset's 53 numbers (hand pose relative to the base, gripper, base pose). Actions are used as recorded.
- **GR-1 tabletop:** five pick-and-place tasks. One head camera. State and actions are the 29 joints this benchmark moves (two arms, two hands, waist); arm joints are learned as deltas, hands and waist absolute.
- **BiGym:** tasks DishwasherClose, DishwasherCloseTrays, DrawerTopClose, DrawersAllClose, WallCupboardClose. Three cameras (head, two wrists) at 224x224. Only demos that solve the task when replayed are kept. Arm joints are learned as deltas; base steps and grippers as recorded. BiGym has no instructions, so the prompts are ours.
- **Chunks:** 16 steps for RoboCasa and GR-1 (20 fps), 50 steps for BiGym (50 Hz).

## Scripts in `trinity/`

| Script | Purpose |
|---|---|
| `train.sbatch`, `submit.sh` | Train any config (`CONFIG=… EXP=… STEPS=…`). Resubmitting resumes from the last checkpoint. `submit.sh` picks fast GPU partitions. |
| `eval.sbatch`, `setup_libero_client.sh` | LIBERO success rate, using openpi's `examples/libero` client. |
| `setup_aloha.sbatch`, `eval_aloha_sim.py`, `eval_aloha.sbatch` | ALOHA sim client environment and norm stats; success rate from gym-aloha's `is_success` (400-step cap). |
| `robotwin_feasibility.sbatch` | One-time RoboTwin install and render test (needs an RTX-class GPU). |
| `setup_robotwin.sbatch` | Builds CuRobo (`REBUILD_CUROBO=1` to recompile), installs the client dependencies, and checks that the scripted expert solves a task on each arm. |
| `convert_robotwin.py`, `.sbatch` | Downloads RoboTwin demos for one arm, converts them to LeRobot format, and computes norm stats. |
| `eval_robotwin.py`, `.sbatch` | RoboTwin success rate following RoboTwin's protocol: expert-solvable seeds, its instruction generator, the Easy setting, 20 episodes per task. |
| `setup_xarm.sbatch`, `check_xarm.py` | xArm client environment and norm stats; checks that a lift can count as success and how good the demos are. |
| `eval_xarm_sim.py`, `eval_xarm.sbatch` | xArm success rate: cube 0.15 m above its resting height, 50-step episodes. gym-xarm's own `is_success` mixes two height references and never fires (confirmed on PACE), so the scorer applies the intended 0.15 m rule itself. |
| `sim_robots.py` | One table of tasks, cameras and state/action layouts for RoboCasa, GR-1 and BiGym, shared by their converters and eval clients. |
| `setup_robocasa.sbatch`, `convert_groot_sim.py` | For `ROBOT=robocasa` or `gr1`: installs the simulator, downloads and converts the demos, checks the simulator against them, and computes norm stats. |
| `setup_bigym.sbatch`, `convert_bigym.py` | Installs BiGym, replays its demos with cameras, writes the dataset, and computes norm stats. |
| `eval_sim.sbatch`, `eval_robocasa.py`, `eval_bigym.py` | Success rate for `ROBOT=robocasa`, `gr1` or `bigym`: 5 tasks x 20 episodes. |
| `run_all.sh` | Submits one stage (`setup`, `train` or `eval`) for all four new robots. |
| `convert_rby1.py`, `.sbatch`, `check_rby1.py`, `eval_rby1.py`, `.sbatch` | RB-Y1 conversion with a held-out split, and action-error scoring (no simulator). |
| `check_arm_spacing.py`, `.sbatch` | Compares a reset simulator with the first demo frame on each arm, to verify the Franka and UR5 spacing. |
| `collect_results.py` | Gathers every `logs/eval-*.out` into one table and `logs/results.csv`. |
| `adapter_size.py` | Trainable parameters and per-robot size for each variant. |

## Typical run (on PACE, from the repo root)

```bash
# Train (paid inferno queue; resubmit the same command to continue after a timeout)
CONFIG=pi05_aloha_sim_lora_vision_r128 sbatch -q inferno --gres=gpu:h200:1 trinity/train.sbatch

# Score ALOHA sim
CONFIG=pi05_aloha_sim_lora_vision_r128 CKPT=checkpoints/pi05_aloha_sim_lora_vision_r128/run1/19999 \
  TRIALS=100 sbatch trinity/eval_aloha.sbatch

# Score RoboTwin (RTX-class GPU)
ARM=franka CONFIG=pi05_robotwin_franka_lora_vision_r128 \
  CKPT=checkpoints/pi05_robotwin_franka_lora_vision_r128/run1/4999 \
  sbatch -p gpu-rtx6000 --gres=gpu:rtx_6000:1 trinity/eval_robotwin.sbatch

python3 trinity/collect_results.py
```

Weights, datasets and checkpoints live only in PACE scratch, never in the repo.

## Results so far (success rate)

| Robot | A | B | C r16 | C r128 | Notes |
|---|---|---|---|---|---|
| LIBERO, 5k steps, 50 episodes | 92 | 46 | 80 | 92 | |
| ALOHA sim, 5k steps, 50 episodes | 48 | 4 | 16 | 32 | r128 plus = 34 |
| ALOHA sim, 20k steps, 100 episodes | 82 | – | – | 41 | r256 = 54, r512 = 58, r128 plus = 48 |
| RoboTwin Aloha-AgileX, 5k, 100 episodes | 70 | 50 | 52 | 70 | |
| RoboTwin UR5, 5k, 100 episodes | 56 | – | 50 | 67 | |
| RoboTwin Franka, 5k, 100 episodes | 35 | – | 22 | 22 | |

What the results show so far:
- **Where r128 is enough:** on LIBERO, RoboTwin Aloha-AgileX and RoboTwin UR5, C r128 matches or beats A.
- **Where a gap remains:** on ALOHA sim and RoboTwin Franka, C trails A.
  - A larger rank doesn't close it: all ALOHA ranks end at the same training loss.
  - A higher learning rate doesn't close it either: `_vision_r128_plus` reaches a lower training loss than A (0.0020 vs 0.0025 over the last 5k steps) but scores 48 vs 82.

## Known deviations from the standard setups

- **Fewer evaluation episodes:** LIBERO uses 5 trials per task (openpi uses 50), and RoboTwin uses 20 episodes per task (RoboTwin uses 100).
- **ALOHA sim episode cap:** 400 steps instead of gym-aloha's registered 300, because the human demos are about 400 steps long.
- **xArm episode cap:** 50 steps, the Lift task's own `episode_length`, rather than the 300 gym-xarm registers. Its demos are 25 steps long.

- **RoboCasa and GR-1 protocol:** NVIDIA's GR00T settings (8 of each 16 predicted actions executed, 720-step limit, episode ends on success), but 5 of each benchmark's 24 tasks, chosen as the ones GR00T solves most often, and 20 episodes per task. RoboCasa's five include no pick-and-place task.
- **BiGym episode limit:** BiGym sets none, so each task's limit is 1.5x its longest training demo. All five tasks are closing tasks.
- **GR-1 instead of DexMimicGen:** the planned top-of-range robot was the GR-1 in DexMimicGen. The GR-1 tabletop benchmark has the same robot, public demos in LeRobot layout and a public scoring setup, so it is used instead.

## Checks

- **Franka and UR5 arm spacing:** RoboTwin pairs single arms 0.8 m apart. We inferred that value from the demos' starting end-effector positions. `check_arm_spacing.py` confirmed it: on all three arms, a reset simulator matches the first frame of a click_bell demo in joints and end-effector positions (differences below 0.0001).
