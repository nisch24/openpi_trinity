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
| `src/openpi/training/config.py` | All Trinity configs (below), plus the data configs `LeRobotRoboTwinDataConfig` and the RB-Y1 one. |
| `src/openpi/policies/robotwin_policy.py` | New: RoboTwin input/output transforms. Three cameras map to π0.5's image slots; outputs are cut back to the arm's 14 or 16 action dimensions. |
| `src/openpi/policies/rby1_policy.py` | New: RB-Y1 input/output transforms. |
| `src/openpi/training/data_loader.py` | Decodes video datasets with PyAV, because PACE nodes lack the FFmpeg libraries LeRobot's default decoder needs. Image datasets are unaffected. |
| `src/openpi/serving/websocket_policy_server.py` | Turns off websocket keepalive pings. A simulator client can block for more than 20 s between requests (RoboTwin's planner) and would otherwise be disconnected mid-evaluation. |

With `vision_lora_rank=0`, every existing openpi config behaves as before.

## Configs added

| Robot | Configs | Data | Default steps |
|---|---|---|---|
| LIBERO (1 Franka arm) | `pi05_libero_lora{,_strict,_vision,_vision_r128}` | `physical-intelligence/libero`, norm stats from the published `pi05_libero` | 30k (runs stopped at 5k) |
| ALOHA sim (2 arms, transfer cube) | `pi05_aloha_sim_lora{,_strict,_vision,_vision_r128,_vision_r256,_vision_r512,_vision_r128_plus}` | `lerobot/aloha_sim_transfer_cube_human` (50 human demos) | 20k |
| RoboTwin 2.0 (Aloha-AgileX, Franka, UR5) | `pi05_robotwin_{aloha,franka,ur5}_lora{,_strict,_vision,_vision_r128}` | `trinity/robotwin_<arm>`, from `convert_robotwin.py` | 20k (runs stopped at 5k) |
| RB-Y1 (real robot, no simulator) | `pi05_rby1`, `pi05_rby1_lora{,_strict,_vision,_vision_r64,_vision_r128}` | `trinity/<Rainbow dataset>`, from `convert_rby1.py` | |

How RoboTwin is set up:
- **State and action:** the state is the joint vector at frame t; the action is the joint vector at frame t+1.
- **Action encoding:** arm joints are learned as deltas from the current state; grippers are absolute (as in openpi's ALOHA config).
- **Chunks:** 15-step action chunks (1 s at 15 fps).
- **Data:** five tasks with 50 clean demos each: beat_block_hammer, click_bell, move_can_pot, place_empty_cup, press_stapler.
- **Images:** the converter undoes the BGR channel order in RoboTwin's older JPEGs.

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
| `convert_rby1.py`, `.sbatch`, `check_rby1.py`, `eval_rby1.py`, `.sbatch` | RB-Y1 conversion with a held-out split, and action-error scoring (no simulator). |
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
| ALOHA sim, 5k steps, 50 episodes | 48 | 4 | 16 | 32 | |
| ALOHA sim, 20k steps, 100 episodes | 82 | – | – | 41 | r256 = 54, r512 = 58 |
| RoboTwin Aloha-AgileX, 5k, 100 episodes | 70 | 50 | 52 | 70 | |
| RoboTwin UR5, 5k, 100 episodes | 56 | – | 50 | 67 | |
| RoboTwin Franka, 5k, 100 episodes | 35 | – | 22 | 22 | |

What the results show so far:
- **Where r128 is enough:** on LIBERO, RoboTwin Aloha-AgileX and RoboTwin UR5, C r128 matches or beats A.
- **Where a gap remains:** on ALOHA sim and RoboTwin Franka, C trails A.
  - A larger rank doesn't close it: all ALOHA ranks end at the same training loss.
  - `_vision_r128_plus` tests whether the learning rate is the cause.

## Known deviations from the standard setups

- **Fewer evaluation episodes:** LIBERO uses 5 trials per task (openpi uses 50), and RoboTwin uses 20 episodes per task (RoboTwin uses 100).
- **ALOHA sim episode cap:** 400 steps instead of gym-aloha's registered 300, because the human demos are about 400 steps long.
- **Franka and UR5 arm spacing:** RoboTwin pairs single arms 0.8 m apart. We inferred that value from the demos' starting end-effector positions; it is not yet verified against RoboTwin's collection settings.
