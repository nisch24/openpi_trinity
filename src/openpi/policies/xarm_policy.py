"""Input/output transforms for the simulated xArm lifting a cube (gym-xarm Lift).

Matches lerobot/xarm_lift_medium (read as LeRobot v2.1) and trinity/eval_xarm_sim.py:
  image    one 84x84 camera ("camera0"); pi0.5's two wrist slots stay empty and masked
  state    4 numbers: hand position x, y, z and gripper angle
  actions  4 numbers in [-1, 1]: hand move in x, y, z and gripper command (already relative moves)
Used for training and inference alike; the eval client sends the same keys.
"""
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

ACTION_DIM = 4  # x, y, z move + gripper


def _parse_image(image) -> np.ndarray:
    """LeRobot hands training images as float CHW; the model wants uint8 HWC."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class XarmInputs(transforms.DataTransformFn):
    """Robot observation -> model inputs. Only the base camera slot holds a real view."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        # Empty wrist slots are masked for pi0 / pi0.5 (pi0-FAST expects them unmasked, as in LIBERO)
        wrist_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": wrist_mask, "right_wrist_0_rgb": wrist_mask},
        }
        if "actions" in data:  # only present during training
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class XarmOutputs(transforms.DataTransformFn):
    """Model outputs -> robot commands: keep the 4 real dims, drop the model's padding up to 32."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}
