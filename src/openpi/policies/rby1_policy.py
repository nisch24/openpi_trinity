"""Input/output transforms for the Rainbow Robotics RB-Y1 (bimanual arms + grippers).

Matches datasets converted by trinity/convert_rby1.py:
  image / left_wrist_image / right_wrist_image   three cameras (front, left, right)
  state / actions                                 16 numbers in this order:
      right arm joints 0-6, left arm joints 0-6, right gripper, left gripper
Used for training and inference alike; at inference a robot client sends the same keys.
"""
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

ACTION_DIM = 16  # 7 + 7 arm joints, 2 grippers


def make_rby1_example() -> dict:
    """A random observation in the format an RB-Y1 client sends to the policy server."""
    return {
        "observation/state": np.random.rand(ACTION_DIM).astype(np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the object",
    }


def _parse_image(image) -> np.ndarray:
    """LeRobot hands training images as float CHW; the model wants uint8 HWC."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RBY1Inputs(transforms.DataTransformFn):
    """Robot observation -> model inputs. All three pi0.5 camera slots are real views on the RB-Y1."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": _parse_image(data["observation/image"]),  # front camera
                "left_wrist_0_rgb": _parse_image(data["observation/left_wrist_image"]),
                "right_wrist_0_rgb": _parse_image(data["observation/right_wrist_image"]),
            },
            # No slot is padding here, so every camera is marked as present
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.True_},
        }
        if "actions" in data:  # only present during training
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class RBY1Outputs(transforms.DataTransformFn):
    """Model outputs -> robot commands: keep the 16 real dims, drop the model's padding up to 32."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}
