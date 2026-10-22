"""Input/output transforms for RoboTwin 2.0 bimanual arms (aloha-agilex, franka, ur5, ...).

Matches datasets converted by trinity/convert_robotwin.py:
  image / left_wrist_image / right_wrist_image   head camera, left and right wrist cameras
  state / actions                                 RoboTwin's joint vector, absolute positions:
      left arm joints, left gripper, right arm joints, right gripper (6 + 1 + 6 + 1 = 14, franka 7 + 1 + 7 + 1 = 16)
Used for training and inference alike; the RoboTwin eval client sends the same keys.
"""
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """LeRobot hands training images as float CHW; the model wants uint8 HWC."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RoboTwinInputs(transforms.DataTransformFn):
    """Robot observation -> model inputs. All three pi0.5 camera slots are real views."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": _parse_image(data["observation/image"]),  # head camera
                "left_wrist_0_rgb": _parse_image(data["observation/left_wrist_image"]),
                "right_wrist_0_rgb": _parse_image(data["observation/right_wrist_image"]),
            },
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.True_},
        }
        if "actions" in data:  # only present during training
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class RoboTwinOutputs(transforms.DataTransformFn):
    """Model outputs -> robot commands: keep the arm's real dims, drop the model's padding up to 32."""

    action_dim: int  # 2 * (arm joints + 1 gripper)

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
