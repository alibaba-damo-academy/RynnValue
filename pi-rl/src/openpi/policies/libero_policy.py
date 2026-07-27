# adapted from openpi
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.shared import image_tools


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class LiberoRLInputs(transforms.DataTransformFn):
    """LiberoInputs variant for offline RL training.

    The RL data config asks lerobot for state/image/wrist_image at offsets
    ``[0, H/fps]`` and (when present) ``progress`` at ``[0, 1/fps, ..., H/fps]``
    via ``delta_timestamps``, so each of those fields arrives as a 2-frame
    stack ``[current, next]`` (progress as ``H+1``). We split them, parse the
    images, pack the current view in standard pi0 format, expose next-step
    pixels/state as side-channel keys, and derive reward+mask from progress.

    Next-step images are resized here (the standard ``ResizeImages`` model
    transform only touches ``data["image"]``, so without this the next-step
    pixels would skip resizing and end up shape-mismatched against the IQL
    encoder).
    """

    model_type: _model.ModelType
    # Match the standard ResizeImages step in ModelTransformFactory.
    resize_height: int = 224
    resize_width: int = 224

    def __call__(self, data: dict) -> dict:
        base_pair = np.asarray(data["observation/image"])
        wrist_pair = np.asarray(data["observation/wrist_image"])
        state_pair = np.asarray(data["observation/state"])

        cur_base = _parse_image(base_pair[0])
        cur_wrist = _parse_image(wrist_pair[0])
        nxt_base = _parse_image(base_pair[1])
        nxt_wrist = _parse_image(wrist_pair[1])

        # Pre-resize the next-step pixels so they match the trained encoder
        # input size (the standard ResizeImages further down the chain only
        # touches data["image"]).
        nxt_base = np.asarray(image_tools.resize_with_pad(nxt_base, self.resize_height, self.resize_width))
        nxt_wrist = np.asarray(image_tools.resize_with_pad(nxt_wrist, self.resize_height, self.resize_width))

        inputs: dict = {
            "state": np.asarray(state_pair[0], dtype=np.float32),
            "image": {
                "base_0_rgb": cur_base,
                "left_wrist_0_rgb": cur_wrist,
                "right_wrist_0_rgb": np.zeros_like(cur_base),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
            # RL side-channel keys (the pi0 model never sees these).
            "next_state": np.asarray(state_pair[1], dtype=np.float32),
            "next_image_base": nxt_base,
            "next_image_wrist": nxt_wrist,
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "actions_is_pad" in data:
            inputs["actions_is_pad"] = np.asarray(data["actions_is_pad"]).reshape(-1).astype(bool)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "progress" in data:
            # progress sampled at [t, t+1, ..., t+H] (H+1 values aligned with
            # the action chunk + endpoint). Expose the raw sequence; the IQL
            # trainer reads ``batch["progress"]`` and computes the PBRS shaped
            # chunk reward R = Σ γ^h · (γ·Φ(s_{h+1}) - Φ(s_h)) itself, so we
            # don't precompute per-step deltas here. The scalar ``reward``
            # (endpoint delta) is still emitted as a human-readable debug
            # field and as the terminal-mode fallback in _chunk_reward.
            progress = np.asarray(data["progress"], dtype=np.float32).reshape(-1)
            inputs["progress"] = progress
            inputs["reward"] = np.float32(progress[-1] - progress[0])
            is_pad = data.get("progress_is_pad")
            if is_pad is not None:
                is_pad = np.asarray(is_pad).reshape(-1).astype(bool)
                inputs["progress_is_pad"] = is_pad
                terminal = bool(is_pad[-1])
            else:
                terminal = False
        else:
            # No progress feature: fall back to terminal reward, using the
            # state pad flag to detect when the next frame ran off the episode.
            is_pad = data.get("observation/state_is_pad")
            terminal = bool(np.asarray(is_pad).reshape(-1)[1]) if is_pad is not None else False
            inputs["reward"] = np.float32(0.0 if terminal else -1.0)
        inputs["mask"] = np.float32(0.0 if terminal else 1.0)
        return inputs
