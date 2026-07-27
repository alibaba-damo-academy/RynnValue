import os
import argparse
import re

import numpy as np
import torch

from datetime import datetime

import imageio.v2 as imageio
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModel,
    AutoProcessor,
)

from plot_utils import save_video_with_trend


def build_output_path(args):
    model_name = os.path.basename(args.model_path.rstrip("/")) or "model"
    sample_name = os.path.splitext(os.path.basename(args.video_path.rstrip("/")))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_name = f"{model_name}_{sample_name}_{timestamp}"
    return os.path.join(args.output_path, output_name)


def parse_args():
    parser = argparse.ArgumentParser(description="RynnValue Inference on a self-constructed sample")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HuggingFace model directory or repo id.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the input video file to run inference on.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Task instruction describing what the agent should accomplish.",
    )
    parser.add_argument(
        "--robot_description",
        type=str,
        default=None,
        help="Robot embodiment description for the meta block (required when the model was trained with use_meta=True).",
    )
    parser.add_argument(
        "--camera_description",
        type=str,
        default=None,
        help="Camera viewpoint description for the meta block (required when the model was trained with use_meta=True).",
    )
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--mixed_precision", action="store_true", default=True)
    parser.add_argument(
        "--num_frames",
        type=int,
        default=64,
        help="Number of frames uniformly resampled from each prefix sub-sample.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=0,
        help="Number of prefix steps evaluated, uniformly spaced over the video (0 = every frame, default).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of prefix sub-samples per GPU forward pass.",
    )
    parser.add_argument(
        "--max_image_side",
        type=int,
        default=640,
        help="Resize frames so their longer side is at most this before feeding the model (0 = no resize).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum tokens generated for the Analysis (description/match/success) block.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the rendered trend video.",
    )
    return parser.parse_args()


def load_video_frames(video_path):
    """Decode every frame of a video file into a list of RGB PIL images."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"video_path is not a file: {video_path}")
    reader = imageio.get_reader(video_path)
    try:
        frames = [Image.fromarray(frame).convert("RGB") for frame in reader]
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"No frames decoded from video: {video_path}")
    return frames


def resize_frames(frames, max_side):
    """Downscale frames so the longer side is at most ``max_side`` (keeps aspect ratio)."""
    if max_side <= 0:
        return frames
    w, h = frames[0].size
    if max(w, h) <= max_side:
        return frames
    scale = max_side / max(w, h)
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return [f.resize(new_size, resample=Image.BICUBIC) for f in frames]


def sample_frame_indices(total, num_frames):
    """Uniformly pick ``num_frames`` indices from ``total`` frames."""
    if num_frames <= 0 or num_frames >= total:
        return list(range(total))
    if num_frames == 1:
        return [total - 1]
    step = (total - 1) / (num_frames - 1)
    return [int(round(j * step)) for j in range(num_frames)]


_DESCRIPTION_RE = re.compile(r"-\s*Video Description:\s*(.+)", re.IGNORECASE)
_MATCH_RE = re.compile(r"-\s*Match:\s*(Yes|No)", re.IGNORECASE)
_SUCCESS_RE = re.compile(r"-\s*Success:\s*(Yes|No)", re.IGNORECASE)


def parse_analysis(text: str):
    """Extract description / match / success from the generated Analysis block.

    The training layout emits:
        - Video Description: ...
        - Match: Yes/No
        - Success: Yes/No
    Any field that the model omitted returns as None.
    """
    def _first(pattern):
        m = pattern.search(text)
        return m.group(1).strip() if m else None

    return {
        "description": _first(_DESCRIPTION_RE),
        "match": _first(_MATCH_RE),
        "success": _first(_SUCCESS_RE),
    }


def main():
    args = parse_args()
    output_path = build_output_path(args)
    os.makedirs(output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.mixed_precision else torch.float32
    print("Loading model...")
    # This already-exported model bundles an older config (no attn default) and
    # config.json doesn't persist it, so force the custom prediction-slot
    # isolation attention. Load onto a single device (no device_map sharding) so
    # the forward pass doesn't mix tensors across GPUs.
    hf_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    hf_config._attn_implementation = "pred_slot_isolated_eager"
    model = AutoModel.from_pretrained(
        args.model_path,
        config=hf_config,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    model = model.to(device=device, dtype=dtype)
    model.eval()

    tokenizer = processor.tokenizer
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Construct a single sample from the user-provided video + instruction.
    print(f"Loading video: {args.video_path}")
    images = load_video_frames(args.video_path)
    instruction = args.instruction
    total = len(images)
    print(f"Loaded {total} frames; instruction: {instruction!r}")

    # Model inputs use downscaled frames; the rendered trend video keeps the
    # original resolution.
    model_images = resize_frames(images, args.max_image_side)
    if model_images is not images:
        print(f"Resized frames from {images[0].size} to {model_images[0].size} for the model")

    # Prefix-uniform sampling (mirrors robometer's compute_episode_progress):
    # for each evaluated step i, the prefix frames[0:i+1] is resampled to
    # num_frames via linspace and one value (the last prediction slot) is read
    # out per prefix, so each score only conditions on frames seen so far.
    eval_indices = sample_frame_indices(total, args.num_steps)
    print(
        f"Prefix-uniform sampling: {len(eval_indices)} steps, "
        f"{args.num_frames} frames per prefix, batch_size={args.batch_size}"
    )

    def build_prefix_sample(end_idx):
        frame_idx = np.linspace(0, end_idx, args.num_frames, dtype=int)
        prefix_images = [model_images[j] for j in frame_idx]
        return processor.process_episode(
            instruction=instruction,
            images=prefix_images,
            robot_description=args.robot_description,
            camera_description=args.camera_description,
        )

    def run_batch(samples):
        """One forward pass over a batch of prefix sub-samples; returns the
        last-slot value per sample (remaining time at the prefix end)."""
        batch_kwargs = dict(
            input_ids=torch.cat([s["input_ids"] for s in samples], dim=0).to(device).long(),
            attention_mask=torch.cat([s["attention_mask"] for s in samples], dim=0).to(device).long(),
            pixel_values=torch.cat(
                [s["pixel_values"].flatten(0, 1) for s in samples], dim=0
            ).to(device),
            image_grid_thw=torch.cat(
                [s["image_grid_thw"].flatten(0, 1) for s in samples], dim=0
            ).to(device).long(),
        )
        with torch.inference_mode():
            outputs = model(**batch_kwargs)
        pred = outputs.value.pred_value
        if pred.dim() == 2 and pred.shape[0] == 1:
            pred = pred.reshape(len(samples), -1)
        if pred.dim() == 3:
            pred = pred.mean(dim=0)
        if pred.dim() == 2 and pred.shape[-1] > 1:
            pred = pred[:, -1]
        elif pred.dim() == 2:
            pred = pred[:, 0]
        return pred.float().reshape(-1).tolist()

    pred_value = []
    final_sample = None
    batch = []
    for step, end_idx in enumerate(eval_indices):
        sample = build_prefix_sample(end_idx)
        if end_idx == eval_indices[-1]:
            final_sample = sample
        batch.append(sample)
        if len(batch) >= args.batch_size or step == len(eval_indices) - 1:
            pred_value.extend(run_batch(batch))
            batch = []
            print(f"  progress: {len(pred_value)}/{len(eval_indices)} steps")

    # Analysis pass on the final prefix (the full video uniformly sampled).
    input_ids = final_sample["input_ids"].to(device).long()
    with torch.inference_mode():
        gen_out = model.generate(
            input_ids=input_ids,
            attention_mask=final_sample["attention_mask"].to(device).long(),
            pixel_values=final_sample["pixel_values"].flatten(0, 1).to(device),
            image_grid_thw=final_sample["image_grid_thw"].flatten(0, 1).to(device).long(),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
            use_cache=True,
        )

    generated_ids = gen_out[0, input_ids.shape[1]:]
    analysis_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    analysis = parse_analysis(analysis_text)

    print(f"instruction: {instruction}")
    print(f"  generated: {analysis_text!r}")
    print(f"  description: {analysis['description']}")
    print(f"  match:       {analysis['match']}")
    print(f"  success:     {analysis['success']}")

    out_file = os.path.join(output_path, "output_with_trend.mp4")
    save_video_with_trend(
        images=images,
        value=pred_value,
        output_path=out_file,
        fps=args.fps,
        title="Remaining Time (s)",
        task_title=instruction,
        sampled_indices=eval_indices,
    )
    print(f"Saved {out_file}")

    print("Inference complete.")


if __name__ == "__main__":
    main()
