#!/usr/bin/env python3
"""
RynnValue baseline for progress reward prediction.

RynnValue predicts the remaining time for task completion using a Qwen3VL-based
model with an ensemble of value heads. The model also supports optional success
prediction via a binary classification head.

Official model: https://huggingface.co/Alibaba-DAMO-Academy/RynnValue-8B
"""


from typing import List, Optional, Tuple, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from transformers import (
    AutoConfig,
    AutoModel,
    AutoProcessor,
    PreTrainedModel,
    ProcessorMixin
)
from accelerate import init_empty_weights
import numpy as np
import torch
import os
import re

from robometer.configs.constants import DATA_SOURCE_ROBOT_DESCRIPTION, DATA_SOURCE_CAMERA_DESCRIPTION
from robometer.data.collators.utils import convert_frames_to_pil_images
from robometer.utils.logger import get_logger

logger = get_logger()


# Analysis-block parsing (mirrors rynn_infer/infer_value.py). The model emits,
# after the frames/value tokens:
#     - Video Description: ...
#     - Match: Yes/No
#     - Success: Yes/No
_DESCRIPTION_RE = re.compile(r"-\s*Video Description:\s*(.+)", re.IGNORECASE)
_MATCH_RE = re.compile(r"-\s*Match:\s*(Yes|No)", re.IGNORECASE)
_SUCCESS_RE = re.compile(r"-\s*Success:\s*(Yes|No)", re.IGNORECASE)


def parse_analysis(text: str) -> dict:
    """Extract description / match / success from the generated Analysis block.

    Any field the model omitted returns as None.
    """
    def _first(pattern):
        m = pattern.search(text)
        return m.group(1).strip() if m else None

    return {
        "description": _first(_DESCRIPTION_RE),
        "match": _first(_MATCH_RE),
        "success": _first(_SUCCESS_RE),
    }

def fuse_td_lambda(pred_value, pred_relative, lam=0.5):
    """Bidirectional TD(lambda) fusion of the absolute and relative value heads.

    ``pred_value`` (``a``, length N) is the per-frame remaining time; ``pred_relative``
    (``r``, length N-1) is the per-step elapsed time with ``r[i] ~= a[i] - a[i+1]``.

    A backward lambda-return anchored on the terminal value ``a[-1]`` and a forward
    lambda-return anchored on the initial value ``a[0]`` are averaged so that the
    integration drift of each direction is cancelled near the opposite end.
    ``lam=0`` leans on the absolute head one step away; ``lam=1`` integrates the
    relative head from each anchor.
    """
    a = list(pred_value)
    n = len(a)
    if pred_relative is None or n < 2:
        return a
    r = list(pred_relative)
    if len(r) != n - 1:
        print(
            f"[fuse_td_lambda] skip: len(pred_relative)={len(r)} != N-1={n - 1}"
        )
        return a

    # Backward pass, anchored on the terminal value a[-1].
    backward = [0.0] * n
    backward[n - 1] = a[n - 1]
    for i in range(n - 2, -1, -1):
        backward[i] = r[i] + (1.0 - lam) * a[i + 1] + lam * backward[i + 1]

    # Forward pass, anchored on the initial value a[0]. Direction reversed => -r.
    forward = [0.0] * n
    forward[0] = a[0]
    for i in range(1, n):
        forward[i] = -r[i - 1] + (1.0 - lam) * a[i - 1] + lam * forward[i - 1]

    return [0.5 * (backward[i] + forward[i]) for i in range(n)]

class RynnValue:
    """RynnValue baseline for progress reward prediction.

    Predicts remaining time (in seconds) until task completion using a
    Qwen3VL-based model with an ensemble of value heads. Optionally produces
    success probabilities when the model includes a success head.
    """

    def __init__(
        self,
        model_path: str = "Alibaba-DAMO-Academy/RynnValue-8B",
        checkpoint_path: Optional[str] = None,
        stride: int = 1,
        num_frames: int = 8,
        mode: str = "absolute",
        camera_desc_lookup_path: Optional[str] = None,
        max_new_tokens: int = 128,
        confusion_score_mode: str = "match_binary",
        use_fuse: bool = False,
        fuse_lambda: float = 0.5,
        attn_implementation: Optional[str] = None,
    ):
        if confusion_score_mode not in ("match_binary", "normalized_value"):
            raise ValueError(
                f"Unknown confusion_score_mode: {confusion_score_mode}. "
                "Expected 'match_binary' or 'normalized_value'."
            )

        model, processor = self._load_model(model_path, checkpoint_path, attn_implementation)

        self.model = model
        self.processor = processor
        self.stride = stride
        self.num_frames = num_frames
        self.mode = mode
        self.max_new_tokens = max_new_tokens
        self.confusion_score_mode = confusion_score_mode
        self.use_fuse = use_fuse
        self.fuse_lambda = fuse_lambda

        # Per-trajectory camera description lookup (id -> camera_description)
        self._camera_desc_lookup = self._load_camera_desc_lookup(camera_desc_lookup_path)

        # Filled after each compute_progress call for downstream consumers
        self.last_success_probs: Optional[List[float]] = None

        logger.info(f"RynnValue model loaded on device: {self.model.device}")
        logger.info(f"  stride={self.stride}, num_frames={self.num_frames}, mode={mode}, "
                    f"max_new_tokens={self.max_new_tokens}, confusion_score_mode={self.confusion_score_mode}, "
                    f"use_fuse={self.use_fuse}, fuse_lambda={self.fuse_lambda}, "
                    f"attn_implementation={attn_implementation or 'model-default'}")
        if self._camera_desc_lookup:
            logger.info(f"  camera_desc_lookup: {len(self._camera_desc_lookup)} entries")

    # ------------------------------------------------------------------
    # Camera description lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _load_camera_desc_lookup(path: Optional[str]) -> dict:
        """Load id -> camera_description mapping from a JSON file."""
        if not path or not os.path.exists(path):
            return {}
        import json
        with open(path) as f:
            entries = json.load(f)
        return {
            entry["id"]: entry["camera_description"]
            for entry in entries
            if entry.get("id") and entry.get("camera_description")
        }

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(
        self,
        model_path: str,
        checkpoint_path: Optional[str],
        attn_implementation: Optional[str],
    ) -> Tuple[PreTrainedModel, ProcessorMixin]:
        if checkpoint_path is not None:
            return self._load_from_checkpoint(checkpoint_path, attn_implementation)
        return self._load_from_hf(model_path, attn_implementation)

    @staticmethod
    def _load_from_hf(
        model_path: str, attn_implementation: Optional[str]
    ) -> Tuple[PreTrainedModel, ProcessorMixin]:
        logger.info(f"Loading RynnValue model from HuggingFace: {model_path}")

        # Without an override the exported config self-selects the custom
        # pred_slot_isolated_eager attention.
        hf_kwargs = {}
        if attn_implementation:
            hf_kwargs["attn_implementation"] = attn_implementation

        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **hf_kwargs,
        )
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        model = model.to(device="cuda", dtype=torch.bfloat16)
        return model, processor

    @staticmethod
    def _load_from_checkpoint(
        checkpoint_path: str, attn_implementation: Optional[str]
    ) -> Tuple[PreTrainedModel, ProcessorMixin]:
        """Load from a fine-tuned checkpoint directory containing ``model.pt``.

        HF artifacts (config, tokenizer, processor, custom code) are expected in
        a sibling ``huggingface/`` directory relative to *checkpoint_path*.
        """
        logger.info(f"Loading RynnValue model from checkpoint: {checkpoint_path}")

        hf_artifacts_path = Path(checkpoint_path).parent / "huggingface"

        hf_config = AutoConfig.from_pretrained(hf_artifacts_path, trust_remote_code=True)
        # Without an override the config class defaults to (and persists) the
        # custom pred_slot_isolated_eager attention.
        if attn_implementation:
            hf_config._attn_implementation = attn_implementation
        processor = AutoProcessor.from_pretrained(
            hf_artifacts_path, trust_remote_code=True
        )

        with init_empty_weights():
            model = AutoModel.from_config(hf_config, trust_remote_code=True)

        state_dict = torch.load(
            os.path.join(checkpoint_path, "model.pt"),
            map_location="cuda",
        )

        text_config = getattr(hf_config, "text_config", hf_config)
        tie_word_embeddings = getattr(text_config, "tie_word_embeddings", False)

        if tie_word_embeddings:
            # 4B variant: lm_head is tied to embed_tokens and may be saved as a 1D FSDP shard
            _lm_key = "lm_head.weight"
            if _lm_key in state_dict and state_dict[_lm_key].dim() == 1:
                state_dict.pop(_lm_key)

        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
        if tie_word_embeddings:
            model.tie_weights()
        if missing:
            logger.warning(f"Missing keys when loading checkpoint: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading checkpoint: {unexpected}")

        model = model.to(device="cuda", dtype=torch.bfloat16)
        return model, processor

    # ------------------------------------------------------------------
    # Video frame loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_video_frames(video_path: str, max_frames: int = 64) -> Optional[np.ndarray]:
        """Load frames from a video file using decord.

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract

        Returns:
            numpy array of shape (T, H, W, C), or None on error
        """
        try:
            import decord
        except ImportError:
            logger.error("decord is required for video input. Install with: pip install decord")
            return None

        if not os.path.exists(video_path):
            logger.warning(f"Video path does not exist: {video_path}")
            return None

        try:
            vr = decord.VideoReader(video_path, num_threads=1)
            total_frames = len(vr)
            if total_frames == 0:
                return None

            if total_frames <= max_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()

            frames_array = vr.get_batch(frame_indices).asnumpy()
            del vr
            return frames_array
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def compute_episode_progress(
        self,
        frames_array: Union[np.ndarray, str],
        task_description: str = "",
        metadata: Optional[dict] = None,
        batch_size: int = 4,
        max_workers: int = 32,
    ) -> List[float]:
        """Compute per-frame-step progress for an entire episode.

        Constructs T sub-samples (one per frame in the episode). For frame step
        i, takes frames[0:i+1] and subsamples to ``self.num_frames`` via
        linspace. Preprocessing is parallelized across threads and inference is
        batched on GPU.

        Args:
            frames_array: (T, H, W, 3) uint8 array of trajectory frames, or a
                path to a video file (mp4, avi, etc.).
            task_description: Natural-language task description.
            metadata: Trajectory metadata dict for resolving descriptions.
            batch_size: Number of sub-samples per GPU forward pass.
            max_workers: Number of threads for parallel preprocessing.

        Returns:
            List of T progress scores (one per frame step), or empty list on
            failure.
        """
        if isinstance(frames_array, str):
            frames_array = self._load_video_frames(frames_array)

        if frames_array is None or frames_array.size == 0:
            return []

        robot_desc, camera_desc = self._resolve_descriptions(metadata)

        total_frames = len(frames_array)

        logger.info(f"RynnValue: computing episode progress for {total_frames} frame steps "
                     f"(num_frames={self.num_frames}, batch_size={batch_size}, workers={max_workers})")

        def _preprocess(i: int) -> Optional[dict]:
            indices = np.linspace(0, i - 1, self.num_frames, dtype=int)
            sub_frames = frames_array[indices]

            frames_pil = convert_frames_to_pil_images(sub_frames)
            if not frames_pil:
                return None

            processed = self.processor.process_episode(
                instruction=task_description,
                images=frames_pil,
                robot_description=robot_desc,
                camera_description=camera_desc,
            )
            return processed

        # Batched GPU forward pass
        device = self.model.device
        progress_scores: List[Optional[float]] = [None] * total_frames

        def _run_batch(batch_items: List[Tuple[int, dict]]) -> None:
            batch_idx = [idx for idx, _ in batch_items]
            batch_processed = [p for _, p in batch_items]

            # Collate batch
            input_ids = torch.cat([p["input_ids"] for p in batch_processed], dim=0).to(device).long()
            attention_mask = torch.cat([p["attention_mask"] for p in batch_processed], dim=0).to(device).long()

            model_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)

            pixel_values_list = [p.get("pixel_values") for p in batch_processed]
            if pixel_values_list[0] is not None:
                model_kwargs["pixel_values"] = torch.cat(
                    [pv.flatten(0, 1) for pv in pixel_values_list], dim=0
                ).to(device)

            image_grid_thw_list = [p.get("image_grid_thw") for p in batch_processed]
            if image_grid_thw_list[0] is not None:
                model_kwargs["image_grid_thw"] = torch.cat(
                    [g.flatten(0, 1) for g in image_grid_thw_list], dim=0
                ).to(device).long()

            pixel_values_videos_list = [p.get("pixel_values_videos") for p in batch_processed]
            if pixel_values_videos_list[0] is not None:
                model_kwargs["pixel_values_videos"] = torch.cat(
                    [pv.flatten(0, 1) for pv in pixel_values_videos_list], dim=0
                ).to(device)

            video_grid_thw_list = [p.get("video_grid_thw") for p in batch_processed]
            if video_grid_thw_list[0] is not None:
                model_kwargs["video_grid_thw"] = torch.cat(
                    [g.flatten(0, 1) for g in video_grid_thw_list], dim=0
                ).to(device).long()

            with torch.inference_mode():
                outputs = self.model(**model_kwargs)

            pred_value = outputs.value.pred_value
            if pred_value.dim() == 2 and pred_value.shape[0] == 1:
                pred_value = pred_value.reshape(len(batch_idx), -1)

            if pred_value.dim() == 3:
                pred_value = pred_value.mean(dim=0)
            if pred_value.dim() == 2 and pred_value.shape[-1] > 1:
                pred_value = pred_value[:, -1]
            elif pred_value.dim() == 2:
                pred_value = pred_value[:, 0]

            pred_value = self._apply_mode(pred_value)

            for j, idx in enumerate(batch_idx):
                progress_scores[idx] = pred_value[j].item()

        # Bounded-prefetch pipeline: cap in-flight preprocessing so host memory
        # stays O(window) instead of O(total_frames). Workers refill the window
        # as the GPU drains each batch, preserving CPU/GPU overlap.
        window = batch_size * 2
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            frame_iter = iter(range(total_frames))
            inflight: dict = {}

            def _submit_next() -> bool:
                idx = next(frame_iter, None)
                if idx is None:
                    return False
                inflight[executor.submit(_preprocess, idx + 1)] = idx
                return True

            for _ in range(window):
                if not _submit_next():
                    break

            pending_batch: List[Tuple[int, dict]] = []
            while inflight:
                done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                for fut in done:
                    idx = inflight.pop(fut)
                    processed = fut.result()
                    _submit_next()
                    if processed is None:
                        continue
                    pending_batch.append((idx, processed))
                    if len(pending_batch) >= batch_size:
                        _run_batch(pending_batch)
                        pending_batch = []

            if pending_batch:
                _run_batch(pending_batch)

        logger.info(f"RynnValue: episode progress computed "
                     f"(first={progress_scores[0]:.3f}, last={progress_scores[-1]:.3f})")

        return progress_scores

    def compute_progress(
        self,
        frames_array: Union[np.ndarray, str],
        task_description: str = "",
        reference_video_path: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Tuple[List[Optional[float]], Optional[dict]]:
        """Compute per-frame progress predictions using RynnValue.

        The model predicts remaining time (in seconds) at each frame using an
        ensemble of value heads (averaged across heads and transformed per
        ``self.mode``). A second greedy ``generate`` pass over the same prompt
        produces the Analysis block, parsed for description / match / success
        (see :func:`parse_analysis`, mirroring ``rynn_infer/infer_value.py``).
        When the model reports ``Match: No`` the video does not match the task,
        so the progress scores are nulled out to ``-100.0``.

        Confusion-matrix samples are detected by the presence of both
        ``lang_task`` and ``video_task`` in ``metadata`` (set by
        ``ConfusionMatrixSampler``). For those, ``self.confusion_score_mode``
        selects a bounded score instead of value-head progress:

        - ``"match_binary"`` (default): ``1.0`` when the Analysis reports
          ``Match: Yes`` else ``0.0``. Isolates task<->video matching from task
          success (a matched-but-failed clip is still ``Match: Yes`` -> ``1.0``).
        - ``"normalized_value"``: when the Analysis reports ``Match: Yes`` the
          value head normalized to ``[0, 1]`` (``1 - t/t_max``); otherwise
          (``Match: No`` or no verdict) the score is ``0.0``.

        Both keep the score in ``[0, 1]`` (comparable to other baselines) and
        avoid the ``-100`` sentinel that craters cell means.

        Args:
            frames_array: (N, H, W, 3) uint8 array of trajectory frames, or a
                path to a video file (mp4, avi, etc.).
            task_description: Natural-language task description.
            reference_video_path: Unused; kept for API compatibility.
            metadata: Trajectory metadata dict. ``data_source`` is extracted
                from it to resolve robot_type and camera_pos. When it carries
                both ``lang_task`` and ``video_task`` the confusion-matrix
                match-score path above is used.

        Returns:
            Tuple of (progress_scores, analysis):
                progress_scores: List of per-frame progress scores (one per input
                    frame); overridden to ``[-100.0] * N`` when ``Match: No``, or
                    to a confusion-matrix score (constant ``1.0``/``0.0`` for
                    ``match_binary``, normalized ``[0, 1]`` for ``normalized_value``).
                analysis: Parsed ``{description, match, success}`` dict, or None
                    if generation was skipped/failed.
        """
        if isinstance(frames_array, str):
            frames_array = self._load_video_frames(frames_array)

        if frames_array is None or frames_array.size == 0:
            return [], None

        robot_desc, camera_desc = self._resolve_descriptions(metadata)

        frames_pil = convert_frames_to_pil_images(frames_array)
        if not frames_pil:
            return [], None

        logger.info(f"RynnValue: processing {len(frames_pil)} frames "
                     f"(stride={self.stride}, num_frames={self.num_frames})")

        processed = self.processor.process_episode(
            instruction=task_description,
            images=frames_pil,
            robot_description=robot_desc,
            camera_description=camera_desc,
        )

        device = self.model.device
        model_kwargs = dict(
            input_ids=processed["input_ids"].to(device).long(),
            attention_mask=processed["attention_mask"].to(device).long(),
        )

        pixel_values = processed.get("pixel_values")
        if pixel_values is not None:
            model_kwargs["pixel_values"] = pixel_values.flatten(0, 1).to(device)

        image_grid_thw = processed.get("image_grid_thw")
        if image_grid_thw is not None:
            model_kwargs["image_grid_thw"] = image_grid_thw.flatten(0, 1).to(device).long()

        pixel_values_videos = processed.get("pixel_values_videos")
        if pixel_values_videos is not None:
            model_kwargs["pixel_values_videos"] = pixel_values_videos.flatten(0, 1).to(device)

        video_grid_thw = processed.get("video_grid_thw")
        if video_grid_thw is not None:
            model_kwargs["video_grid_thw"] = video_grid_thw.flatten(0, 1).to(device).long()

        # ---- Forward pass -------------------------------------------------
        with torch.inference_mode():
            outputs = self.model(**model_kwargs)

        # ---- Extract value predictions ------------------------------------
        pred_value = self._reduce_pred_value(outputs.value.pred_value)

        # ---- Optional bidirectional TD(lambda) fusion with relative head --
        if self.use_fuse:
            pred_value = self._fuse_with_relative(pred_value, outputs)

        # ---- Apply mode transform -----------------------------------------
        raw_pred_value = pred_value  # raw remaining-time (pre-transform), used by normalized_value
        pred_value = self._apply_mode(pred_value)

        logger.info(f"RynnValue: progress scores computed (first={pred_value[0]:.3f}, "
                     f"last={pred_value[-1]:.3f})")

        result = pred_value.tolist()

        # ---- Extract success probabilities --------------------------------
        if getattr(outputs, "success", None) is not None and outputs.success.pred_success is not None:
            success_probs = outputs.success.pred_success
            if success_probs.dim() > 1:
                success_probs = success_probs.squeeze(-1)
            self.last_success_probs = success_probs.tolist()
        else:
            self.last_success_probs = None

        # ---- Confusion-matrix samples: score task<->video matching --------
        # Detected by both lang_task and video_task in metadata (ConfusionMatrixSampler).
        is_confusion_sample = bool(metadata) and "lang_task" in metadata and "video_task" in metadata

        # Every path uses the Analysis block (match / success / description).
        analysis = self._generate_analysis(model_kwargs)
        match_val = analysis.get("match") if analysis else None

        if is_confusion_sample and self.confusion_score_mode == "normalized_value":
            # Value-only variant: value head normalized to [0,1] (1 - t/t_max)
            # when the model reports Match:Yes; otherwise (Match:No or no verdict)
            # fall back to 0.0.
            if (match_val or "").lower() == "yes":
                result = self._apply_mode(raw_pred_value, mode="normalized").tolist()
                logger.info(f"RynnValue: confusion normalized_value score -> last={result[-1]:.3f} (match=Yes)")
            else:
                result = [0.0] * len(result)
                logger.info(f"RynnValue: confusion normalized_value, match={match_val} -> 0.0")
        elif is_confusion_sample:
            # match_binary variant: 1.0 if Match:Yes else 0.0 (bounded, no -100 sentinel).

            match_is_yes = (match_val or "").lower() == "yes"
            result = [1.0 if match_is_yes else 0.0] * len(result)
            logger.info(
                "RynnValue: confusion-matrix match score -> "
                f"{'1.0 (Match:Yes)' if match_is_yes else '0.0 (Match:No/Unknown)'}"
            )
        elif match_val is not None and match_val.lower() == "no":
            logger.info("RynnValue: match=No, nulling out progress scores with -100.")
            result = [-100.0] * len(result)

        return result, analysis

    def _generate_analysis(self, model_kwargs: dict) -> Optional[dict]:
        """Generate the Analysis block and parse match / success / description.

        Follows ``rynn_infer/infer_value.py``: one greedy ``generate`` from the
        same prompt used for the value forward pass, decoded with
        ``skip_special_tokens=True`` and parsed by :func:`parse_analysis`.
        Returns None if the processor has no tokenizer or generation raised.
        """
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            logger.warning("RynnValue: processor has no tokenizer; skipping analysis.")
            return None

        eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        input_len = model_kwargs["input_ids"].shape[1]
        try:
            with torch.inference_mode():
                gen_out = self.model.generate(
                    **model_kwargs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=eos_token_id,
                    pad_token_id=eos_token_id,
                    use_cache=True,
                )
        except Exception as e:
            logger.warning(f"RynnValue: analysis generation failed: {e}")
            return None

        analysis_text = tokenizer.decode(gen_out[0, input_len:], skip_special_tokens=True)
        analysis = parse_analysis(analysis_text)
        logger.info(
            "RynnValue: analysis parsed\n"
            f"  generated:   {analysis_text!r}\n"
            f"  description: {analysis['description']}\n"
            f"  match:       {analysis['match']}\n"
            f"  success:     {analysis['success']}"
        )
        return analysis

    @staticmethod
    def _reduce_pred_value(pred_value: torch.Tensor) -> torch.Tensor:
        """Collapse a value-head output to a 1-D per-frame sequence.

        Averages an optional leading ensemble dim, then selects the per-token
        column so both the absolute and relative heads reduce identically.
        """
        if pred_value.dim() == 3:
            pred_value = pred_value.mean(dim=0)
        if pred_value.dim() == 2 and pred_value.shape[0] == 1:
            pred_value = pred_value[0]
        elif pred_value.dim() == 2 and pred_value.shape[-1] > 1:
            pred_value = pred_value[:, -1]
        elif pred_value.dim() == 2:
            pred_value = pred_value[:, 0]
        return pred_value

    def _fuse_with_relative(self, pred_value: torch.Tensor, outputs) -> torch.Tensor:
        """Blend the absolute remaining-time head with the relative per-step head.

        Returns ``pred_value`` unchanged when the checkpoint exposes no relative
        head; :func:`fuse_td_lambda` itself no-ops on any per-step length mismatch.
        """
        relative = getattr(outputs, "relative", None)
        pred_relative = getattr(relative, "pred_value", None) if relative is not None else None
        if pred_relative is None:
            logger.warning("RynnValue: use_fuse=True but the model exposes no relative head; skipping fusion.")
            return pred_value

        pred_relative = self._reduce_pred_value(pred_relative)
        fused = fuse_td_lambda(pred_value.tolist(), pred_relative.tolist(), lam=self.fuse_lambda)
        logger.info(f"RynnValue: fused absolute + relative heads (fuse_lambda={self.fuse_lambda})")
        return torch.tensor(fused, dtype=pred_value.dtype, device=pred_value.device)

    def _apply_mode(self, pred_value: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        """Transform raw remaining-time predictions according to ``mode`` (defaults to ``self.mode``)."""
        mode = mode if mode is not None else self.mode
        if mode == "absolute":
            # Negate so that more progress → higher value
            return -pred_value
        elif mode == "normalized":
            vmax = pred_value.max()
            if vmax > 0:
                return 1 - pred_value / vmax
            return torch.ones_like(pred_value)
        elif mode == "remaining_time":
            return pred_value
        else:
            raise ValueError(
                f"Unknown mode: {mode}. "
                "Expected 'absolute', 'normalized', or 'remaining_time'."
            )

    def _resolve_descriptions(self, metadata: Optional[dict]) -> Tuple[Optional[str], Optional[str]]:
        """Resolve robot and camera descriptions from metadata."""
        robot_desc = None
        camera_desc = None
        if metadata:
            robot_desc = metadata.get("robot_description")
            camera_desc = metadata.get("camera_description")
            if robot_desc is None or camera_desc is None:
                data_source = metadata.get("data_source")
                if robot_desc is None:
                    robot_desc = DATA_SOURCE_ROBOT_DESCRIPTION.get(data_source)
                if camera_desc is None:
                    camera_desc = DATA_SOURCE_CAMERA_DESCRIPTION.get(data_source)
            if camera_desc is None and metadata.get("id"):
                camera_desc = self._camera_desc_lookup.get(metadata["id"])
        return robot_desc, camera_desc
