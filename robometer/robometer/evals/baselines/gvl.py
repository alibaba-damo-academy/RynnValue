import cv2
import base64
import random
import re
import requests
import json
import os
import time
from typing import List, Dict, Optional
import numpy as np


class GVL:
    def __init__(
        self,
        api_key: str = None,
        max_frames: int = 15,
        offset: float = 0.5,
        model_name: str = "gemini-2.0-flash",
        provider: str = "gemini",
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        **kwargs,
    ):
        """
        :param api_key:          API key for the provider (defaults to env var based on provider)
        :param max_frames:       If N > max_frames, sample exactly max_frames frames
        :param offset:           Time offset used when sampling frames (seconds/frame). Keep consistent meaning with the frontend if applicable.
        :param model_name:       Model name to use (e.g., "gemini-2.0-flash" for Gemini, "gpt-4o" for OpenAI)
        :param provider:         API provider: "gemini" or "openai"
        :param max_retries:      Maximum number of retries on API throttling/errors
        :param base_delay:       Base delay in seconds for exponential backoff
        :param max_delay:        Maximum delay in seconds between retries
        """
        self.provider = provider.lower()
        if self.provider not in ["gemini", "openai"]:
            raise ValueError(f"Unsupported provider: {provider}. Must be 'gemini' or 'openai'")

        # Set API key based on provider
        if api_key is None:
            if self.provider == "gemini":
                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY environment variable must be set")
            else:  # openai
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable must be set")

        self.api_key = api_key
        self.max_frames = max_frames
        self.offset = offset
        self.model_name = model_name
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        # List to store frame info: each element contains
        # {"gt_index": i, "shuffled_index": ..., "base64": "..."}
        self.frames_info: List[Dict] = []

    def extract_frames_from_memory(self, frames_array: np.ndarray) -> None:
        """
        Improved sampling logic:
        - If N <= max_frames, keep previous behavior (typically capturing first and last frames)
        - If N > max_frames:
          1) Always include frame 0 and frame N-1
          2) Uniformly sample (max_frames - 2) frames from the middle (N-2) range
          3) Merge indices, deduplicate, sort, then encode
        """
        total_frames = frames_array.shape[0]
        if total_frames == 0:
            print("[!] frames_array is empty; cannot extract frames.")
            self.frames_info = []
            return

        # If total_frames <= max_frames: keep original behavior
        if total_frames <= self.max_frames:
            frame_count = total_frames
            frame_interval = 1.0
            print(f"Extracting all {frame_count} frames (<= max_frames).")

            temp_indices = []
            for i in range(frame_count):
                # Use offset + i*frame_interval then int(...) to compute index
                sample_time = self.offset + i * frame_interval
                frame_index = int(sample_time)
                if 0 <= frame_index < total_frames:
                    temp_indices.append(frame_index)

        else:
            # total_frames > max_frames
            print(
                f"Extracting exactly {self.max_frames} frames from total={total_frames}, ensuring first & last included."
            )
            # 1) Always include frame 0 and frame N-1
            temp_indices = [0, total_frames - 1]

            # 2) Uniformly sample (max_frames - 2) middle frames
            #    Note: you can keep offset as offset + i*frame_interval if desired.
            #    Whether offset is meaningful for the first frame=0 depends on your use case.
            #    Here we keep offset so that middle frames also carry a 0.5 shift.
            inner_count = self.max_frames - 2
            if (total_frames - 2) <= 0:
                # If there are only 1 or 2 frames, skip middle sampling
                print("Warning: total_frames - 2 <= 0, cannot sample middle frames.")
            else:
                frame_interval = (total_frames - 2) / float(inner_count)
                for i in range(inner_count):
                    sample_time = self.offset + i * frame_interval
                    # Middle frame index falls within [1, N-2]
                    frame_index = int(1 + sample_time)
                    if 1 <= frame_index < (total_frames - 1):
                        temp_indices.append(frame_index)

            # Deduplicate and sort
            temp_indices = sorted(set(temp_indices))
            print(f"Extracted {len(temp_indices)} frames: {temp_indices}")

        # ============ Convert indices to JPEG + base64 ============
        temp_frames_info = []
        for idx in temp_indices:
            frame = frames_array[idx]  # shape = (H, W, 3)
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            frame_b64 = base64.b64encode(buffer).decode("utf-8")
            temp_frames_info.append({"gt_index": len(temp_frames_info) + 1, "base64": frame_b64})

        self.frames_info = temp_frames_info

    def shuffle_frames(self) -> None:
        """
        Randomly shuffle frames_info. Similar to the previous shuffle_frames behavior.
        """
        indices = list(range(1, len(self.frames_info) + 1))
        random.shuffle(indices)
        for frame, new_idx in zip(self.frames_info, indices):
            frame["shuffled_index"] = new_idx

    def build_prompt_parts(self, task_description: str) -> List[Dict]:
        """
        Build the `parts` list for the request, following the previous logic:
          - First find the frame with gt_index=1 as the initial scene
          - Add prompt1 + that frame
          - Add prompt2
          - Then, sorted by shuffled_index, append "Frame i:" + the corresponding frame
        """
        # Find the frame with gt_index = 1
        initial_frame = next((f for f in self.frames_info if f["gt_index"] == 1), None)
        if not initial_frame:
            # If not found (extreme case), use the first frame
            if not self.frames_info:
                print("[!] No available frames; cannot build prompt_parts")
                return []
            initial_frame = self.frames_info[0]

        prompt1 = (
            f"You are an expert roboticist tasked to predict task completion percentages "
            f"for frames of a robot for the task of {task_description}. "
            f"The task completion percentages are between 0 and 100, where 100 corresponds to full task completion. "
            f"Note that these frames are in random order, so please pay attention to the individual frames. "
            f"\nInitial robot scene:\nThis frame:"
        )

        prompt2 = (
            f" shows the initial robot scene, where the task completion percentage is 0.\n\n"
            f"Now, for the task of *{task_description}*, output the task completion percentage "
            f"for the following frames that are presented in random order. "
            f"Format your response in JSON as follows, making sure to include all frames:\n\n"
            f"[\n"
            f'  {{"frame_number": i, "frame_description": "...", "task_completion_percentage": 0-100}}\n'
            f"]\n"
        )

        parts = []
        # 1) prompt1
        parts.append({"text": prompt1})
        # 2) initial_frame inline
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": initial_frame["base64"]}})
        # 3) prompt2
        parts.append({"text": prompt2})

        # 4) "Frame X" + inline, sorted by shuffled_index
        frames_sorted_by_shuffle = sorted(self.frames_info, key=lambda f: f["shuffled_index"])

        for i, frame in enumerate(frames_sorted_by_shuffle, start=1):
            parts.append({"text": f"Frame {i}:"})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": frame["base64"]}})

        return parts

    def _stream_inference_gemini(self, parts: List[Dict]) -> str:
        """
        Call the Gemini SSE endpoint and return the concatenated streamed text.
        Implements exponential backoff retry on throttling (429) and server errors (5xx).
        """
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:streamGenerateContent?alt=sse&key={self.api_key}"
        body = {"contents": [{"parts": parts}]}
        headers = {"Content-Type": "application/json"}

        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                full_text = ""
                with requests.post(url, headers=headers, json=body, stream=True) as resp:
                    # Check for throttling or server errors
                    if resp.status_code == 429 or resp.status_code >= 500:
                        delay = min(self.base_delay * (2**attempt), self.max_delay)
                        print(
                            f"[GVL] API returned {resp.status_code}, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})"
                        )
                        time.sleep(delay)
                        continue

                    resp.raise_for_status()

                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_str = line[len("data: ") :]
                            if data_str == "[DONE]":
                                break
                            try:
                                data_json = json.loads(data_str)
                                candidates = data_json.get("candidates")
                                if candidates and len(candidates) > 0:
                                    content = candidates[0].get("content", {})
                                    parts_list = content.get("parts", [])
                                    if parts_list:
                                        text_piece = parts_list[0].get("text", "")
                                        full_text += text_piece
                            except json.JSONDecodeError:
                                continue

                return full_text

            except requests.exceptions.RequestException as e:
                last_exception = e
                delay = min(self.base_delay * (2**attempt), self.max_delay)
                print(
                    f"[GVL] Request failed: {e}, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})"
                )
                time.sleep(delay)

        # All retries exhausted
        print(f"[GVL] All {self.max_retries + 1} attempts failed")
        if last_exception:
            raise last_exception
        return ""

    def _stream_inference_openai(self, parts: List[Dict]) -> str:
        """
        Call the OpenAI API with vision capabilities and return the response text.
        Implements exponential backoff retry on throttling (429) and server errors (5xx).
        """
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        # Convert parts to OpenAI message format
        content = []
        for part in parts:
            if "text" in part:
                content.append({"type": "text", "text": part["text"]})
            elif "inline_data" in part:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{part['inline_data']['mime_type']};base64,{part['inline_data']['data']}",
                        "detail": "low",  # Use low detail for efficiency
                    },
                })

        body = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 4096,
            "stream": True,
        }

        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                full_text = ""
                with requests.post(url, headers=headers, json=body, stream=True) as resp:
                    # Check for throttling or server errors
                    if resp.status_code == 429 or resp.status_code >= 500:
                        delay = min(self.base_delay * (2**attempt), self.max_delay)
                        print(
                            f"[GVL-OpenAI] API returned {resp.status_code}, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})"
                        )
                        time.sleep(delay)
                        continue

                    resp.raise_for_status()

                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_str = line[len("data: ") :]
                            if data_str == "[DONE]":
                                break
                            try:
                                data_json = json.loads(data_str)
                                choices = data_json.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    text_piece = delta.get("content", "")
                                    if text_piece:
                                        full_text += text_piece
                            except json.JSONDecodeError:
                                continue

                return full_text

            except requests.exceptions.RequestException as e:
                last_exception = e
                delay = min(self.base_delay * (2**attempt), self.max_delay)
                print(
                    f"[GVL-OpenAI] Request failed: {e}, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})"
                )
                time.sleep(delay)

        # All retries exhausted
        print(f"[GVL-OpenAI] All {self.max_retries + 1} attempts failed")
        if last_exception:
            raise last_exception
        return ""

    def stream_inference(self, parts: List[Dict]) -> str:
        """
        Call the appropriate API endpoint based on provider and return the response text.
        """
        if self.provider == "gemini":
            return self._stream_inference_gemini(parts)
        else:  # openai
            return self._stream_inference_openai(parts)

    @staticmethod
    def extract_json_from_response(text: str) -> str:
        """
        Extract JSON from a long text similar to the frontend JS logic:
        - Prefer a ```json ... ``` fenced code block
        - Otherwise try to match an array like [... { ... }, ...]
        If not found, return an empty string.
        """
        code_block_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")
        match = code_block_pattern.search(text)
        if match:
            return match.group(1).strip()

        array_pattern = re.compile(r"\[\s*\{[\s\S]*?\}\s*\]")
        match = array_pattern.search(text)
        if match:
            return match.group(0).strip()

        return ""

    @staticmethod
    def parse_model_output(model_text: str) -> Optional[List[Dict]]:
        """
        Try to extract JSON from the model text and json.loads it into a Python list.
        Return None on failure.
        """
        json_str = GVL.extract_json_from_response(model_text)
        if not json_str:
            return None
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                return data
            return None
        except (json.JSONDecodeError, TypeError):
            return None

    def compute_progress(self, frames_array: np.ndarray, task_description: str = "", reference_video_path: Optional[str] = None) -> List[Optional[float]]:
        """
        Compute progress predictions for frames using GVL baseline.

        Main pipeline:
          1) Extract and encode frames from the in-memory array
          2) Randomly shuffle
          3) Build prompt
          4) Run SSE inference
          5) Parse JSON
          6) Return completion array aligned with original frame order (gt_index)

        :param frames_array: (N, H, W, 3) uint8 array from HDF5 (per-frame video data)
        :param task_description: Robot task description
        :return: List of completion percentages (0-1, normalized from 0-100) aligned with (gt_index)
        """
        # 1) Extract frames
        self.extract_frames_from_memory(frames_array)
        if not self.frames_info:
            print("[!] No frames extracted from memory array.")
            return []

        # 2) Randomly shuffle
        self.shuffle_frames()

        # 3) Build prompt
        parts = self.build_prompt_parts(task_description)
        if not parts:
            print("[!] build_prompt_parts failed; possibly no frame data")
            return []

        # 4) SSE inference
        model_output_text = self.stream_inference(parts)

        # 5) Parse JSON
        result_data = self.parse_model_output(model_output_text)
        if result_data is None:
            print("[!] Failed to extract valid JSON; please check the model output.")
            print("Full model output text:", model_output_text)
            return []

        # 6) Map model output back to gt_index
        mapped_by_shuffled = {}
        for item in result_data:
            sidx = item.get("frame_number")
            if isinstance(sidx, int):
                mapped_by_shuffled[sidx] = item

        # Record model results in frames_info
        for frame in self.frames_info:
            sidx = frame.get("shuffled_index")
            if sidx in mapped_by_shuffled:
                frame["model_output"] = mapped_by_shuffled[sidx]
            else:
                frame["model_output"] = None

        # Iterate in ascending gt_index order
        frames_in_gt_order = sorted(self.frames_info, key=lambda f: f["gt_index"])
        task_completion_list = []
        for f in frames_in_gt_order:
            if f["model_output"] is not None:
                percentage = f["model_output"].get("task_completion_percentage")
                # Normalize from 0-100 to 0-1
                normalized = percentage / 100.0
                task_completion_list.append(normalized)
        print("Task completion list:", task_completion_list)
        return task_completion_list
