import os
import sys
import json
from typing import Dict, Any, Iterable, Optional, List
import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from loguru import logger as loguru_logger

from robometer.utils.distributed import get_rank, is_rank_0

# Define custom log levels for more granular debugging
# Standard levels: CRITICAL=50, ERROR=40, WARNING=30, INFO=20, DEBUG=10
# TRACE (5) already exists in loguru by default
DEBUG2_LEVEL = 8  # Intermediate debug level between TRACE (5) and DEBUG (10)


def _add_custom_log_levels():
    """Add custom log levels to loguru for more granular debugging control."""
    # TRACE level (5) already exists in loguru by default, no need to add it
    # Only add DEBUG2 level (intermediate verbose)
    # Check if level already exists to avoid errors on multiple calls
    try:
        loguru_logger.level("DEBUG2", no=DEBUG2_LEVEL, color="<dim><cyan>")
    except ValueError:
        # Level already exists, which is fine
        pass


def setup_loguru_logging(log_level: str = "INFO", output_dir: Optional[str] = None):
    """
    Initialize loguru logger with rank-aware formatting and log level.
    Uses loguru's default format to preserve automatic rich formatting for booleans, numbers, etc.

    Supported log levels (from most to least verbose):
    - TRACE: Most verbose debugging (level 5)
    - DEBUG2: Intermediate debug level (level 8)
    - DEBUG: Standard debug level (level 10)
    - INFO: Informational messages (level 20)
    - WARNING: Warning messages (level 30)
    - ERROR: Error messages (level 40)
    - CRITICAL: Critical errors (level 50)

    Args:
        log_level: Logging level (e.g., "TRACE", "DEBUG2", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        output_dir: Optional directory to write log files to
    """
    # Add custom log levels first
    _add_custom_log_levels()

    loguru_logger.remove()
    rank = get_rank()
    rank_prefix = f"[Rank {rank}] "
    format_string = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        f"{rank_prefix}"
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Console handler using sys.stderr to preserve loguru's rich formatting
    loguru_logger.add(
        sys.stderr,
        format=format_string,
        level=log_level.upper(),
        colorize=True,
    )

    # Optional file handler if output_dir is provided
    if output_dir and is_rank_0():
        log_file = os.path.join(output_dir, "training.log")
        loguru_logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}\n",
            level=log_level.upper(),
            rotation="10 MB",
            retention="7 days",
            encoding="utf-8",
        )


class Logger:
    """Logger for metrics (wandb/tensorboard). For console logging, use loguru logger directly."""

    def __init__(
        self,
        log_to: Iterable[str] | None,
        output_dir: str,
        is_main_process: bool = True,
        wandb_run: Optional[Any] = None,
        log_level: str = "INFO",
    ):
        backends = [b.lower() for b in (list(log_to) if log_to is not None else [])]
        self._use_wandb = "wandb" in backends
        self._use_tb = "tensorboard" in backends
        self._is_main = bool(is_main_process)

        # Setup loguru for console logging
        setup_loguru_logging(log_level=log_level, output_dir=output_dir if is_main_process else None)

        self._wandb_run = wandb.run if (self._use_wandb and self._is_main) else None

        self._tb_writer = None
        if self._use_tb and self._is_main:
            logging_dir = os.path.join(output_dir, "tb")
            os.makedirs(logging_dir, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=logging_dir)

    def init_wandb(
        self,
        project: Optional[str],
        entity: Optional[str],
        name: Optional[str],
        config: Optional[dict],
        notes: Optional[str] = None,
        mode: Optional[str] = None,
        resume_id: Optional[str] = None,
    ):
        if not (self._use_wandb and self._is_main):
            return None
        if self._wandb_run is not None:
            return self._wandb_run
        init_kwargs = {"project": project, "entity": entity, "name": name, "config": config}
        if notes:
            init_kwargs["notes"] = notes
        # Use offline mode if specified, or check environment variable
        if mode is None:
            mode = os.environ.get("WANDB_MODE", "online")
        init_kwargs["mode"] = mode
        # Resume existing run if resume_id is provided
        if resume_id:
            init_kwargs["id"] = resume_id
            init_kwargs["resume"] = "must"
        self._wandb_run = wandb.init(**init_kwargs)
        return self._wandb_run

    def enabled(self, backend: str) -> bool:
        backend = backend.lower()
        if backend == "wandb":
            return self._use_wandb and self._is_main and (self._wandb_run is not None)
        if backend == "tensorboard":
            return self._use_tb and self._is_main and self._tb_writer is not None
        return False

    def log_scalars(self, scalars: Dict[str, float], step: Optional[int] = None):
        if not self._is_main:
            return
        if self.enabled("wandb"):
            self._wandb_run.log(scalars, step=step)
        if self.enabled("tensorboard"):
            for k, v in scalars.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(k, float(v), global_step=step)

    def log_figure(self, tag: str, figure, step: Optional[int] = None):
        if not self._is_main:
            return
        if self.enabled("wandb"):
            self._wandb_run.log({tag: wandb.Image(figure)}, step=step)
        if self.enabled("tensorboard"):
            self._tb_writer.add_figure(tag, figure, global_step=step)

    def log_image(self, tag: str, image: np.ndarray, step: Optional[int] = None):
        """
        Log a static image.
        - image: numpy array in (H, W, C) format, uint8, RGB
        """
        if not self._is_main:
            return
        if self.enabled("wandb"):
            # Ensure image is in (H, W, C) format and uint8
            if image.ndim == 3 and image.shape[2] == 3:
                if image.dtype != np.uint8:
                    image = np.clip(image, 0, 255).astype(np.uint8)
                self._wandb_run.log({tag: wandb.Image(image)}, step=step)
        if self.enabled("tensorboard"):
            # Convert to (C, H, W) for tensorboard
            if image.ndim == 3 and image.shape[2] == 3:
                image_chw = image.transpose(2, 0, 1)
                if image_chw.dtype != np.uint8:
                    image_chw = np.clip(image_chw, 0, 255).astype(np.uint8)
                self._tb_writer.add_image(tag, image_chw, global_step=step)

    def log_video_table(
        self,
        tag: str,
        videos_and_figures: List[tuple],
        columns: List[str],
        step: Optional[int] = None,
        fps: int = 6,
    ):
        """
        Log a table where first column can be video (wandb), second a figure, etc.
        videos_and_figures: list of tuples e.g. [(video_array_or_path, figure), ...]
        Only supported for wandb; TensorBoard has no native table/video support.
        """
        if not self._is_main:
            return

        if self.enabled("wandb"):
            rows = []
            for item in videos_and_figures:
                row = []
                for x in item:
                    if x is None:
                        row.append(None)
                    else:
                        # Wrap images/figures
                        if hasattr(x, "savefig") or getattr(x, "__class__", type("x", (), {})).__name__ == "Figure":
                            row.append(wandb.Image(x))
                            # Free matplotlib figure memory after wrapping
                            plt.close(x)
                        # Wrap file path as video
                        elif isinstance(x, str):
                            row.append(wandb.Video(x, fps=fps, format="gif"))
                        else:
                            # Try to interpret as array/tensor video
                            arr = None
                            if isinstance(x, np.ndarray):
                                arr = x
                            elif isinstance(x, torch.Tensor):
                                arr = x.detach().cpu().numpy()
                            if arr is not None and arr.ndim == 4:
                                # wandb.Video expects T x C x H x W format
                                # If last dimension is 3 (or 1), it's THWC format, convert to TCHW
                                if arr.shape[-1] in (1, 3) and arr.shape[1] not in (1, 3):
                                    arr = np.transpose(arr, (0, 3, 1, 2))  # T x H x W x C -> T x C x H x W
                                row.append(wandb.Video(arr, fps=fps, format="gif"))
                            else:
                                # Fallback: store raw value
                                row.append(x)
                rows.append(row)
            # if step is not None:
            #     tag_with_step = f"{tag}/step_{step}"
            # else:
            #     tag_with_step = tag
            self._wandb_run.log({tag: wandb.Table(data=rows, columns=columns)}, step=step)

    def add_text(self, tag: str, text: str, step: Optional[int] = None):
        if not self._is_main:
            return
        if self.enabled("tensorboard"):
            self._tb_writer.add_text(tag, text, global_step=step)
        # For wandb, text can be added via log_scalars or a media panel; skip for simplicity
        if self.enabled("wandb"):
            # Store as a simple text panel by wrapping in a dict
            self._wandb_run.log({tag: wandb.Html(f"<pre>{text}</pre>")}, step=step)

    def log_table(self, tag: str, data: List[List[Any]], columns: List[str], step: Optional[int] = None):
        """
        Log a generic table (wandb only). TensorBoard has no native table support.
        """
        if not self._is_main:
            return
        if self.enabled("wandb"):
            # if step is not None:
            #     tag_with_step = f"{tag}/step_{step}"
            # else:
            #     tag_with_step = tag
            self._wandb_run.log({tag: wandb.Table(data=data, columns=columns)}, step=step)

    def log_video(self, tag: str, video: Any, fps: int = 10, step: Optional[int] = None):
        """
        Log a single video clip.
        - For wandb: accepts file path or numpy/torch array; arrays are expected as T x H x W x C.
        - For TensorBoard: accepts numpy/torch array; converted to 1 x C x T x H x W with values in [0,1].
        """
        if not self._is_main:
            return
        # wandb
        if self.enabled("wandb"):
            if isinstance(video, str):
                self._wandb_run.log({tag: wandb.Video(video, fps=fps)}, step=step)
            else:
                arr = None
                if isinstance(video, np.ndarray):
                    arr = video
                elif isinstance(video, torch.Tensor):
                    arr = video.detach().cpu().numpy()
                if arr is not None and arr.ndim == 4:
                    # wandb.Video expects T x C x H x W format
                    # If last dimension is 3 (or 1), it's THWC format, convert to TCHW
                    if arr.shape[-1] in (1, 3) and arr.shape[1] not in (1, 3):
                        arr = np.transpose(arr, (0, 3, 1, 2))  # T x H x W x C -> T x C x H x W
                    self._wandb_run.log({tag: wandb.Video(arr, fps=fps, format="mp4")}, step=step)
        # tensorboard
        if self.enabled("tensorboard"):
            tens = None
            if isinstance(video, torch.Tensor):
                tens = video.detach().cpu()
            elif isinstance(video, np.ndarray):
                tens = torch.from_numpy(video)
            if tens is not None and tens.dim() == 4:
                # Convert to C x T x H x W
                if tens.shape[-1] in (1, 3):  # T x H x W x C
                    tens = tens.permute(0, 3, 1, 2)  # T x C x H x W
                if tens.dtype != torch.float32:
                    tens = tens.float()
                if tens.numel() > 0 and tens.max().item() > 1.0:
                    tens = tens / 255.0
                tens = tens.permute(1, 0, 2, 3).unsqueeze(0)  # 1 x C x T x H x W
                self._tb_writer.add_video(tag, tens, global_step=step, fps=fps)

    def close(self):
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()

    def write_wandb_info(self, output_dir: str, run_name: str):
        """
        Persist basic wandb run information alongside outputs, if wandb is active.
        Safe to call even if wandb isn't enabled or initialized.
        """
        if not self.enabled("wandb"):
            return
        run = self._wandb_run
        if run is None:
            return
        info = {
            "wandb_id": run.id,
            "wandb_name": run.name or run_name,
            "wandb_project": run.project,
            "wandb_entity": run.entity,
            "wandb_url": run.url,
        }
        # Include notes if available
        if hasattr(run, "notes") and run.notes:
            info["wandb_notes"] = run.notes
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "wandb_info.json")
        with open(path, "w") as f:
            json.dump(info, f, indent=2)


def get_logger():
    """Get the loguru logger instance for structured logging."""
    return loguru_logger


def rank_0_info(*args, **kwargs):
    """Log info message only on rank 0 (main process)."""
    if is_rank_0():
        loguru_logger.info(*args, **kwargs)


def rank_0_warning(*args, **kwargs):
    """Log warning message only on rank 0 (main process)."""
    if is_rank_0():
        loguru_logger.warning(*args, **kwargs)


def rank_0_debug(*args, **kwargs):
    """Log debug message only on rank 0 (main process)."""
    if is_rank_0():
        loguru_logger.debug(*args, **kwargs)


def trace(*args, **kwargs):
    """Log trace message (most verbose debug level)."""
    loguru_logger.trace(*args, **kwargs)


def debug2(*args, **kwargs):
    """Log debug2 message (intermediate debug level)."""
    loguru_logger.debug2(*args, **kwargs)


def rank_0_trace(*args, **kwargs):
    """Log trace message only on rank 0 (main process)."""
    if is_rank_0():
        loguru_logger.trace(*args, **kwargs)


def rank_0_debug2(*args, **kwargs):
    """Log debug2 message only on rank 0 (main process)."""
    if is_rank_0():
        loguru_logger.debug2(*args, **kwargs)


def log_memory_usage(prefix="", rank=None, output_dir=None):
    """Log GPU and CPU memory usage in a readable format.

    Args:
        prefix: Prefix string to identify the logging point
        rank: Process rank (auto-detected if None)
        output_dir: Directory to write memory log file (uses OUTPUT_DIR env var if None)

    Returns:
        str: Memory usage string
    """
    import psutil
    import datetime

    if rank is None:
        rank = get_rank()

    memory_info = []

    # GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3  # GB
            memory_info.append(
                f"GPU{i}: {allocated:.2f}GB alloc, {reserved:.2f}GB reserved, {max_allocated:.2f}GB peak"
            )

    # CPU memory
    process = psutil.Process()
    mem_info = process.memory_info()
    cpu_mem_rss = mem_info.rss / 1024**3  # GB - Resident Set Size (actual physical memory)
    cpu_mem_vms = mem_info.vms / 1024**3  # GB - Virtual Memory Size

    # Get system-wide memory stats
    system_mem = psutil.virtual_memory()
    system_total = system_mem.total / 1024**3  # GB
    system_available = system_mem.available / 1024**3  # GB
    system_percent = system_mem.percent

    memory_info.append(
        f"CPU: {cpu_mem_rss:.2f}GB RSS, {cpu_mem_vms:.2f}GB VMS | "
        f"System: {system_available:.2f}GB/{system_total:.2f}GB avail ({system_percent:.1f}% used)"
    )

    memory_str = " | ".join(memory_info) if memory_info else "No memory info available"
    loguru_logger.info(f"[Rank {rank}] {prefix} Memory: {memory_str}")

    # Also write to a memory log file for post-mortem analysis
    if output_dir is None:
        output_dir = os.environ.get("OUTPUT_DIR", ".")

    memory_log_file = os.path.join(output_dir, f"memory_log_rank{rank}.txt")
    with open(memory_log_file, "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp} | {prefix} | {memory_str}\n")

    return memory_str
