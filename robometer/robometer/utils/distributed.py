import os
from rich import print as rprint
from typing import Optional


def is_rank_0():
    """Check if current process is rank 0 (main process)."""
    # First check environment variables (most reliable for accelerate)
    if "LOCAL_RANK" in os.environ:
        return int(os.environ.get("LOCAL_RANK", 0)) == 0

    # Fallback to torch.distributed
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return dist.get_rank() == 0
        else:
            return True  # If not distributed, consider it rank 0
    except:
        return True  # If distributed module not available, consider it rank 0


def rank_0_print(*args, verbose=True, **kwargs):
    """Print only if on rank 0."""
    if is_rank_0() and verbose:
        rprint(*args, **kwargs)


def get_rank():
    """Get the current rank (process index) in distributed training.

    Returns:
        int: Rank number, or 0 if not in distributed training or rank cannot be determined.
    """
    # First check environment variables (most reliable for accelerate)
    if "LOCAL_RANK" in os.environ:
        return int(os.environ.get("LOCAL_RANK", 0))

    # Also check RANK for global rank
    if "RANK" in os.environ:
        return int(os.environ.get("RANK", 0))

    # Fallback to torch.distributed
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return dist.get_rank()
    except:
        pass

    # Default to 0 if not distributed
    return 0


def banner(*lines, inner_padding=3):
    rank_0_print("\n" + "#" * 60)

    # top inner padding
    for _ in range(inner_padding):
        rank_0_print("#")

    # content lines
    for line in lines:
        rank_0_print("# " + line)

    # bottom inner padding
    for _ in range(inner_padding):
        rank_0_print("#")

    rank_0_print("#" * 60 + "\n")


def log_fsdp_diagnostics(model, accelerator=None, logger=None):
    """
    Log comprehensive FSDP diagnostics including parameter counts, types, and FSDP wrapping status.
    This helps verify if FSDP is properly configured and working.

    Args:
        model: The model to check (may be wrapped in FSDP/DDP)
        accelerator: Optional Accelerate accelerator object to check FSDP plugin configuration
        logger: Optional logger instance (defaults to loguru logger if not provided)
    """
    if not is_rank_0():
        return

    # Import logger if not provided
    if logger is None:
        from robometer.utils.logger import get_logger

        logger = get_logger()

    logger.info("=" * 80)
    logger.info("FSDP DIAGNOSTICS (After Accelerate/Trainer Wrapping)")
    logger.info("=" * 80)

    # Check if FSDP is available
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        has_fsdp_v1 = True
    except ImportError:
        has_fsdp_v1 = False
        FSDP = None

    try:
        from torch.distributed._composable.fsdp import FSDPModule

        has_fsdp_v2 = True
    except ImportError:
        has_fsdp_v2 = False
        FSDPModule = None

    logger.info(f"FSDP v1 (FullyShardedDataParallel) available: {has_fsdp_v1}")
    logger.info(f"FSDP v2 (FSDPModule) available: {has_fsdp_v2}")

    # Unwrap model if it's wrapped in DDP/FSDP for checking
    # The model passed to training_step might already be unwrapped, but let's check
    unwrapped_model = model
    if hasattr(model, "module"):
        unwrapped_model = model.module

    # Check if model is wrapped in FSDP
    is_fsdp_wrapped = False
    fsdp_version = 0
    if has_fsdp_v1 and isinstance(model, FSDP):
        is_fsdp_wrapped = True
        fsdp_version = 1
        logger.info("Model is wrapped in FSDP v1 (FullyShardedDataParallel)")
    elif has_fsdp_v2 and isinstance(model, FSDPModule):
        is_fsdp_wrapped = True
        fsdp_version = 2
        logger.info("Model is wrapped in FSDP v2 (FSDPModule)")
    else:
        logger.info("Model root is NOT wrapped in FSDP (checking submodules...)")

    # Check for FSDP-wrapped submodules
    fsdp_wrapped_modules = []
    if has_fsdp_v1:
        for name, module in model.named_modules():
            if isinstance(module, FSDP):
                fsdp_wrapped_modules.append((name, "FSDP_v1"))
    if has_fsdp_v2:
        for name, module in model.named_modules():
            if isinstance(module, FSDPModule):
                fsdp_wrapped_modules.append((name, "FSDP_v2"))

    if fsdp_wrapped_modules:
        logger.info(f"Found {len(fsdp_wrapped_modules)} FSDP-wrapped submodules:")
        for name, fsdp_type in fsdp_wrapped_modules:
            logger.info(f"  - {name} ({fsdp_type})")
    else:
        logger.info("⚠ No FSDP-wrapped submodules found - FSDP may not be active")

    # Parameter statistics
    all_params = list(model.parameters())
    trainable_params = [p for p in all_params if p.requires_grad]
    frozen_params = [p for p in all_params if not p.requires_grad]

    # Count parameters by device
    device_counts = {}
    for param in all_params:
        device = str(param.device)
        device_counts[device] = device_counts.get(device, 0) + param.numel()

    logger.info("\nParameter Statistics:")
    logger.info(f"  Total parameters: {sum(p.numel() for p in all_params):,}")
    logger.info(f"  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    logger.info(f"  Frozen parameters: {sum(p.numel() for p in frozen_params):,}")
    if sum(p.numel() for p in all_params) > 0:
        logger.info(
            f"  Trainable %: {100 * sum(p.numel() for p in trainable_params) / sum(p.numel() for p in all_params):.4f}%"
        )

    logger.info("\nParameters by device:")
    for device, count in sorted(device_counts.items()):
        total = sum(device_counts.values())
        if total > 0:
            logger.info(f"  {device}: {count:,} parameters ({100 * count / total:.2f}%)")

    # Parameter types and dtypes
    dtype_counts = {}
    for param in all_params:
        dtype = str(param.dtype)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + param.numel()

    logger.info("\nParameters by dtype:")
    for dtype, count in sorted(dtype_counts.items()):
        total = sum(dtype_counts.values())
        if total > 0:
            logger.info(f"  {dtype}: {count:,} parameters ({100 * count / total:.2f}%)")

    # Check for FSDP-specific attributes if wrapped
    if is_fsdp_wrapped:
        logger.info("\nFSDP Configuration (Root Model):")
        if fsdp_version == 1:
            # FSDP v1 attributes
            if hasattr(model, "sharding_strategy"):
                logger.info(f"  Sharding strategy: {model.sharding_strategy}")
            if hasattr(model, "mixed_precision"):
                logger.info(f"  Mixed precision: {model.mixed_precision}")
            if hasattr(model, "cpu_offload"):
                logger.info(f"  CPU offload: {model.cpu_offload}")
            if hasattr(model, "use_orig_params"):
                logger.info(f"  Use orig params: {model.use_orig_params}")
        elif fsdp_version == 2:
            # FSDP v2 attributes
            logger.info("  FSDP v2 detected (composable FSDP)")

    # Check Accelerate FSDP plugin configuration
    if accelerator is not None:
        logger.info("\nAccelerate Configuration:")
        if hasattr(accelerator.state, "fsdp_plugin"):
            fsdp_plugin = accelerator.state.fsdp_plugin
            logger.info(f"  FSDP Plugin found: {type(fsdp_plugin).__name__}")
            if hasattr(fsdp_plugin, "sharding_strategy"):
                logger.info(f"  Sharding strategy: {fsdp_plugin.sharding_strategy}")
            if hasattr(fsdp_plugin, "mixed_precision_policy"):
                logger.info(f"  Mixed precision policy: {fsdp_plugin.mixed_precision_policy}")
            if hasattr(fsdp_plugin, "cpu_offload_policy"):
                logger.info(f"  CPU offload policy: {fsdp_plugin.cpu_offload_policy}")
        else:
            logger.info("  FSDP Plugin: Not found (FSDP may not be configured)")

    # Check gradient checkpointing status
    logger.info("\nGradient Checkpointing:")
    gradient_checkpointing_enabled = False
    gradient_checkpointing_info = []

    # Check unwrapped model first
    if hasattr(unwrapped_model, "is_gradient_checkpointing"):
        gradient_checkpointing_enabled = unwrapped_model.is_gradient_checkpointing
        gradient_checkpointing_info.append(f"  Root model: {gradient_checkpointing_enabled}")
    elif hasattr(unwrapped_model, "gradient_checkpointing"):
        gradient_checkpointing_enabled = unwrapped_model.gradient_checkpointing
        gradient_checkpointing_info.append(f"  Root model: {gradient_checkpointing_enabled}")
    else:
        gradient_checkpointing_info.append("  Root model: is_gradient_checkpointing attribute not found")

    # Check if the underlying model (e.g., base_model) has gradient checkpointing
    if hasattr(unwrapped_model, "model"):
        base_model = unwrapped_model.model
        if hasattr(base_model, "is_gradient_checkpointing"):
            base_gc = base_model.is_gradient_checkpointing
            gradient_checkpointing_info.append(f"  Base model: {base_gc}")
        elif hasattr(base_model, "gradient_checkpointing"):
            base_gc = base_model.gradient_checkpointing
            gradient_checkpointing_info.append(f"  Base model: {base_gc}")

    # Check wrapped model as well
    if hasattr(model, "is_gradient_checkpointing") and model != unwrapped_model:
        wrapped_gc = model.is_gradient_checkpointing
        gradient_checkpointing_info.append(f"  Wrapped model: {wrapped_gc}")

    for info in gradient_checkpointing_info:
        logger.info(info)

    if not gradient_checkpointing_info or all("not found" in info.lower() for info in gradient_checkpointing_info):
        logger.info("  Note: Could not determine gradient checkpointing status from model attributes")

    # Check distributed environment
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        logger.info(f"\nDistributed Environment:")
        logger.info(f"  World size: {dist.get_world_size()}")
        logger.info(f"  Rank: {dist.get_rank()}")
        logger.info(f"  Backend: {dist.get_backend()}")
    else:
        logger.info("\nDistributed Environment: Not initialized (single process)")

    logger.info("=" * 80)
