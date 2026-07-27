#!/usr/bin/env python3
"""
Utility functions for working with Hydra configs and dataclasses.
"""

from typing import TypeVar, Type
from omegaconf import OmegaConf, DictConfig
from rich.console import Console

from robometer.utils.distributed import is_rank_0

T = TypeVar("T")


def display_config(cfg):
    """Display the configuration in a nice Rich format."""
    if not is_rank_0():
        return  # Only display config on rank 0

    console = Console()
    console.print(cfg)


def convert_hydra_to_dataclass(cfg: DictConfig, dataclass_type: Type[T]) -> T:
    """
    Convert Hydra DictConfig to a dataclass instance.

    Args:
        cfg: Hydra DictConfig to convert
        dataclass_type: The dataclass type to convert to (e.g., ExperimentConfig, OfflineEvalConfig)

    Returns:
        Instance of the specified dataclass type
    """
    # Convert to dict and then to dataclass
    # Use structured config if available, otherwise convert manually
    if OmegaConf.is_struct(cfg):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True, structured_config_mode="convert")
    else:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    return dataclass_type(**cfg_dict)
