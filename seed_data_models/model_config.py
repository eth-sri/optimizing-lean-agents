"""
Utility functions for loading and parsing model configurations.

This module provides helpers to extract effective parameters from model config files.
"""

from pathlib import Path
from typing import Optional
import yaml


def get_effective_parameters(model_config_path: Optional[str], base_path: Optional[Path] = None) -> Optional[float]:
    """
    Extract effective parameters (in billions) from a model config file.

    Args:
        model_config_path: Path to the model config (e.g., "configs/models/goedel_prover_v2/32b.yaml")
                          If None, returns None.
        base_path: Base directory to resolve relative paths. Defaults to project root.

    Returns:
        The effective parameters value (as a float) if found, otherwise None.
        For example: 8, 32, 5.1, 3.6, etc.

    Examples:
        >>> get_effective_parameters("configs/models/goedel_prover_v2/32b.yaml")
        32
        >>> get_effective_parameters("configs/models/openai/oss-120b-high.yaml")
        5.1
    """
    if model_config_path is None:
        return None

    if base_path is None:
        # Try to find project root
        base_path = Path(__file__).parent.parent

    config_path = Path(base_path) / model_config_path

    if not config_path.exists():
        # Try absolute path
        config_path = Path(model_config_path)
        if not config_path.exists():
            return None

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            return None

        effective_params = config.get('effective_parameters')

        if effective_params is not None:
            return float(effective_params)

        return None
    except Exception:
        return None


def calculate_sflops_from_tokens(output_tokens: int, model_config_path: Optional[str], base_path: Optional[Path] = None) -> int:
    """
    Calculate SFLOPs (scaled FLOPs) from output tokens and model config.

    SFLOPs = output_tokens * effective_parameters

    Args:
        output_tokens: Number of output tokens
        model_config_path: Path to the model config file
        base_path: Base directory for resolving relative paths

    Returns:
        Scaled FLOPs as an integer. Returns output_tokens unchanged if config not found.

    Examples:
        >>> calculate_sflops_from_tokens(100, "configs/models/goedel_prover_v2/32b.yaml")
        3200  # 100 * 32
        >>> calculate_sflops_from_tokens(100, "configs/models/openai/oss-120b-high.yaml")
        510   # 100 * 5.1, rounded to int
    """
    effective_params = get_effective_parameters(model_config_path, base_path)

    if effective_params is None:
        # Fallback: return tokens unchanged if config not found
        return output_tokens

    sflops = output_tokens * effective_params
    return int(sflops)


# Default mapping for model names to effective parameters (for fallback)
# This provides a way to estimate SFLOPs even without model_config_path
MODEL_NAME_TO_EFFECTIVE_PARAMS = {
    # Goedel models
    "goedel-lm/goedel-formalizer-v2-8b": 8,
    "goedel-lm/goedel-formalizer-v2-32b": 32,
    "goedel-lm/goedel-prover-v2-8b": 8,
    "goedel-lm/goedel-prover-v2-32b": 32,
    # OpenAI OSS models
    "openai/gpt-oss-120b": 5.1,
    "openai/gpt-oss-20b": 3.6,
    # Qwen models
    "qwen/qwen3-4b": 4,
    "qwen/qwen3-235b": 235,
}
