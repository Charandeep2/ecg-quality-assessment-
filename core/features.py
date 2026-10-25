"""
Feature extraction pipeline from CWT tensors and Tucker decomposition cores.
"""

from typing import Optional, Tuple

import numpy as np

from core.config import (
    DEFAULT_MAX_FREQUENCY,
    FIXED_FEATURE_DIM,
    FIXED_RANK,
    MAX_RANK,
)
from core.cwt import create_cwt_tensor
from core.tucker import calculate_tucker


def extract_fixed_classifier_features(
    signal: np.ndarray,
    maximum_frequency: float = DEFAULT_MAX_FREQUENCY,
    rank: Tuple[int, int, int] = FIXED_RANK,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the exact 6,000-dimensional feature vector expected by the trained
    ecg_quality_fixed_tucker model:
    1. CWT with 1-40 Hz, 100 frequencies, Morlet wavelet -> Tensor of shape (samples, 100, 12)
    2. Fixed Tucker decomposition with rank (50, 20, 6) -> Core of shape (50, 20, 6)
    3. Flattened core tensor -> 1 × 6000 feature vector

    Args:
        signal (np.ndarray): Shape (samples, 12) ECG array.
        maximum_frequency (float): Upper frequency bound (default 40 Hz).
        rank (tuple): Tucker decomposition rank (default (50, 20, 6)).

    Returns:
        features (np.ndarray): Shape (1, 6000) float32 feature array.
        tensor (np.ndarray): The intermediate CWT tensor.
        core (np.ndarray): The 3D Tucker core tensor.
    """
    if signal.ndim != 2 or signal.shape[1] != 12:
        raise ValueError("Expected ECG input with shape (samples, 12).")

    if not np.isfinite(signal).all():
        raise ValueError("ECG signal contains invalid numeric values (NaN or Inf).")

    # Generate CWT tensor
    tensor, _ = create_cwt_tensor(
        signal.astype(np.float32),
        maximum_frequency=maximum_frequency,
    )

    if tensor.shape[1:] != (100, 12):
        raise ValueError(f"Unexpected CWT tensor shape: {tensor.shape}")

    # Perform Tucker decomposition
    core, _ = calculate_tucker(tensor, rank=rank)

    # Flatten core to 1D feature vector
    features = core.reshape(1, -1).astype(np.float32)

    if features.shape[1] != FIXED_FEATURE_DIM:
        raise ValueError(
            f"Expected {FIXED_FEATURE_DIM} features; extracted {features.shape[1]}."
        )

    return features, tensor, core


def pad_core_tensor(
    core: np.ndarray,
    target_shape: Tuple[int, int, int] = MAX_RANK,
) -> np.ndarray:
    """
    Pad an adaptive-rank Tucker core tensor with zeros up to target_shape (e.g. (50, 36, 12)).
    Ensures that variable-sized adaptive cores can be mapped into a uniform feature dimension.
    """
    padded = np.zeros(target_shape, dtype=np.float32)
    slices = tuple(slice(0, min(core.shape[i], target_shape[i])) for i in range(3))
    padded[slices] = core[slices]
    return padded


def extract_adaptive_core_features(
    tensor: np.ndarray,
    rank: Tuple[int, int, int],
    target_shape: Tuple[int, int, int] = MAX_RANK,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose tensor using an adaptive rank and extract zero-padded fixed-size features.

    Returns:
        features (np.ndarray): Flattened padded core of shape (1, prod(target_shape)).
        core (np.ndarray): The raw unpadded core tensor.
    """
    core, _ = calculate_tucker(tensor, rank=rank)
    padded = pad_core_tensor(core, target_shape=target_shape)
    features = padded.reshape(1, -1).astype(np.float32)
    return features, core

