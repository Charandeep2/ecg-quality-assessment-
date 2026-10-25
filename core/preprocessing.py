"""
Signal quality inspection, statistical characterization, and adaptive preprocessing.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from core.config import FS, LEAD_NAMES


def calculate_characteristics(signal: np.ndarray, fs: int = FS) -> Dict[str, float]:
    """
    Compute key statistical characteristics and baseline variation of the 12-lead ECG.

    Features computed:
    - mean_rms, max_rms: Root-mean-square amplitude across leads
    - mean_std, max_std: Standard deviation across leads
    - mean_abs_amplitude: Mean absolute amplitude
    - mean_baseline_variation, max_baseline_variation: Standard deviation of moving average trend
    """
    if signal.ndim != 2:
        raise ValueError("Signal must have shape (samples, leads).")

    rms_values = np.sqrt(np.mean(signal ** 2, axis=0))
    std_values = np.std(signal, axis=0)
    abs_values = np.mean(np.abs(signal), axis=0)

    # 200ms moving-average kernel to estimate baseline wander trend
    window = max(3, int(fs * 0.2))
    kernel = np.ones(window) / window
    baseline_values = []

    for lead in range(signal.shape[1]):
        trend = np.convolve(signal[:, lead], kernel, mode="same")
        baseline_values.append(float(np.std(trend)))

    return {
        "mean_rms": float(np.mean(rms_values)),
        "max_rms": float(np.max(rms_values)),
        "mean_std": float(np.mean(std_values)),
        "max_std": float(np.max(std_values)),
        "mean_abs_amplitude": float(np.mean(abs_values)),
        "mean_baseline_variation": float(np.mean(baseline_values)),
        "max_baseline_variation": float(np.max(baseline_values)),
    }


def inspect_leads(
    signal: np.ndarray, lead_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Inspect each ECG lead individually to detect physical acquisition artifacts:
    - Constant / flat leads (std <= 1e-8)
    - Clipped or missing sensor dropout markers (-80 mV artifact in CinC 2011)
    - Amplitude bounds (min, max) and count of unique digital levels
    """
    if lead_names is None:
        lead_names = LEAD_NAMES[: signal.shape[1]]

    rows = []
    for index in range(signal.shape[1]):
        values = signal[:, index]
        std_val = float(np.std(values))
        unique_count = int(np.unique(values).size)

        rows.append({
            "Lead": lead_names[index] if index < len(lead_names) else f"Lead_{index+1}",
            "Minimum": float(np.min(values)),
            "Maximum": float(np.max(values)),
            "Standard deviation": std_val,
            "Unique values": unique_count,
            "Constant lead": bool(std_val <= 1e-8),
            "Contains -80 marker": bool(
                np.any(np.isclose(values, -80.0, atol=1e-6))
            ),
        })

    return pd.DataFrame(rows)


def high_pass_filter(
    signal: np.ndarray, cutoff: float = 0.5, order: int = 4, fs: int = FS
) -> np.ndarray:
    """
    Apply a zero-phase 4th-order Butterworth high-pass filter (0.5 Hz default cutoff)
    using Second-Order Sections (SOS) to suppress baseline wander.
    """
    sos = butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    filtered = np.column_stack([
        sosfiltfilt(sos, signal[:, lead])
        for lead in range(signal.shape[1])
    ])
    return filtered.astype(np.float32)


def should_apply_highpass(
    characteristics: Dict[str, float],
    baseline_threshold: float = 0.08,
) -> bool:
    """
    Adaptive preprocessing rule (Week 6 & Week 8):
    Determines whether a high-pass filter should be applied based on the
    computed baseline variation.
    """
    return characteristics.get("mean_baseline_variation", 0.0) >= baseline_threshold


def clean_ecg_adaptive(
    signal: np.ndarray,
    fs: int = FS,
    baseline_threshold: float = 0.08,
) -> Tuple[np.ndarray, bool, Dict[str, float]]:
    """
    Execute adaptive preprocessing:
    1. Compute signal characteristics
    2. Decide whether high-pass filtering is needed
    3. Return (filtered_or_raw_signal, filter_was_applied, characteristics)
    """
    chars = calculate_characteristics(signal, fs=fs)
    apply_hp = should_apply_highpass(chars, baseline_threshold=baseline_threshold)

    processed_signal = high_pass_filter(signal, fs=fs) if apply_hp else signal.copy()
    return processed_signal, apply_hp, chars

