"""
Continuous Wavelet Transform (CWT) and Time-Frequency Tensor Generation.
"""

from typing import Dict, Tuple

import numpy as np

try:
    import pywt
except ImportError:
    pywt = None

from core.config import (
    DEFAULT_MAX_FREQUENCY,
    FREQUENCY_COUNT,
    FS,
    HIGH_MAX_FREQUENCY,
    WAVELET,
)


def _check_pywt():
    if pywt is None:
        raise ImportError(
            "PyWavelets ('pywt') is required for Continuous Wavelet Transform. "
            "Install it using: pip install PyWavelets"
        )


def create_cwt_tensor(
    signal: np.ndarray,
    maximum_frequency: float = DEFAULT_MAX_FREQUENCY,
    frequency_count: int = FREQUENCY_COUNT,
    wavelet: str = WAVELET,
    fs: int = FS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform a 12-lead ECG signal into a 3D Continuous Wavelet Transform tensor:
    Shape: (time_samples, frequency_bins, num_leads)

    Args:
        signal (np.ndarray): Shape (samples, leads) ECG array.
        maximum_frequency (float): Upper frequency limit in Hz (typically 40 or 50).
        frequency_count (int): Number of frequency evaluation points (default 100).
        wavelet (str): Continuous wavelet family (default 'morl' - Morlet).
        fs (int): Sampling frequency in Hz.

    Returns:
        tensor (np.ndarray): Shape (samples, frequency_count, leads) float32 array.
        frequencies (np.ndarray): Frequency points array in Hz.
    """
    _check_pywt()

    if signal.ndim != 2:
        raise ValueError("ECG signal must have shape (samples, leads).")

    frequencies = np.linspace(1.0, maximum_frequency, frequency_count)
    scales = pywt.central_frequency(wavelet) * fs / frequencies
    scalograms = []

    for lead in range(signal.shape[1]):
        coefficients, _ = pywt.cwt(
            signal[:, lead],
            scales,
            wavelet,
            sampling_period=1.0 / fs,
        )
        # Transpose so dimensions are (time × frequency)
        scalograms.append(np.abs(coefficients).T)

    # Stack leads along dimension 2 -> (samples, frequencies, leads)
    tensor = np.stack(scalograms, axis=2).astype(np.float32)
    return tensor, frequencies


def compute_lead_scalogram(
    lead_signal: np.ndarray,
    maximum_frequency: float = DEFAULT_MAX_FREQUENCY,
    frequency_count: int = FREQUENCY_COUNT,
    wavelet: str = WAVELET,
    fs: int = FS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D scalogram (time × frequency) for a single ECG lead.
    """
    _check_pywt()

    frequencies = np.linspace(1.0, maximum_frequency, frequency_count)
    scales = pywt.central_frequency(wavelet) * fs / frequencies

    coefficients, _ = pywt.cwt(
        lead_signal,
        scales,
        wavelet,
        sampling_period=1.0 / fs,
    )
    scalogram = np.abs(coefficients).T.astype(np.float32)
    return scalogram, frequencies


def select_cwt_frequency(
    characteristics: Dict[str, float],
    high_freq_threshold: float = 0.05,
) -> int:
    """
    Adaptive CWT rule (Week 7/Week 8 Experiment 2):
    Selects 50 Hz if the record has substantial high-frequency noise or muscle tremor;
    otherwise defaults to standard 40 Hz.
    """
    high_freq_ratio = characteristics.get("mean_high_frequency_ratio", 0.0)
    if high_freq_ratio >= high_freq_threshold:
        return HIGH_MAX_FREQUENCY
    return DEFAULT_MAX_FREQUENCY

