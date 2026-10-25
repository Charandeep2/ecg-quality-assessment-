"""
Central configuration for the Upgraded Adaptive ECG Quality Assessment System.
"""

from pathlib import Path

# Paths
CORE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CORE_DIR.parent

# Models & Metadata Directories
MODELS_DIR = PROJECT_ROOT / "models"
METADATA_DIR = PROJECT_ROOT / "metadata"

MODEL_FILE = MODELS_DIR / "ecg_quality_fixed_tucker.joblib"
MODEL_INFO_FILE = MODELS_DIR / "ecg_quality_fixed_tucker_info.csv"

# Metadata Files
METADATA_DIR = PROJECT_ROOT / "metadata"
RESULTS_DIR = METADATA_DIR  # Compatibility alias for research results
GROUP_FILE = METADATA_DIR / "record_signal_groups.csv"
CHAR_FILE = METADATA_DIR / "signal_characteristics.csv"
LEAD_FILE = METADATA_DIR / "lead_quality_features.csv"
COMPLETE_RANK_FILE = METADATA_DIR / "complete_rank_summary_998.csv"
RANK_FILE = COMPLETE_RANK_FILE  # Default rank summary file
SUMMARY_FILE = METADATA_DIR / "FINAL_EXPERIMENT_SUMMARY.md"

# Dataset Resolution (supports local directory or parent reference)
if (PROJECT_ROOT / "dataset" / "raw" / "set-a").exists():
    DATA_DIR = PROJECT_ROOT / "dataset" / "raw" / "set-a"
elif (PROJECT_ROOT / "dataset").exists():
    DATA_DIR = PROJECT_ROOT / "dataset"
elif (PROJECT_ROOT.parent / "TENSOR-AIBS-app" / "dataset" / "raw" / "set-a").exists():
    DATA_DIR = PROJECT_ROOT.parent / "TENSOR-AIBS-app" / "dataset" / "raw" / "set-a"
else:
    DATA_DIR = PROJECT_ROOT / "dataset"

# Signal Acquisition Parameters
FS = 500  # 500 Hz standard sampling frequency
RECORD_DURATION_SEC = 10  # 10s recording
EXPECTED_SAMPLES = 5000   # 10s * 500 Hz
NUM_LEADS = 12
LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVF", "aVL",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

# Wavelet & CWT Upgrades
WAVELET = "morl"
FREQUENCY_COUNT = 100
DEFAULT_MAX_FREQUENCY = 40  # 1-40 Hz standard band
HIGH_MAX_FREQUENCY = 50     # 1-50 Hz extended noise band

# Tucker Decomposition Upgrades
FIXED_RANK = (50, 20, 6)
MAX_RANK = (50, 36, 12)
ENERGY_THRESHOLD = 0.95
FIXED_FEATURE_DIM = 6000  # 50 * 20 * 6

# Class Labels
LABEL_MAP = {
    "acceptable": 0,
    "accept": 0,
    "0": 0,
    "unacceptable": 1,
    "unaccept": 1,
    "1": 1,
}

CLASS_NAMES = {
    0: "Acceptable",
    1: "Unacceptable",
}

