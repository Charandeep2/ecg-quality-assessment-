"""
Data loading and dataset management routines for PhysioNet Challenge 2011 ECG records.
"""

from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb

from core.config import (
    CHAR_FILE,
    COMPLETE_RANK_FILE,
    DATA_DIR,
    FS,
    GROUP_FILE,
    LEAD_FILE,
    RANK_FILE,
)


def find_record(record_id: str, data_dir: Optional[Path] = None) -> Path:
    """
    Locate the WFDB record file (.hea) for a given record ID.
    Searches both root dataset dir and possible nested subdirectories.
    """
    search_dir = Path(data_dir) if data_dir else DATA_DIR
    record_id = str(record_id).strip()

    candidates = [
        search_dir / record_id,
        search_dir / "set-a" / record_id,
    ]

    for path in candidates:
        if Path(str(path) + ".hea").exists():
            return path

    raise FileNotFoundError(
        f"ECG record {record_id} was not found in {search_dir}"
    )


def load_dataset_ecg(
    record_id: str, data_dir: Optional[Path] = None
) -> Tuple[np.ndarray, dict]:
    """
    Load a 12-lead ECG recording by record ID using WFDB.
    Validates sampling frequency, shape (N, 12), and numeric sanity.

    Returns:
        signal (np.ndarray): Shape (samples, 12) float32 array
        fields (dict): WFDB header metadata dictionary
    """
    path = find_record(record_id, data_dir=data_dir)
    signal, fields = wfdb.rdsamp(str(path))

    if int(fields.get("fs", 0)) != FS:
        raise ValueError(
            f"Expected sampling rate {FS} Hz; received {fields.get('fs')} Hz."
        )

    if signal.ndim != 2 or signal.shape[1] != 12:
        raise ValueError(
            f"Expected a (samples, 12) array; got shape {signal.shape}."
        )

    if not np.isfinite(signal).all():
        raise ValueError(f"ECG record {record_id} contains invalid numeric values (NaN or Inf).")

    return signal.astype(np.float32), fields


def read_csv_cached(path: Path) -> pd.DataFrame:
    """Read a CSV file if it exists, otherwise return an empty DataFrame."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_project_metadata() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and merge dataset metadata:
    - Group-aware cross-validation signal clusters (Week 4)
    - Signal characteristics (Week 5)
    - Lead-level quality features (Week 8)

    Returns:
        (merged_data, groups_df, characteristics_df, lead_features_df)
    """
    groups = read_csv_cached(GROUP_FILE)
    characteristics = read_csv_cached(CHAR_FILE)
    lead_features = read_csv_cached(LEAD_FILE)

    if groups.empty:
        raise FileNotFoundError(
            f"Required dataset group metadata not found at: {GROUP_FILE}"
        )

    for frame in [groups, characteristics, lead_features]:
        if not frame.empty and "record_id" in frame.columns:
            frame["record_id"] = frame["record_id"].astype(str).str.strip()

    data = groups.copy()

    if not characteristics.empty:
        selected_char_cols = [
            col for col in [
                "record_id",
                "mean_baseline_variation",
                "mean_high_frequency_ratio",
                "mean_rms",
                "mean_std",
                "mean_abs_amplitude",
                "mean_powerline_ratio",
            ]
            if col in characteristics.columns
        ]
        data = data.merge(
            characteristics[selected_char_cols],
            on="record_id",
            how="left",
            validate="one_to_one",
        )

    if not lead_features.empty:
        selected_lead_cols = [
            col for col in lead_features.columns
            if col not in ["label"]
        ]
        data = data.merge(
            lead_features[selected_lead_cols],
            on="record_id",
            how="left",
            suffixes=("", "_lead"),
            validate="one_to_one",
        )

    return data, groups, characteristics, lead_features


def load_rank_results() -> pd.DataFrame:
    """Load precomputed Tucker decomposition rank results (Week 2 & recovery)."""
    target_path = COMPLETE_RANK_FILE if COMPLETE_RANK_FILE.exists() else RANK_FILE
    result = read_csv_cached(target_path)

    if not result.empty and "record_id" in result.columns:
        result["record_id"] = result["record_id"].astype(str).str.strip()

    return result


def detect_duplicate_records(data_dir: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Scan the raw dataset directory and compute SHA-256 digests for all .dat files.
    Identifies duplicate recording groups where two or more record IDs share
    identical byte contents.

    Returns:
        dict: {sha256_hash: [record_id_1, record_id_2, ...]} for all groups with >1 record.
    """
    search_dir = Path(data_dir) if data_dir else DATA_DIR
    hash_groups = defaultdict(list)

    for header in sorted(search_dir.glob("*.hea")):
        record_id = header.stem
        data_file = search_dir / f"{record_id}.dat"
        if data_file.exists():
            digest = hashlib.sha256(data_file.read_bytes()).hexdigest()
            hash_groups[digest].append(record_id)

    duplicates = {digest: rids for digest, rids in hash_groups.items() if len(rids) > 1}
    return duplicates

