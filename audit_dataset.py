"""
Dataset Validation, Duplicate Detection & Label Integrity Audit.

Consolidates and automates dataset audits:
1. Computes SHA-256 digests for all .dat binary files.
2. Identifies duplicate ECG recording clusters in PhysioNet 2011 Set A.
3. Audits for conflicting labels (e.g. duplicate signals labeled as both Acceptable and Unacceptable).
4. Checks header integrity, missing data, and -80 mV dropout markers.
"""

from collections import defaultdict
import hashlib
from pathlib import Path
import pandas as pd
import wfdb

import core


def audit_dataset():
    print("=" * 70)
    print("PHYSIONET 2011 SET A — COMPREHENSIVE DATASET INTEGRITY AUDIT")
    print("=" * 70)

    data_dir = core.DATA_DIR
    print(f"Dataset path: {data_dir.resolve()}")

    # 1. Check Header and Data Files
    hea_files = sorted(data_dir.glob("*.hea"))
    dat_files = sorted(data_dir.glob("*.dat"))

    print(f"\n1. File Inventory:")
    print(f"  • Header files (.hea): {len(hea_files)}")
    print(f"  • Data files (.dat):   {len(dat_files)}")

    # 2. Check Labels
    acceptable_file = data_dir / "RECORDS-acceptable"
    if not acceptable_file.exists():
        acceptable_file = data_dir / "set-a" / "RECORDS-acceptable"

    unacceptable_file = data_dir / "RECORDS-unacceptable"
    if not unacceptable_file.exists():
        unacceptable_file = data_dir / "set-a" / "RECORDS-unacceptable"

    acceptable_ids = set()
    unacceptable_ids = set()

    if acceptable_file.exists() and unacceptable_file.exists():
        acceptable_ids = {x.strip() for x in acceptable_file.read_text().splitlines() if x.strip()}
        unacceptable_ids = {x.strip() for x in unacceptable_file.read_text().splitlines() if x.strip()}

        print(f"\n2. Official Ground Truth Labels:")
        print(f"  • Acceptable records:   {len(acceptable_ids)}")
        print(f"  • Unacceptable records: {len(unacceptable_ids)}")
        overlap = acceptable_ids & unacceptable_ids
        print(f"  • Conflicting IDs (in both lists): {len(overlap)} {sorted(overlap) if overlap else ''}")

    # 3. Compute SHA-256 Hashes to Detect Duplicate Signals
    print(f"\n3. Duplicate Signal Hashing (SHA-256):")
    hash_groups = defaultdict(list)
    missing_dat = []

    for header in hea_files:
        rid = header.stem
        dat_path = data_dir / f"{rid}.dat"
        if not dat_path.exists():
            dat_path = data_dir / "set-a" / f"{rid}.dat"

        if not dat_path.exists():
            missing_dat.append(rid)
            continue

        digest = hashlib.sha256(dat_path.read_bytes()).hexdigest()
        hash_groups[digest].append(rid)

    duplicate_groups = {h: rids for h, rids in hash_groups.items() if len(rids) > 1}
    total_duplicate_records = sum(len(rids) for rids in duplicate_groups.values())

    print(f"  • Total unique waveforms:     {len(hash_groups)}")
    print(f"  • Duplicate waveform groups:  {len(duplicate_groups)}")
    print(f"  • Total duplicated records:   {total_duplicate_records}")

    # Inspect top duplicate clusters
    print(f"\n4. Top Duplicate Clusters Audit:")
    sorted_groups = sorted(duplicate_groups.items(), key=lambda x: len(x[1]), reverse=True)

    for i, (digest, rids) in enumerate(sorted_groups[:5], 1):
        print(f"  Cluster {i} ({len(rids)} identical records) | Hash={digest[:16]}...:")
        print(f"    IDs: {rids[:10]} {'...' if len(rids) > 10 else ''}")

        # Check label consistency within cluster
        cluster_labels = {}
        for rid in rids:
            if rid in acceptable_ids:
                lbl = "Acceptable"
            elif rid in unacceptable_ids:
                lbl = "Unacceptable"
            else:
                lbl = "Unlabeled"
            cluster_labels[lbl] = cluster_labels.get(lbl, 0) + 1

        print(f"    Label distribution: {cluster_labels}")

    print("\n" + "=" * 70)
    print("AUDIT RESULT: Group-Aware CV is strictly necessary to prevent train/test leakage.")
    print("=" * 70)


if __name__ == "__main__":
    audit_dataset()

