"""
Cross-Validation Evaluation & Research Benchmark Script.

Executes 5-Fold Stratified Group-Aware Cross-Validation (StratifiedGroupKFold)
grouped by duplicate signal hash clusters to ensure zero data leakage, and summarizes
the findings across Experiments E1, E2, and E3.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

import core


def display_saved_experiment_benchmarks():
    """Display the official benchmark results across the three core experiments."""
    summary_path = core.RESULTS_DIR / "FINAL_EXPERIMENT_SUMMARY.md"
    if not summary_path.exists():
        summary_path = core.RESULTS_DIR / "week8" / "FINAL_EXPERIMENT_SUMMARY.md"

    print("=" * 75)
    print("OFFICIAL RESEARCH EXPERIMENT BENCHMARKS (PhysioNet 2011 Set A)")
    print("=" * 75)

    if summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"))
    else:
        print("Summary report not found at:", summary_path)
    print("=" * 75)


def run_group_aware_cv(n_splits=5, seed=42):
    """
    Run 5-Fold Stratified Group-Aware Cross-Validation on the fixed 6000-dimensional
    Tucker core feature set.
    """
    feature_file = core.RESULTS_DIR / "week3" / "features_full_dataset.csv"
    group_file = core.GROUP_FILE

    print("=" * 75)
    print(f"RUNNING {n_splits}-FOLD STRATIFIED GROUP-AWARE CROSS-VALIDATION")
    print("=" * 75)

    if not feature_file.exists():
        raise FileNotFoundError(f"Feature dataset not found: {feature_file}")
    if not group_file.exists():
        raise FileNotFoundError(f"Group metadata file not found: {group_file}")

    features_df = pd.read_csv(feature_file)
    groups_df = pd.read_csv(group_file)

    # Standardize record IDs
    features_df["record_id"] = features_df["record_id"].astype(str).str.strip()
    groups_df["record_id"] = groups_df["record_id"].astype(str).str.strip()

    merged = features_df.merge(groups_df[["record_id", "signal_hash"]], on="record_id", how="inner")
    print(f"Total evaluated records: {len(merged)}")
    print(f"Unique signal clusters (groups): {merged['signal_hash'].nunique()}")

    # Extract X, y, groups
    feature_cols = [c for c in merged.columns if c.startswith("feature_")]
    X = merged[feature_cols].to_numpy(dtype=np.float32)

    labels = merged["label"].astype(str).str.strip().str.lower().map(core.LABEL_MAP).to_numpy(dtype=int)
    groups = merged["signal_hash"].to_numpy()

    # Stratified Group K-Fold
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    fold_predictions = np.zeros(len(labels), dtype=int)

    print(f"\nEvaluating across {n_splits} folds...")
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, labels, groups=groups), 1):
        X_train, y_train = X[train_idx], labels[train_idx]
        X_test, y_test = X[test_idx], labels[test_idx]

        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        clf.fit(X_train, y_train)
        fold_predictions[test_idx] = clf.predict(X_test)

        fold_metrics = core.calculate_metrics(y_test, fold_predictions[test_idx])
        print(f"  • Fold {fold}: Acc={fold_metrics['accuracy']*100:.2f}%, "
              f"Sens={fold_metrics['sensitivity']*100:.2f}%, Spec={fold_metrics['specificity']*100:.2f}%, "
              f"F1={fold_metrics['f1']*100:.2f}%")

    # Pooled out-of-fold metrics
    pooled = core.calculate_metrics(labels, fold_predictions)
    print("\n" + "=" * 75)
    print("POOLED OUT-OF-FOLD CROSS-VALIDATION RESULTS:")
    print("=" * 75)
    print(f"Accuracy:    {pooled['accuracy'] * 100:.2f}%")
    print(f"Precision:   {pooled['precision'] * 100:.2f}%")
    print(f"Sensitivity: {pooled['sensitivity'] * 100:.2f}% (Detection of unacceptable ECGs)")
    print(f"Specificity: {pooled['specificity'] * 100:.2f}% (Retention of acceptable ECGs)")
    print(f"F1-Score:    {pooled['f1'] * 100:.2f}%")
    print(f"Confusion Matrix: TN={pooled['tn']}, FP={pooled['fp']}, FN={pooled['fn']}, TP={pooled['tp']}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ECG Quality Models and Benchmarks")
    parser.add_argument("--cv", action="store_true", help="Run 5-Fold Stratified Group-Aware Cross-Validation")
    parser.add_argument("--summary", action="store_true", help="Display official experiment summaries (E1, E2, E3)")
    args = parser.parse_args()

    if args.cv:
        run_group_aware_cv()
    else:
        display_saved_experiment_benchmarks()

