"""
Model Training Script for Adaptive ECG Quality Assessment.

Trains the Random Forest classifier on 6,000-dimensional fixed Tucker core features
extracted from the PhysioNet CinC 2011 Set A dataset and exports the model artifact.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import core

# Paths
FEATURE_FILE = core.PROJECT_ROOT / "results" / "week3" / "features_full_dataset.csv"
OUTPUT_MODEL_FILE = core.MODEL_FILE
OUTPUT_INFO_FILE = core.MODEL_INFO_FILE

SEED = 42
N_ESTIMATORS = 100


def train():
    print("=" * 65)
    print("ECG QUALITY CLASSIFIER TRAINING")
    print("=" * 65)

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Precomputed feature dataset not found at: {FEATURE_FILE}\n"
            "Ensure the feature CSV is present before training."
        )

    print(f"Loading feature dataset: {FEATURE_FILE}")
    df = pd.read_csv(FEATURE_FILE)

    if "label" not in df.columns:
        raise ValueError("Dataset CSV must contain a 'label' column.")

    # Convert labels
    labels_text = df["label"].astype(str).str.strip().str.lower()
    y = labels_text.map(core.LABEL_MAP).to_numpy(dtype=np.int32)

    # Extract feature columns (feature_0 to feature_5999)
    feature_cols = [col for col in df.columns if col.startswith("feature_")]
    if not feature_cols:
        raise ValueError("No 'feature_' columns found in CSV.")

    print(f"Features count: {len(feature_cols)} (Expected: {core.FIXED_FEATURE_DIM})")
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    if not np.isfinite(X).all():
        raise ValueError("Features contain NaN or Inf values.")

    acceptable_count = int(np.sum(y == 0))
    unacceptable_count = int(np.sum(y == 1))

    print(f"Total records: {len(y)}")
    print(f"  • Acceptable (0):   {acceptable_count}")
    print(f"  • Unacceptable (1): {unacceptable_count}")

    # Train / Validation Split (80/20 stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    print(f"\nTraining RandomForestClassifier (n_estimators={N_ESTIMATORS}, random_state={SEED})...")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    # Validation Evaluation
    val_preds = model.predict(X_val)
    metrics = core.calculate_metrics(y_val, val_preds)

    print("\n--- Validation Performance (Holdout 20%) ---")
    print(f"Accuracy:    {metrics['accuracy'] * 100:.2f}%")
    print(f"Sensitivity: {metrics['sensitivity'] * 100:.2f}% (Unacceptable detection)")
    print(f"Specificity: {metrics['specificity'] * 100:.2f}% (Acceptable retention)")
    print(f"Precision:   {metrics['precision'] * 100:.2f}%")
    print(f"F1-Score:    {metrics['f1'] * 100:.2f}%")
    print(f"Confusion Matrix: TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}")

    # Retrain on full dataset for final production artifact
    print("\nFitting final production model on full dataset (100% data)...")
    final_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    final_model.fit(X, y)

    # Save model artifact
    OUTPUT_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": final_model,
        "n_features": X.shape[1],
        "feature_method": "fixed CWT 1-40 Hz, Tucker rank (50, 20, 6)",
        "classes": [0, 1],
        "class_mapping": core.CLASS_NAMES,
    }
    joblib.dump(artifact, OUTPUT_MODEL_FILE)
    print(f"Model artifact saved to: {OUTPUT_MODEL_FILE}")

    # Save training summary CSV
    info_df = pd.DataFrame([{
        "model_file": str(OUTPUT_MODEL_FILE),
        "feature_count": X.shape[1],
        "training_records": len(y),
        "acceptable_records": acceptable_count,
        "unacceptable_records": unacceptable_count,
        "feature_method": artifact["feature_method"],
    }])
    info_df.to_csv(OUTPUT_INFO_FILE, index=False)
    print(f"Model metadata saved to: {OUTPUT_INFO_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    train()

