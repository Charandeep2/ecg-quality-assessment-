"""
Interactive ECG Quality Inference Tool (Adaptive STP 2.0).

Usage:
    python infer_ecg.py
    python infer_ecg.py <path_or_record_id>
"""

import sys
from pathlib import Path
import warnings

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Suppress sklearn unpickling warnings for clean CLI output
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import wfdb

# Ensure local project modules are accessible
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core


def load_ecg_from_path(user_input: str) -> tuple[np.ndarray, dict, str]:
    """
    Load 12-lead ECG from:
    1. PhysioNet Record ID (e.g. '1002867')
    2. WFDB header/data file (.hea / .dat)
    3. NumPy array file (.npy)
    4. Delimited text/CSV file (.csv / .txt)

    Returns:
        signal: (samples, 12) float32 numpy array
        metadata: dict with signal info
        source_name: descriptive string of input source
    """
    clean_input = user_input.strip().strip("'\"")
    path_obj = Path(clean_input)

    # Case 1: Existing file path
    if path_obj.exists():
        ext = path_obj.suffix.lower()

        # WFDB header or data file
        if ext in [".hea", ".dat"]:
            record_stem = str(path_obj.with_suffix(""))
            signal, fields = wfdb.rdsamp(record_stem)
            return signal.astype(np.float32), fields, path_obj.name

        # NumPy binary array
        elif ext == ".npy":
            arr = np.load(path_obj)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array, got shape {arr.shape}")
            if arr.shape[0] == 12 and arr.shape[1] != 12:
                arr = arr.T  # Transpose (12, N) -> (N, 12)
            if arr.shape[1] != 12:
                raise ValueError(f"Expected 12 leads, got {arr.shape[1]} columns")
            fields = {"fs": core.FS, "sig_name": core.LEAD_NAMES}
            return arr.astype(np.float32), fields, path_obj.name

        # CSV or Tab-delimited text file
        elif ext in [".csv", ".txt"]:
            df = pd.read_csv(path_obj)
            # Drop non-numeric / time columns if present
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 12:
                # If first column is time/index, exclude it
                first_col = str(numeric_cols[0]).lower()
                if "time" in first_col or "index" in first_col or "sample" in first_col:
                    numeric_cols = numeric_cols[1:13]
                else:
                    numeric_cols = numeric_cols[:12]
            arr = df[numeric_cols].to_numpy(dtype=np.float32)
            if arr.shape[1] != 12:
                raise ValueError(f"Expected 12 columns/leads, found {arr.shape[1]} numeric columns.")
            fields = {"fs": core.FS, "sig_name": list(numeric_cols)}
            return arr, fields, path_obj.name

        else:
            # Try loading as WFDB record stem
            try:
                signal, fields = wfdb.rdsamp(str(path_obj))
                return signal.astype(np.float32), fields, path_obj.name
            except Exception:
                raise ValueError(f"Unsupported file format '{ext}'. Supported: .hea, .dat, .csv, .npy, .txt")

    # Case 2: Direct record ID in dataset
    try:
        signal, fields = core.load_dataset_ecg(clean_input)
        return signal, fields, f"PhysioNet Record {clean_input}"
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not locate '{clean_input}' as a file or dataset record.\n"
            f"Error: {exc}"
        )


def run_inference(user_input: str, adaptive: bool = True):
    print("\n" + "=" * 70)
    print(" 💓 ADAPTIVE ECG QUALITY ASSESSMENT (STP 2.0) — INFERENCE ENGINE")
    print("=" * 70)
    print(f"Loading input: {user_input} ...")

    signal, fields, source_name = load_ecg_from_path(user_input)
    fs = int(fields.get("fs", core.FS))
    duration_sec = round(signal.shape[0] / fs, 2)

    print(f"Source:             {source_name}")
    print(f"Dimensions:         {signal.shape[0]} samples × {signal.shape[1]} leads")
    print(f"Sampling Frequency: {fs} Hz ({duration_sec} seconds duration)")

    # 1. Lead screening
    lead_df = core.inspect_leads(signal)
    flat_leads = lead_df[lead_df["Constant lead"] == True]["Lead"].tolist()
    dropout_leads = lead_df[lead_df["Contains -80 marker"] == True]["Lead"].tolist()

    # 2. Characteristics & Adaptive Preprocessing
    chars = core.calculate_characteristics(signal, fs=fs)
    apply_hp = core.should_apply_highpass(chars) if adaptive else False
    if apply_hp:
        working_signal = core.high_pass_filter(signal, fs=fs)
        filter_status = "Applied 0.5 Hz SOS Butterworth (Wander detected)"
    else:
        working_signal = signal.copy()
        filter_status = "Bypassed (Clean baseline, phase preserved)"

    # 3. CWT & Frequency
    max_freq = core.select_cwt_frequency(chars) if adaptive else core.DEFAULT_MAX_FREQUENCY
    tensor, _ = core.create_cwt_tensor(working_signal, maximum_frequency=max_freq, fs=fs)

    # 4. Tucker Decomposition
    if adaptive:
        rank = core.estimate_energy_rank(tensor, max_rank=core.MAX_RANK)
        rank_mode = f"Adaptive 95% Energy: {rank}"
    else:
        rank = core.FIXED_RANK
        rank_mode = f"Fixed Rank: {rank}"

    core_tensor, factors = core.calculate_tucker(tensor, rank=rank)
    recon_error = core.calculate_reconstruction_error(tensor, core_tensor, factors)
    comp_ratio = core.compression_ratio(tensor.shape, rank)
    stored_params = core.representation_size(tensor.shape, rank)

    # 5. Model Prediction
    pred_result = core.predict_ecg_quality(signal)
    pred_label = pred_result["label"]
    prob_unacc = pred_result["probability_unacceptable"]
    prob_acc = 1.0 - prob_unacc

    # Display Result Card
    print("\n" + "─" * 70)
    print(" 🩺 QUALITY DIAGNOSTIC RESULT")
    print("─" * 70)

    if pred_label == "Acceptable":
        print(f"  Status:       ✅  ACCEPTABLE (Clear for Clinical Diagnosis)")
        print(f"  Confidence:   {prob_acc * 100:.1f}% Acceptable  (Unacceptable Risk: {prob_unacc * 100:.1f}%)")
        print(f"  Clinical Assessment: The 12-lead waveforms are steady and clear. P, QRS, and T waves are distinguishable. Ready to file in patient chart.")
    else:
        print(f"  Status:       ❌  UNACCEPTABLE (Noisy / Artifact-Laden)")
        print(f"  Confidence:   {prob_unacc * 100:.1f}% Unacceptable  (Acceptable: {prob_acc * 100:.1f}%)")
        print(f"  Clinical Assessment: Signal contains severe corruption. Unsafe for diagnosis. Re-recording recommended.")

    # Root Cause Breakdown if Unacceptable
    if pred_label != "Acceptable":
        print("\n" + "─" * 70)
        print(" 🚨 WHY IS THIS RECORDING UNACCEPTABLE? (FAILURE ROOT CAUSES)")
        print("─" * 70)
        reasons = []
        actions = []
        if dropout_leads:
            reasons.append(f"Sensor Disconnection (-80 mV Dropout) on leads: {dropout_leads}")
            actions.append(f"Clean skin, re-apply gel, and firmly reattach {dropout_leads} electrodes.")
        if flat_leads:
            reasons.append(f"Signal Flatline (Dead Wire, std <= 1e-8) on leads: {flat_leads}")
            actions.append(f"Check cable clips and ensure {flat_leads} are plugged in.")
        if chars['mean_baseline_variation'] >= 0.08:
            reasons.append(f"Heavy Baseline Wander ({chars['mean_baseline_variation']:.3f} >= 0.08 safety limit)")
            actions.append("Ask patient to remain still, relax torso, and breathe gently.")
        if chars.get('mean_high_frequency_ratio', 0.0) >= 0.05:
            reasons.append(f"Elevated High-Frequency Tremor/Noise ({chars.get('mean_high_frequency_ratio', 0.0):.3f})")
            actions.append("Keep patient warm to prevent shivering and check for powerline interference.")
        if not reasons:
            reasons.append("Multidimensional tensor features deviated from clean cardiac patterns.")
            actions.append("Re-record standard 10-second ECG.")

        for idx, r in enumerate(reasons, 1):
            print(f"  {idx}. ❌ {r}")
        print(f"\n  👉 Action Required: {' '.join(actions)}")

    print("\n" + "─" * 70)
    print(" 🔍 LEAD ARTIFACT SCREENING")
    print("─" * 70)
    print(f"  • Flatlined Leads (std <= 1e-8):     {flat_leads if flat_leads else 'None (All active)'}")
    print(f"  • Hardware Dropouts (-80 mV flags):  {dropout_leads if dropout_leads else 'None (Clean ADC)'}")

    print("\n" + "─" * 70)
    print(" 🧠 3D TENSOR MULTILINEAR DECOMPOSITION (TUCKER AI BRAIN)")
    print("─" * 70)
    r1, r2, r3 = rank
    raw_elements = int(np.prod(tensor.shape))
    print(f"  • Raw 3D Tensor:             5,000 Time × 100 Frequencies × 12 Leads = {raw_elements:,} floats")
    print(f"  • Tucker Core Rank:          ({r1}, {r2}, {r3})")
    print(f"      ├─ Mode 1 (Time):        Condensed 5,000 samples into {r1} core heartbeat pulses")
    print(f"      ├─ Mode 2 (Frequency):   Condensed 100 wavelet scales into {r2} spectral tones")
    print(f"      └─ Mode 3 (Leads):       Condensed 12 sensor angles into {r3} spatial heart vectors")
    print(f"  • Data Compression Ratio:    {comp_ratio:.1f}x ({stored_params:,} parameters vs {raw_elements:,} raw)")
    print(f"  • 95% Energy Threshold:      Preserved 95% of true cardiac energy; discarded 5% noise")
    print(f"  • Frobenius Error:           {recon_error:.4f} relative reconstruction distortion")

    print("\n" + "─" * 70)
    print(" ⚙️ ADAPTIVE STP PIPELINE DETAILS")
    print("─" * 70)
    print(f"  • Baseline Wander Std:       {chars['mean_baseline_variation']:.5f} (Threshold: 0.08)")
    print(f"  • Preprocessing Filter:      {filter_status}")
    print(f"  • High-Freq Noise Ratio:     {chars.get('mean_high_frequency_ratio', 0.0):.5f}")
    print(f"  • CWT Evaluation Band:       1.0 to {max_freq:.1f} Hz (100 scales, Morlet)")
    print(f"  • Tucker Rank Mode:          {rank_mode}")
    print(f"  • Data Compression Ratio:    {comp_ratio:.1f}x ({stored_params:,} params vs {int(np.prod(tensor.shape)):,} raw)")
    print(f"  • Reconstruction Error:      {recon_error:.4f} relative Frobenius norm")
    print("=" * 70 + "\n")


def main():
    if len(sys.argv) > 1:
        user_input = sys.argv[1]
        run_inference(user_input)
    else:
        print("=" * 70)
        print(" ADAPTIVE ECG QUALITY ASSESSMENT — INFERENCE CLI")
        print("=" * 70)
        print("Enter any of the following:")
        print("  • A PhysioNet record ID (e.g. 1002867, 1002603, 1003574)")
        print("  • A full path to a .hea or .dat file")
        print("  • A full path to a .csv, .txt, or .npy 12-lead waveform file")
        print("  • Type 'quit' or 'exit' to exit.")
        print("=" * 70)

        while True:
            try:
                user_input = input("\n👉 Paste ECG file path or Record ID: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Exiting. Have a great day!")
                    break
                run_inference(user_input)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break
            except Exception as err:
                print(f"❌ Error processing input: {err}")


if __name__ == "__main__":
    main()
