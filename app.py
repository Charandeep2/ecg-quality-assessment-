"""
Modular Streamlit Dashboard for Adaptive ECG Quality Assessment.
Powered by the consolidated 'core' package.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

import core
from core import (
    FS,
    LEAD_NAMES,
    MAX_RANK,
    FIXED_RANK,
    WAVELET,
    FREQUENCY_COUNT,
    DATA_DIR,
    RESULTS_DIR,
    GROUP_FILE,
    CHAR_FILE,
    LEAD_FILE,
    RANK_FILE,
    COMPLETE_RANK_FILE,
    find_record,
    calculate_characteristics,
    inspect_leads,
    high_pass_filter,
    create_cwt_tensor,
    estimate_energy_rank,
    calculate_tucker,
    extract_fixed_classifier_features,
    predict_ecg_quality,
    calculate_reconstruction_error,
    representation_size,
    parse_saved_rank,
    plot_ecg,
    plot_scalogram,
    plot_3d_tensor,
    plot_core_slice,
)

warnings.filterwarnings("ignore")

DEVICE = core.get_device()

# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Adaptive ECG Quality Assessment",
    page_icon="💓",
    layout="wide",
)

st.title("Adaptive ECG Quality Assessment")
st.caption("Adaptive Scalogram–Tensor Processing (Adaptive STP)")

st.markdown(
    """
    **Project objective:** classify an ECG recording as
    **Acceptable** or **Unacceptable**.

    Explore how the input signal is analysed, how preprocessing
    is selected, how CWT creates a time-frequency representation,
    and how Tucker decomposition compresses the resulting tensor.
    """
)

# Streamlit cached wrappers
@st.cache_data
def load_project_data():
    return core.load_project_metadata()

@st.cache_data
def load_rank_results():
    return core.load_rank_results()

@st.cache_data(show_spinner=False)
def load_dataset_ecg(record_id):
    return core.load_dataset_ecg(record_id)

@st.cache_resource
def load_quality_classifier():
    return core.load_quality_classifier()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Adaptive STP Walkthrough",
        "Research Results",
        "About the Method",
    ],
)

try:
    data, groups_df, chars_df, leads_df = load_project_data()
    rank_df = load_rank_results()
except Exception as exc:
    st.error(f"Could not load project metadata: {exc}")
    st.stop()


# ============================================================
# INPUT SELECTION
# ============================================================

if page == "Adaptive STP Walkthrough":
    st.header("Step-by-Step ECG Processing")

    input_mode = st.sidebar.radio(
        "ECG input source",
        [
            "Existing dataset record",
            "Upload CSV or NPY",
        ],
    )

    raw_signal = None
    selected_record = None
    record_metadata = pd.Series(dtype=object)
    reference_label = "Not available"

    if input_mode == "Existing dataset record":
        available_records = sorted(
            data["record_id"].astype(str).unique()
        )

        selected_record = st.sidebar.selectbox(
            "Select ECG record",
            available_records,
        )

        matches = data[
            data["record_id"].astype(str) == selected_record
        ]

        if matches.empty:
            st.error("Record metadata was not found.")
            st.stop()

        record_metadata = matches.iloc[0]

        label = str(
            record_metadata.get("label", "Unknown")
        ).strip().lower()

        if label in ["0", "accept", "acceptable"]:
            reference_label = "Acceptable"
        elif label in ["1", "unaccept", "unacceptable"]:
            reference_label = "Unacceptable"

        try:
            with st.spinner("Loading ECG record..."):
                raw_signal, fields = load_dataset_ecg(
                    selected_record
                )
        except Exception as exc:
            st.error(f"Could not load ECG: {exc}")
            st.stop()

    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload ECG CSV or NPY",
            type=["csv", "npy"],
            help=(
                "CSV: 12 numeric lead columns, no header or a header. "
                "NPY: numeric array shaped samples × 12 leads."
            ),
        )

        uploaded_fs = st.sidebar.number_input(
            "Sampling frequency (Hz)",
            min_value=100,
            max_value=2000,
            value=500,
            step=50,
        )

        if uploaded_file is None:
            st.info(
                "Upload a CSV or NPY file in the sidebar to start."
            )
            st.stop()

        try:
            if uploaded_file.name.lower().endswith(".csv"):
                uploaded_df = pd.read_csv(uploaded_file)

                # Drop nonnumeric columns only if they are entirely
                # empty; otherwise require numeric signal values.
                numeric_df = uploaded_df.apply(
                    pd.to_numeric, errors="coerce"
                )

                if numeric_df.isna().any().any():
                    st.error(
                        "CSV must contain only numeric signal columns. "
                        "Remove text columns and missing values."
                    )
                    st.stop()

                raw_signal = numeric_df.to_numpy(
                    dtype=np.float32
                )
            else:
                raw_signal = np.load(
                    uploaded_file,
                    allow_pickle=False,
                ).astype(np.float32)

            if raw_signal.ndim != 2 or raw_signal.shape[1] != 12:
                st.error(
                    "Expected an array shaped samples × 12 leads, "
                    "for example (5000, 12)."
                )
                st.stop()

            if raw_signal.shape[0] < 10:
                st.error("The uploaded ECG contains too few samples.")
                st.stop()

            if not np.isfinite(raw_signal).all():
                st.error("The ECG contains NaN or infinite values.")
                st.stop()

            if int(uploaded_fs) != FS:
                st.error(
                    "This version expects 500 Hz. Resampling for "
                    "other sampling rates has not been implemented."
                )
                st.stop()

            selected_record = Path(uploaded_file.name).stem
            record_metadata = pd.Series(
                calculate_characteristics(raw_signal)
            )
            reference_label = "Unknown — uploaded ECG"

        except Exception as exc:
            st.error(f"Could not read uploaded ECG: {exc}")
            st.stop()

    # Make sure both input paths have identical processing shape.
    if raw_signal.ndim != 2 or raw_signal.shape[1] != 12:
        st.error(f"Invalid ECG shape: {raw_signal.shape}")
        st.stop()

    duration = raw_signal.shape[0] / FS

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Record", selected_record)
    c2.metric("Sampling rate", f"{FS} Hz")
    c3.metric("Leads", str(raw_signal.shape[1]))
    c4.metric("Duration", f"{duration:.2f} s")

    st.caption(
        f"Reference label: {reference_label}. "
        "This is dataset ground truth when available, not a prediction."
    )

    # ========================================================
    # STEP 1 — RAW SIGNAL
    # ========================================================

    st.divider()
    st.subheader("Step 1 — Raw ECG Signal")

    st.write(
        f"Signal shape: **{raw_signal.shape}** "
        "(time samples × leads)"
    )

    lead_audit = inspect_leads(raw_signal)
    constant_leads = lead_audit[
        lead_audit["Constant lead"]
    ]

    if not constant_leads.empty:
        st.warning(
            f"{len(constant_leads)} of 12 leads are constant or "
            "nearly constant. Interpret subsequent results cautiously."
        )

        st.dataframe(
            lead_audit,
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("View all 12 original ECG leads", expanded=True):
        st.plotly_chart(
            plot_ecg(raw_signal, "Original 12-Lead ECG"),
            use_container_width=True,
        )

    # ========================================================
    # STEP 2 — SIGNAL ANALYSIS
    # ========================================================

    st.divider()
    st.subheader("Step 2 — Signal Analysis")

    measured = calculate_characteristics(raw_signal)

    # For dataset records, use the same saved characteristic used
    # by the research pipeline if it is available.
    baseline_saved = record_metadata.get(
        "mean_baseline_variation", np.nan
    )

    if pd.notna(baseline_saved):
        measured["mean_baseline_variation"] = float(
            baseline_saved
        )

    metrics = st.columns(4)

    metrics[0].metric(
        "Mean RMS", f"{measured['mean_rms']:.4f}"
    )
    metrics[1].metric(
        "Mean standard deviation",
        f"{measured['mean_std']:.4f}",
    )
    metrics[2].metric(
        "Mean absolute amplitude",
        f"{measured['mean_abs_amplitude']:.4f}",
    )
    metrics[3].metric(
        "Baseline variation",
        f"{measured['mean_baseline_variation']:.4f}",
    )

    st.dataframe(
        pd.DataFrame(
            [
                {"Characteristic": key, "Value": value}
                for key, value in measured.items()
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # STEP 3 — ADAPTIVE PREPROCESSING
    # ========================================================

    st.divider()
    st.subheader("Step 3 — Adaptive Preprocessing")

    thresholds_file = (
        RESULTS_DIR / "complete_adaptive_pipeline"
        / "fold_thresholds.csv"
    )

    baseline_threshold = None
    threshold_description = ""

    if thresholds_file.exists():
        threshold_data = pd.read_csv(thresholds_file)

        if (
            "baseline_threshold" in threshold_data.columns
            and not threshold_data.empty
        ):
            baseline_threshold = float(
                threshold_data["baseline_threshold"].median()
            )

            threshold_description = (
                "Median saved training-fold threshold; this is "
                "illustrative, not a deployment threshold."
            )

    if baseline_threshold is None:
        values = pd.to_numeric(
            data.get(
                "mean_baseline_variation",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        ).dropna()

        if len(values):
            baseline_threshold = float(values.quantile(0.80))

            threshold_description = (
                "Exploratory dataset 80th percentile. "
                "Not a leakage-safe deployment threshold."
            )

    if baseline_threshold is None:
        st.warning(
            "No baseline threshold available. Filtering is skipped."
        )
        use_highpass = False
    else:
        use_highpass = (
            measured["mean_baseline_variation"]
            > baseline_threshold
        )

        left, right = st.columns(2)

        left.metric(
            "Baseline threshold",
            f"{baseline_threshold:.6f}",
        )
        right.metric(
            "Measured variation",
            f"{measured['mean_baseline_variation']:.6f}",
        )

        if use_highpass:
            st.warning(
                "Decision: APPLY a 0.5 Hz high-pass filter."
            )
        else:
            st.success(
                "Decision: SKIP high-pass filtering."
            )

        st.caption(threshold_description)

    if use_highpass:
        processed_signal = high_pass_filter(raw_signal)
    else:
        processed_signal = raw_signal.copy()

    st.write(
        "Filter: fourth-order Butterworth high-pass, 0.5 Hz cutoff."
    )

    signal_tab1, signal_tab2 = st.tabs(
        ["Original signal", "Processed signal"]
    )

    with signal_tab1:
        st.plotly_chart(
            plot_ecg(raw_signal, "Original ECG"),
            use_container_width=True,
        )

    with signal_tab2:
        st.plotly_chart(
            plot_ecg(processed_signal, "Processed ECG"),
            use_container_width=True,
        )

    # ========================================================
    # STEP 4 — CWT
    # ========================================================

    st.divider()
    st.subheader("Step 4 — Continuous Wavelet Transform")

    cwt_mode = st.radio(
        "CWT configuration",
        [
            "Fixed 40 Hz",
            "Fixed 50 Hz",
            "Adaptive choice",
        ],
        horizontal=True,
    )

    high_frequency_ratio = record_metadata.get(
        "mean_high_frequency_ratio", np.nan
    )

    frequency_threshold = None

    if thresholds_file.exists():
        threshold_data = pd.read_csv(thresholds_file)

        if (
            "frequency_threshold" in threshold_data.columns
            and not threshold_data.empty
        ):
            frequency_threshold = float(
                threshold_data["frequency_threshold"].median()
            )

    if cwt_mode == "Fixed 40 Hz":
        maximum_frequency = 40
        cwt_reason = "Fixed baseline setting."
    elif cwt_mode == "Fixed 50 Hz":
        maximum_frequency = 50
        cwt_reason = "Fixed extended-frequency setting."
    else:
        if frequency_threshold is None:
            ratios = pd.to_numeric(
                data.get(
                    "mean_high_frequency_ratio",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            ).dropna()

            if len(ratios):
                frequency_threshold = float(ratios.quantile(0.80))

        if (
            pd.isna(high_frequency_ratio)
            or frequency_threshold is None
        ):
            maximum_frequency = 40
            cwt_reason = (
                "High-frequency ratio unavailable; using 40 Hz fallback."
            )
        else:
            maximum_frequency = (
                50
                if float(high_frequency_ratio) > frequency_threshold
                else 40
            )

            cwt_reason = (
                f"Measured high-frequency ratio: "
                f"{float(high_frequency_ratio):.6f}; "
                f"threshold: {frequency_threshold:.6f}."
            )

    st.write(
        f"Selected maximum frequency: **{maximum_frequency} Hz**"
    )
    st.caption(cwt_reason)

    with st.spinner("Calculating CWT for all 12 leads..."):
        tensor, frequencies = create_cwt_tensor(
            processed_signal,
            maximum_frequency,
        )

    st.success("CWT completed.")

    st.write(
        f"Tensor shape (time × frequency × lead): **{tensor.shape}**"
    )
    st.write(
        f"Wavelet: **{WAVELET}** | Frequency values: "
        f"**{len(frequencies)}**"
    )

    selected_lead = st.selectbox(
        "Select lead for its scalogram",
        options=list(range(12)),
        format_func=lambda index: LEAD_NAMES[index],
    )

    st.plotly_chart(
        plot_scalogram(tensor, frequencies, selected_lead),
        use_container_width=True,
    )

    st.markdown("#### Interactive 3D Tensor")

    st.write(
        "Rotate the plot to explore the tensor. Point color represents "
        "CWT magnitude. Visualization sampling does not change the "
        "full tensor used by Tucker decomposition."
    )

    time_step = st.slider(
        "3D display time-sampling step",
        min_value=1,
        max_value=50,
        value=10,
    )

    st.plotly_chart(
        plot_3d_tensor(tensor, frequencies, time_step),
        use_container_width=True,
    )

    # ========================================================
    # STEP 5 — TUCKER DECOMPOSITION
    # ========================================================

    st.divider()
    st.subheader("Step 5 — Adaptive Tucker Decomposition")

    rank_method = st.radio(
        "Rank-selection method",
        [
            "Combined pipeline — 95% energy rule",
            "E3 — saved rank-budget search",
        ],
    )

    selected_rank = None
    rank_explanation = ""

    if rank_method.startswith("Combined"):
        with st.spinner("Estimating rank from tensor energy..."):
            selected_rank = estimate_energy_rank(tensor)

        rank_explanation = (
            "The rank is estimated from singular-value energy "
            "with a 95% target, capped at (50, 36, 12)."
        )

    else:
        if (
            not rank_df.empty
            and "selected_rank" in rank_df.columns
            and selected_record in set(
                rank_df["record_id"].astype(str)
            )
        ):
            saved_row = rank_df[
                rank_df["record_id"].astype(str)
                == selected_record
            ].iloc[0]

            try:
                selected_rank = parse_saved_rank(
                    saved_row["selected_rank"]
                )

                rank_explanation = (
                    "Rank loaded from the saved E3 rank-budget "
                    "search. Tucker decomposition is recalculated "
                    "for this visualization."
                )
            except Exception:
                selected_rank = None

        if selected_rank is None:
            st.warning(
                "No saved E3 rank is available for this record. "
                "The 95% energy-based rank will be used instead."
            )

            with st.spinner("Estimating fallback rank..."):
                selected_rank = estimate_energy_rank(tensor)

            rank_explanation = (
                "Fallback energy-based rank; this is not an E3 result."
            )

    st.write(f"Selected rank: **{selected_rank}**")
    st.caption(rank_explanation)

    with st.spinner("Calculating Tucker core and factor matrices..."):
        core, factors = calculate_tucker(tensor, selected_rank)

    with st.spinner("Calculating reconstruction error..."):
        error = calculate_reconstruction_error(
            tensor, core, factors
        )

    fixed_size = representation_size(
        tensor.shape, FIXED_RANK
    )
    selected_size = representation_size(
        tensor.shape, selected_rank
    )

    rank_cols = st.columns(4)
    rank_cols[0].metric("Original tensor", str(tensor.shape))
    rank_cols[1].metric("Fixed rank", str(FIXED_RANK))
    rank_cols[2].metric("Selected rank", str(selected_rank))
    rank_cols[3].metric("Core tensor", str(core.shape))

    st.metric(
        "Relative reconstruction error",
        f"{error:.6f}",
    )

    size_cols = st.columns(2)
    size_cols[0].metric(
        "Fixed representation size",
        f"{fixed_size:,}",
    )
    size_cols[1].metric(
        "Selected representation size",
        f"{selected_size:,}",
    )

    if selected_size < fixed_size:
        st.success(
            f"Representation size is reduced by "
            f"{100 * (1 - selected_size / fixed_size):.2f}% "
            "relative to the fixed-rank representation."
        )
    else:
        st.info(
            "The selected representation is not smaller than "
            "the fixed-rank representation."
        )

    st.markdown("#### 3D view of the Tucker core")

    st.write(
        "The core tensor is a smaller 3D tensor. The three dimensions "
        "correspond to the selected Tucker ranks."
    )

    core_fig = go.Figure(
        data=go.Scatter3d(
            x=np.repeat(
                np.arange(core.shape[0]),
                core.shape[1] * core.shape[2],
            ),
            y=np.tile(
                np.repeat(
                    np.arange(core.shape[1]),
                    core.shape[2],
                ),
                core.shape[0],
            ),
            z=np.tile(
                np.arange(core.shape[2]),
                core.shape[0] * core.shape[1],
            ),
            mode="markers",
            marker=dict(
                size=4,
                color=core.ravel(),
                colorscale="Viridis",
                colorbar=dict(title="Core value"),
                opacity=0.8,
            ),
        )
    )

    core_fig.update_layout(
        title="Tucker Core Tensor — 3D",
        scene=dict(
            xaxis_title="Time rank",
            yaxis_title="Frequency rank",
            zaxis_title="Lead rank",
        ),
        height=600,
    )

    st.plotly_chart(core_fig, use_container_width=True)

    core_axis = st.selectbox(
        "Inspect a 2D slice of the core",
        [
            "Time rank × Frequency rank",
            "Time rank × Lead rank",
            "Frequency rank × Lead rank",
        ],
    )

    st.plotly_chart(
        plot_core_slice(core, core_axis),
        use_container_width=True,
    )

    with st.expander("Inspect factor-matrix dimensions"):
        for index, factor in enumerate(factors, start=1):
            st.write(
                f"Factor matrix {index}: {factor.shape}"
            )

        st.write("Core tensor shape:", core.shape)
        st.write("Core values:", f"{core.size:,}")

    # ========================================================
    # STEP 6 — FEATURES
    # ========================================================

    st.divider()
    st.subheader("Step 6 — Feature Representation")

    feature_vector = core.ravel()

    st.write(
        f"Flattened core feature vector length: "
        f"**{feature_vector.size:,}**"
    )

    st.dataframe(
        pd.DataFrame({
            "Feature index": np.arange(
                min(30, feature_vector.size)
            ),
            "Feature value": feature_vector[:30],
        }),
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # STEP 7 — ACTUAL CLASSIFIER PREDICTION
    # ========================================================

    st.divider()
    st.subheader("Step 7 — Final ECG Quality Assessment")

    st.write(
        "The classifier uses its matching fixed 40 Hz CWT and "
        "Tucker (50, 20, 6) feature pipeline."
    )

    if st.button(
        "Predict ECG Quality",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Extracting fixed features and predicting quality..."
            ):
                prediction_result = predict_ecg_quality(raw_signal)

            predicted_label = prediction_result["label"]
            probability = prediction_result[
                "probability_unacceptable"
            ]

            st.session_state["ecg_prediction"] = {
                "record": selected_record,
                "label": predicted_label,
                "probability": probability,
                "reference": reference_label,
            }

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    saved_prediction = st.session_state.get("ecg_prediction")

    if (
        saved_prediction is not None
        and saved_prediction["record"] == selected_record
    ):
        predicted_label = saved_prediction["label"]
        probability = saved_prediction["probability"]

        if predicted_label == "Acceptable":
            st.success("Prediction: ACCEPTABLE")
        else:
            st.error("Prediction: UNACCEPTABLE")

        st.metric(
            "Model-estimated probability of Unacceptable",
            f"{probability:.1%}",
        )

        st.caption(
            "This probability is a model output, not a calibrated "
            "clinical probability. The classifier uses fixed-rank "
            "features even when the adaptive walkthrough above "
            "shows different processing decisions."
        )

        st.write(
            "Reference label:", saved_prediction["reference"]
        )

        if saved_prediction["reference"] in [
            "Acceptable", "Unacceptable"
        ]:
            if predicted_label == saved_prediction["reference"]:
                st.success("Prediction matches the dataset label.")
            else:
                st.warning(
                    "Prediction differs from the dataset label."
                )

    else:
        st.info(
            "Click Predict ECG Quality to obtain a prediction."
        )



# ============================================================
# RESEARCH RESULTS
# ============================================================

elif page == "Research Results":
    st.header("Saved Research Experiment Results")

    st.write(
        "These are saved experiment results. Opening this page "
        "does not rerun the cross-validation experiments."
    )

    experiment_files = {
        "E1 — Adaptive preprocessing": (
            RESULTS_DIR / "E1_adaptive_preprocessing"
            / "pooled_oof_results.csv"
        ),
        "E2 — Adaptive CWT": (
            RESULTS_DIR / "E2_adaptive_cwt"
            / "pooled_oof_results.csv"
        ),
        "E3 — Adaptive Tucker rank": (
            RESULTS_DIR / "E3_adaptive_tucker_rank"
            / "pooled_oof_results.csv"
        ),
        "Combined adaptive pipeline": (
            RESULTS_DIR / "complete_adaptive_pipeline"
            / "pooled_oof_results.csv"
        ),
    }

    for name, path in experiment_files.items():
        st.subheader(name)

        if path.exists():
            result = pd.read_csv(path)
            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning(
                f"Result file not found: {path.relative_to(ROOT)}"
            )

    st.subheader("Saved Tucker Rank Search")

    if not rank_df.empty:
        columns = [
            column for column in [
                "record_id",
                "label",
                "fixed_rank",
                "selected_rank",
                "fixed_error",
                "selected_error",
                "fixed_representation_size",
                "selected_representation_size",
                "representation_reduction_percent",
                "error_reduction_percent",
                "improved_both",
            ]
            if column in rank_df.columns
        ]

        st.write(f"Rank records available: **{len(rank_df)}**")

        st.dataframe(
            rank_df[columns].head(100),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No saved rank-search summary was found.")


# ============================================================
# ABOUT
# ============================================================

else:
    st.header("About the Method")

    st.markdown(
        """
        ### 1. Signal analysis
        Measure ECG characteristics and detect constant or suspicious
        leads before interpreting later processing results.

        ### 2. Adaptive preprocessing
        Select whether to apply the high-pass filter based on a
        baseline-variation measure and threshold.

        ### 3. Continuous Wavelet Transform
        Transform each lead into a time-frequency scalogram and stack
        the scalograms into a tensor.

        **Tensor dimensions:** Time × Frequency × Lead.

        ### 4. Tucker decomposition
        Decompose the tensor into a core tensor and factor matrices.
        The core tensor can be flattened into features for a classifier.

        ### 5. Classification
        The target is Acceptable or Unacceptable. A genuine prediction
        requires a compatible trained classifier.

        ### Important distinction
        E1, E2, and E3 are separate experiments. The combined pipeline
        uses a 95% energy-based rank estimator, which differs from E3's
        greedy rank-budget search.
        """
    )

    st.subheader("Runtime")
    st.write("Compute device:", str(DEVICE))

    if torch.cuda.is_available():
        st.write("GPU:", torch.cuda.get_device_name(0))
    else:
        st.write("GPU unavailable; using CPU.")
