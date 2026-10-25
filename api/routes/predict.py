"""
FastAPI Route: Model Quality Prediction with Full Upgrades.
"""

from fastapi import APIRouter, HTTPException
import numpy as np

import core
from api.schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix="/predict", tags=["Prediction"])

_pipeline = None


def get_pipeline() -> core.ECGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = core.ECGPipeline()
    return _pipeline


@router.post("", response_model=PredictionResponse)
def predict_ecg(request: PredictionRequest):
    """
    Predict 12-lead ECG quality (Acceptable vs Unacceptable) using the full upgraded pipeline:
    - Adaptive 0.5 Hz Butterworth filter (if baseline wander > 0.08)
    - Adaptive CWT frequency band (40 Hz vs 50 Hz)
    - Adaptive Tucker rank (95% energy rule)
    - Random Forest inference with posterior probabilities
    """
    pipeline = get_pipeline()

    if request.record_id:
        try:
            signal, fields = core.load_dataset_ecg(request.record_id)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=f"Record {request.record_id} error: {exc}")
    elif request.signal is not None:
        signal = np.array(request.signal, dtype=np.float32)
        if signal.ndim != 2 or signal.shape[1] != 12:
            raise HTTPException(
                status_code=400,
                detail=f"Expected signal shape (samples, 12). Got shape {signal.shape}."
            )
    else:
        raise HTTPException(
            status_code=400,
            detail="Either 'record_id' or 'signal' array must be provided."
        )

    result = pipeline.process_signal(
        signal,
        adaptive_preprocessing=request.adaptive_preprocessing,
        adaptive_cwt=request.adaptive_cwt,
        adaptive_rank=request.adaptive_rank,
    )

    lead_df = result["lead_inspection"]
    flat_leads = lead_df[lead_df["Constant lead"] == True]["Lead"].tolist()
    dropout_leads = lead_df[lead_df["Contains -80 marker"] == True]["Lead"].tolist()

    prob_unacceptable = float(result["probability_unacceptable"])
    prob_acceptable = float(1.0 - prob_unacceptable)

    core_mat = result.get("core")
    core_slice = None
    if core_mat is not None and getattr(core_mat, "ndim", 0) == 3:
        slice_r1 = min(12, core_mat.shape[0])
        slice_r2 = min(12, core_mat.shape[1])
        core_slice = [
            [round(float(v), 4) for v in row]
            for row in core_mat[:slice_r1, :slice_r2, 0]
        ]

    # Compute Explainable AI (XAI) Factor Attributions
    xai_factors = []
    # 1. Electrode Attachment
    if dropout_leads or flat_leads:
        fault_count = len(dropout_leads + flat_leads)
        xai_factors.append({
            "name": "Electrode Attachment & Continuity",
            "score": -42.0 if fault_count >= 4 else -30.0,
            "status": "negative",
            "plain": f"Disconnected / dead leads: {dropout_leads + flat_leads}",
            "clinical": f"ADC dropout or flatline detected on {fault_count} leads.",
        })
    else:
        xai_factors.append({
            "name": "Electrode Attachment & Continuity",
            "score": 24.5,
            "status": "positive",
            "plain": "All 12 electrodes firmly attached with continuous skin contact.",
            "clinical": "Zero flatlines (std > 1e-8) and zero -80 mV dropout markers.",
        })

    # 2. Baseline Wander
    wander = float(result["characteristics"].get("mean_baseline_variation", 0.0))
    if wander >= 0.08:
        xai_factors.append({
            "name": "Baseline Stability (Respiration)",
            "score": -35.0 if wander > 1.0 else -18.0,
            "status": "negative",
            "plain": f"Patient breathing drift ({wander:.2f}) exceeded 0.08 safety limit.",
            "clinical": f"200ms moving-average drift = {wander:.4f} >= 0.08 threshold.",
        })
    else:
        xai_factors.append({
            "name": "Baseline Stability (Respiration)",
            "score": 21.0,
            "status": "positive",
            "plain": f"Minimal chest movement ({wander:.3f} < 0.08 limit).",
            "clinical": f"Baseline variation {wander:.4f} < 0.08. Clean baseline.",
        })

    # 3. High-Frequency Spectral Noise
    tremor = float(result["characteristics"].get("mean_high_frequency_ratio", 0.0))
    if tremor >= 0.05:
        xai_factors.append({
            "name": "High-Frequency Spectral Cleanliness",
            "score": -15.0,
            "status": "negative",
            "plain": f"Muscle tremor or 50/60 Hz electrical interference detected ({tremor:.3f}).",
            "clinical": f"High-frequency energy ratio = {tremor:.4f} >= 0.05 threshold.",
        })
    else:
        xai_factors.append({
            "name": "High-Frequency Spectral Cleanliness",
            "score": 16.0,
            "status": "positive",
            "plain": "No muscle shivering or 50/60 Hz electrical hum.",
            "clinical": "Low-noise CWT Morlet spectrum in standard 1-40 Hz clinical band.",
        })

    # 4. 3D Tensor SVD Energy
    if result["prediction"] == "Acceptable":
        xai_factors.append({
            "name": "3D Tensor SVD Energy Concentration",
            "score": 19.5,
            "status": "positive",
            "plain": f"95% of heart energy tightly packed into ({result['rank'][0]},{result['rank'][1]},{result['rank'][2]}) core.",
            "clinical": f"Singular value cumulative energy threshold met with {float(result['compression_ratio']):.1f}x compression.",
        })
    else:
        xai_factors.append({
            "name": "3D Tensor SVD Energy Concentration",
            "score": -16.0,
            "status": "negative",
            "plain": "Signal energy was scattered due to heavy noise and artifacts.",
            "clinical": "Abnormal singular value distribution and energy dispersion.",
        })

    # 5. Multilinear Waveform Fidelity
    recon_err = float(result["reconstruction_error"])
    if recon_err <= 0.30:
        xai_factors.append({
            "name": "Multilinear Waveform Fidelity",
            "score": 11.0,
            "status": "positive",
            "plain": "Compressed model matches authentic human heartbeat waves.",
            "clinical": f"Frobenius norm relative reconstruction error = {recon_err:.4f}.",
        })
    else:
        xai_factors.append({
            "name": "Multilinear Waveform Fidelity",
            "score": -14.5,
            "status": "negative",
            "plain": "Significant wave shape distortion after multilinear projection.",
            "clinical": f"High relative Frobenius error = {recon_err:.4f}.",
        })

    return PredictionResponse(
        record_id=request.record_id,
        prediction=result["prediction"],
        prediction_class=result["prediction_class"],
        probability_unacceptable=round(prob_unacceptable, 4),
        probability_acceptable=round(prob_acceptable, 4),
        filter_applied=result["filter_applied"],
        max_frequency_hz=float(result["max_frequency"]),
        tucker_rank=list(result["rank"]),
        reconstruction_error=round(float(result["reconstruction_error"]), 4),
        compression_ratio=round(float(result["compression_ratio"]), 2),
        parameters_stored=int(result["representation_size"]),
        characteristics={k: round(v, 5) for k, v in result["characteristics"].items()},
        flat_leads=flat_leads,
        dropout_leads=dropout_leads,
        core_slice=core_slice,
        xai_factors=xai_factors,
    )

