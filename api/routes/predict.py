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
    )

