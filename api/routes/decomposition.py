"""
FastAPI Route: Higher-Order Tucker Tensor Decomposition & Compression Analysis.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
import numpy as np

import core
from api.schemas import DecompositionResponse

router = APIRouter(prefix="/decompose", tags=["Tensor Decomposition"])


@router.get("/{record_id}", response_model=DecompositionResponse)
def decompose_record(
    record_id: str,
    adaptive_rank: bool = Query(True, description="Compute 95% singular-value energy rank"),
    adaptive_cwt: bool = Query(True, description="Adaptive 40 vs 50 Hz CWT"),
):
    """
    Compute CWT 3D tensor and Tucker decomposition on an ECG record.
    Returns Tucker core shape, data compression ratio, and relative Frobenius reconstruction error.
    """
    try:
        signal, fields = core.load_dataset_ecg(record_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Record {record_id} error: {exc}")

    chars = core.calculate_characteristics(signal)
    max_freq = core.select_cwt_frequency(chars) if adaptive_cwt else core.DEFAULT_MAX_FREQUENCY

    tensor, _ = core.create_cwt_tensor(signal, maximum_frequency=max_freq)

    if adaptive_rank:
        rank = core.estimate_energy_rank(tensor, max_rank=core.MAX_RANK)
    else:
        rank = core.FIXED_RANK

    core_tensor, factors = core.calculate_tucker(tensor, rank=rank)
    error = core.calculate_reconstruction_error(tensor, core_tensor, factors)
    comp_ratio = core.compression_ratio(tensor.shape, rank)
    stored_params = core.representation_size(tensor.shape, rank)

    # Extract primary 2D core slice (Time x Frequency interaction for Lead 1)
    slice_r1 = min(12, core_tensor.shape[0])
    slice_r2 = min(12, core_tensor.shape[1])
    core_slice = [
        [round(float(v), 4) for v in row]
        for row in core_tensor[:slice_r1, :slice_r2, 0]
    ]

    return DecompositionResponse(
        record_id=record_id,
        rank=list(rank),
        core_shape=list(core_tensor.shape),
        compression_ratio=round(float(comp_ratio), 2),
        reconstruction_error=round(float(error), 4),
        parameters_stored=int(stored_params),
        original_shape=list(tensor.shape),
        core_slice=core_slice,
    )

