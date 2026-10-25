"""
FastAPI Route: Lead-Level Quality Inspection & Anomaly Screening.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
import numpy as np

import core
from api.schemas import InspectionResponse, LeadInspectionItem

router = APIRouter(prefix="/inspect", tags=["Inspection"])


@router.get("/{record_id}", response_model=InspectionResponse)
def inspect_record(record_id: str):
    """Inspect an existing record for flatlines, -80 mV dropout markers, and baseline wander."""
    try:
        signal, fields = core.load_dataset_ecg(record_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Record {record_id} error: {exc}")

    lead_df = core.inspect_leads(signal)
    chars = core.calculate_characteristics(signal)

    lead_items = []
    flat_leads = []
    dropout_leads = []

    for _, row in lead_df.iterrows():
        lead_name = str(row["Lead"])
        is_flat = bool(row["Constant lead"])
        is_drop = bool(row["Contains -80 marker"])

        if is_flat:
            flat_leads.append(lead_name)
        if is_drop:
            dropout_leads.append(lead_name)

        lead_items.append(LeadInspectionItem(
            lead=lead_name,
            minimum=round(float(row["Minimum"]), 4),
            maximum=round(float(row["Maximum"]), 4),
            std=round(float(row["Standard deviation"]), 4),
            unique_values=int(row["Unique values"]),
            constant_lead=is_flat,
            contains_dropout_marker=is_drop,
        ))

    return InspectionResponse(
        record_id=record_id,
        flat_leads=flat_leads,
        dropout_leads=dropout_leads,
        characteristics={k: round(v, 5) for k, v in chars.items()},
        leads=lead_items,
    )

