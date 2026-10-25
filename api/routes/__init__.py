"""
API Route Packages.
"""

from api.routes.predict import router as predict_router
from api.routes.inspection import router as inspection_router
from api.routes.decomposition import router as decomposition_router
from api.routes.records import router as records_router

__all__ = [
    "predict_router",
    "inspection_router",
    "decomposition_router",
    "records_router",
]

