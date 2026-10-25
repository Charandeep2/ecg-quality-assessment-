"""
FastAPI Application Entry Point: Adaptive ECG Quality Assessment Backend.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import core
from api.routes import (
    decomposition_router,
    inspection_router,
    predict_router,
    records_router,
)
from api.schemas import HealthResponse

STATIC_DIR = PROJECT_ROOT / "static"
INDEX_FILE = STATIC_DIR / "index.html"

# Global pipeline instance
pipeline_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance
    print("Initializing ECG Quality Assessment Pipeline...")
    pipeline_instance = core.ECGPipeline()
    print("Pipeline initialized successfully! Compute device:", core.get_device())
    yield
    print("Shutting down ECG Quality Assessment API.")


app = FastAPI(
    title="Adaptive ECG Quality Assessment API",
    description=(
        "Production REST API for 12-lead ECG quality classification using "
        "Continuous Wavelet Transforms (CWT) and Higher-Order Tucker Tensor Decomposition. "
        "Includes adaptive preprocessing, 95% energy rank estimation, and macroscopic lead inspection."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend applications (Streamlit, React, Vue, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Sub-Routers
app.include_router(predict_router, prefix="/api")
app.include_router(inspection_router, prefix="/api")
app.include_router(decomposition_router, prefix="/api")
app.include_router(records_router, prefix="/api")


@app.get("/api/status", tags=["Status"])
def api_status():
    return {
        "message": "Welcome to the Adaptive ECG Quality Assessment API",
        "documentation": "/docs",
        "health": "/health",
        "version": "2.0.0",
    }


@app.get("/", tags=["Dashboard"])
def serve_dashboard():
    """Serves the interactive medical dashboard web interface."""
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE)
    return api_status()


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health", response_model=HealthResponse, tags=["Status"])
def health_check():
    """System health check and hardware verification."""
    model_loaded = core.MODEL_FILE.exists()
    records_count = 0
    if core.DATA_DIR.exists():
        records_count = len(list(core.DATA_DIR.glob("*.hea")))

    return HealthResponse(
        status="healthy",
        version="2.0.0-adaptive",
        device=str(core.get_device()),
        model_loaded=model_loaded,
        dataset_records_available=records_count,
    )

