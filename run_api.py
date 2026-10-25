"""
Launcher for the Adaptive ECG Quality Assessment FastAPI backend.
"""

import sys
from pathlib import Path
import uvicorn

# Ensure tensor new is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("=" * 70)
    print("STARTING ADAPTIVE ECG QUALITY ASSESSMENT FASTAPI SERVER")
    print("=" * 70)
    print("Interactive Dashboard:  http://127.0.0.1:8000/")
    print("Interactive Swagger UI: http://127.0.0.1:8000/docs")
    print("ReDoc Documentation:    http://127.0.0.1:8000/redoc")
    print("Health Check:           http://127.0.0.1:8000/health")
    print("=" * 70)

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()

