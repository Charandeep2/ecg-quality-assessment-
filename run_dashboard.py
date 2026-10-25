"""
CardioPulse AI — Interactive Medical Dashboard Launcher.

Starts the FastAPI server and launches the medical dashboard in your default browser.
"""

import sys
import threading
import time
import webbrowser
from pathlib import Path
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def open_browser():
    time.sleep(1.5)
    url = "http://127.0.0.1:8000/"
    print(f"\n[CardioPulse AI] Launching interactive medical dashboard at: {url}")
    webbrowser.open(url)


def main():
    print("=" * 70)
    print(" 💓 CARDIOPULSE AI — ADAPTIVE ECG QUALITY CLINICAL DASHBOARD")
    print("=" * 70)
    print("  • Medical Dashboard:    http://127.0.0.1:8000/")
    print("  • Interactive API Docs: http://127.0.0.1:8000/docs")
    print("  • System Health:        http://127.0.0.1:8000/health")
    print("=" * 70)

    # Launch browser in background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Run FastAPI server with Uvicorn
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()

