import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient
from api.main import app

print("=" * 65)
print("TESTING FASTAPI BACKEND ENDPOINTS")
print("=" * 65)

with TestClient(app) as client:
    # 1. Test Root (Medical Web Dashboard HTML)
    r_root = client.get("/")
    print("1. Root Dashboard GET /:", r_root.status_code, f"HTML bytes: {len(r_root.text)}")
    assert r_root.status_code == 200
    assert "CardioPulse" in r_root.text

    # 1b. Test API Status JSON
    r_stat = client.get("/api/status")
    print("1b. Status Endpoint GET /api/status:", r_stat.status_code, r_stat.json())
    assert r_stat.status_code == 200

    # 2. Test Health
    r_health = client.get("/health")
    print("2. Health Endpoint GET /health:", r_health.status_code, r_health.json())
    assert r_health.status_code == 200

    # 3. Test Records List
    r_records = client.get("/api/records?limit=5")
    print("3. Records Endpoint GET /api/records:", r_records.status_code, f"Returned {len(r_records.json())} records")
    assert r_records.status_code == 200

    # 4. Test Lead Inspection Endpoint
    r_inspect = client.get("/api/inspect/1002867")
    print("4. Inspect Endpoint GET /api/inspect/1002867:", r_inspect.status_code)
    assert r_inspect.status_code == 200
    data_inspect = r_inspect.json()
    print("   Flat leads:", data_inspect["flat_leads"])
    print("   Mean baseline variation:", data_inspect["characteristics"]["mean_baseline_variation"])

print("=" * 65)
print("FASTAPI BASE ENDPOINTS PASSED SUCCESSFULLY!")
print("=" * 65)

