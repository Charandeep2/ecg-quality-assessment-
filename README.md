# Adaptive ECG Quality Assessment (Adaptive STP 2.0)

An upgraded, production-grade 12-lead ECG quality assessment system based on **Adaptive Continuous Wavelet Transforms (CWT)** and **Higher-Order Tucker Tensor Decomposition**.

This project modernizes and substantially upgrades the foundational methodology of *Sharma et al. (2023)* into an adaptive, leakage-free, GPU-accelerated medical signal assessment framework with a **FastAPI backend** and **Streamlit dashboard**.

---

## 🚀 Key Upgrades Over the Research Paper

1. **Leakage-Free Cross-Validation:** Fixes the original paper's data leakage flaw by implementing **`StratifiedGroupKFold`** clustered on SHA-256 duplicate signal hashes.
2. **Macroscopic Lead Quality Screening:** Automatically checks for flatline leads ($\text{std} \le 10^{-8}$) and ADC dropout flags ($-80\text{ mV}$ clipping markers).
3. **Adaptive 0.5 Hz Preprocessing:** Measures $200\text{ ms}$ moving-average baseline wander and triggers a zero-phase 4th-order SOS Butterworth high-pass filter only when needed ($\ge 0.08$).
4. **Adaptive CWT Frequency Band:** Dynamically expands CWT frequency evaluation from standard 40 Hz to 50 Hz when high-frequency noise or muscle tremor is detected ($>0.05$).
5. **95% Cumulative Energy Adaptive Tucker Rank:** Dynamically estimates Tucker decomposition ranks per mode via approximate SVD instead of hardcoding a static $(5, 5, 12)$ rank, boosting unacceptable record detection sensitivity by $+4.95\%$.
6. **High-Resolution Core Features:** Uses $(50, 20, 6) = 6,000$ dimensions to retain fine morphological dynamics with $>95\%$ compression ($23.2\times$).
7. **PyTorch GPU Engine:** Full GPU CUDA acceleration for SVD unfolding and tensor contractions.
8. **Production FastAPI Service:** Complete REST API for real-time clinical integration with interactive Swagger UI.

---

## 📁 Project Architecture

```
tensor new/
├── core/                                # Upgraded modular algorithms
│   ├── __init__.py                      # Package exports (45 symbols)
│   ├── config.py                        # Constants, paths & parameters
│   ├── data_loader.py                   # WFDB reader & duplicate detector
│   ├── preprocessing.py                 # Lead checks, flatline & Butterworth filter
│   ├── cwt.py                           # Morlet CWT & adaptive 40/50Hz selector
│   ├── tucker.py                        # 95% energy rank estimation & PyTorch Tucker
│   ├── features.py                      # 6000-dim feature vector & zero-padding
│   ├── classifier.py                    # Random Forest inference & metrics
│   ├── visualization.py                 # Plotly 12-lead, 2D scalogram, 3D volume
│   └── pipeline.py                      # Unified ECGPipeline orchestrator
├── api/                                 # FastAPI Backend Service
│   ├── __init__.py
│   ├── main.py                          # FastAPI application & CORS
│   ├── schemas.py                       # Pydantic request/response schemas
│   └── routes/                          # Modular API routes
│       ├── predict.py                   # /api/predict (quality classification)
│       ├── inspection.py                # /api/inspect (lead-level checks)
│       ├── decomposition.py             # /api/decompose (Tucker decomposition)
│       └── records.py                   # /api/records (dataset browsing)
├── models/                              # Trained ML Model Weights & Info
│   ├── ecg_quality_fixed_tucker.joblib  # Trained Random Forest Model
│   └── ecg_quality_fixed_tucker_info.csv
├── metadata/                            # Research & Evaluation Metadata
│   ├── record_signal_groups.csv         # Group-aware duplicate clusters
│   ├── signal_characteristics.csv       # Baseline & statistical metrics
│   ├── lead_quality_features.csv        # Lead-level quality features
│   ├── complete_rank_summary_998.csv    # Precomputed rank summary
│   └── FINAL_EXPERIMENT_SUMMARY.md      # Official findings summary
├── app.py                               # Modular Streamlit Interactive Dashboard
├── infer_ecg.py                         # Single-file ECG inference tool
├── train.py                             # Model training script
├── evaluate.py                          # 5-Fold Group-Aware CV evaluation
├── audit_dataset.py                     # Dataset duplicate & integrity auditor
├── run_api.py                           # FastAPI server launcher
└── requirements.txt                     # Dependencies
```

---

## ⚡ Quick Start

### 1. Test Model Inferences on any ECG (Interactive or Path)
```bash
# Interactive prompt mode:
python infer_ecg.py

# Direct file or record evaluation:
python infer_ecg.py 1002867
python infer_ecg.py "C:\path\to\patient_record.hea"
python infer_ecg.py "C:\path\to\waveform.csv"
```

### 2. Launch the FastAPI Backend
```bash
python run_api.py
```
Or directly using Uvicorn:
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser to:
- **Interactive Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### 2. Launch the Streamlit Interactive Dashboard
```bash
streamlit run app.py
```

### 3. Run Dataset Integrity & Duplicate Audit
```bash
python audit_dataset.py
```

### 4. Evaluate Research Benchmarks (5-Fold Group-Aware CV)
```bash
# View saved official benchmark summary (E1, E2, E3):
python evaluate.py --summary

# Run 5-fold cross-validation live:
python evaluate.py --cv
```

### 5. Retrain Model
```bash
python train.py
```

---

## 🔌 API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Hardware device (CUDA/CPU), model status, record count |
| `POST` | `/api/predict` | Classifies ECG quality (Acceptable vs Unacceptable) with full upgrades |
| `GET` | `/api/inspect/{record_id}` | Macroscopic lead check for flatlines, -80 mV markers & baseline wander |
| `GET` | `/api/decompose/{record_id}` | Computes CWT tensor & Tucker decomposition with compression ratio |
| `GET` | `/api/records` | Lists available dataset records with ground truth labels |
| `GET` | `/api/records/{record_id}` | Retrieves waveform points for interactive web charting |

