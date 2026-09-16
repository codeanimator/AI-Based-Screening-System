# 🛡️ SENTINEL: AI-Based Fake Identity & Document Screening System
### Ministry of Home Affairs / Sashastra Seema Bal (SSB) Checkpoint Terminal
**Smart India Hackathon — Problem Statement 26188**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ssb-checkpoint-terminal.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ONNX Runtime](https://img.shields.io/badge/inference-ONNX%20Runtime%20CPU-green.svg)](https://onnxruntime.ai/)
[![InsightFace](https://img.shields.io/badge/biometrics-InsightFace%20ArcFace-orange.svg)](https://github.com/deepinsight/insightface)
[![Zero PII](https://img.shields.io/badge/privacy-Zero--PII%20Compliant-purple.svg)](https://uidai.gov.in/)

A production-grade, dark-mode Security Operations Center (SOC) terminal application engineered for remote border security checkpoints (SSB / Bureau of Immigration). Operates **100% offline and air-gapped on standard edge hardware (Intel CPU / 8GB RAM)** without requiring external cloud API calls.

Integrates edge OCR extraction, multi-tier pixel/metadata forensics, demographic rule integrity auditing, and 1:1 facial biometric matching into an explainable **Composite Threat Risk Index (0–100)** sealed with a **cryptographic SHA-256 audit ledger**.

🌐 **Live Cloud Deployment**: [https://ssb-checkpoint-terminal.streamlit.app](https://ssb-checkpoint-terminal.streamlit.app)

---

## 🚀 Key Architectural Modules (100% Air-Gapped Edge Pipeline)

```
                        ┌──────────────────────────────────────────────┐
                        │      Checkpoint Ingestion & Acquisition      │
                        │   (Document Ingestion + Live Webcam Face)    │
                        └──────────────────────┬───────────────────────┘
                                               │
                ┌───────────────────────────────┼───────────────────────────────┐
                ▼                               ▼                               ▼
     [Module 1: Edge OCR Engine]    [Module 2: Rule Auditing]     [Module 3: Visual Forensics]
       • PaddleOCR (Offline CPU)      • Expiry & 6-month alert      • OpenCV ELA Pixel Splicing
       • ICAO 9303 MRZ Decoding       • Chronological sanity        • Moondream2 (1.8B SLM)
       • UIDAI Aadhaar / Passport     • Demographic contradiction   • EXIF Software Signatures
       • Automated Privacy Masking      (Adult photo vs child DOB)    (Photoshop, GIMP, Canva)
                │                               │                               │
                └───────────────────────────────┼───────────────────────────────┘
                                                │
                                                ▼
                               [Module 4: Biometric Matching]
                                • InsightFace ArcFace (buffalo_l ONNX)
                                • 512-D L2 Normalized Embeddings
                                • Cosine Similarity Metric (≥ 0.55)
                                • Dynamic Sensitivity Calibration
                                                │
                                                ▼
                              [Composite Threat Engine (0–100)]
                                • PASS / CLEARED (< 30)
                                • MANUAL REVIEW (30–65)
                                • REJECT / FORGERY ALERT (> 65)
                                • Zero-PII JSON Dossier + SHA-256 Audit Hash
                                • Official MHA Clearance Slip (.txt)
```

### 1. Module 1: Edge OCR Extraction & Privacy Masking
- **PaddleOCR Engine**: Runs locally on CPU via optimized ONNX/PIR pipelines without any cloud API dependencies.
- **ICAO 9303 MRZ Decoding**: Extracts standardized machine-readable travel document lines (Passports, Visas) directly into verified legal names, document numbers, nationality, and expiry dates.
- **Automated Privacy Masking**: Enforces strict Zero-PII guidelines for Aadhaar (`XXXX-XXXX-1234`) and Indian Passports (`Z****204`) with an authorized officer unmask toggle for terminal inspection.

### 2. Module 2: Rule & Standards Validation
- Audits ICAO 9303 and national identity document standards (Aadhaar, Passport, PAN, Driving License, Voter ID).
- Verifies expiration dates, validity windows, and chronological consistency (issue date vs. birth date vs. expiry date).
- Flags demographic contradictions (e.g., adult photograph detected alongside child date of birth).

### 3. Module 3: Digital Forensics & Tamper Check
- **OpenCV Error Level Analysis (ELA)**: Detects localized resaving compression anomalies, clone-stamping, and spliced portrait borders via JPEG Q=90 in-memory difference matrices.
- **Edge Vision Forensics (Moondream2 1.8B SLM)**: Quantized local SLM reasoning to identify typography alterations, font weight mismatches, and physical border discontinuities.
- **Metadata Forensics**: Scans EXIF headers for desktop image editing software signatures (`Photoshop`, `Canva`, `GIMP`, `MS Paint`).
- **Transit Resampling Profiling**: Distinguishes benign messenger recompression (WhatsApp/Telegram) from malicious localized digital manipulation.

### 4. Module 4: Biometric Face Verification
- **InsightFace ArcFace Engine**: 1:1 facial biometric matching powered by ONNX Runtime CPU (`buffalo_l` architecture).
- **Sub-Second Edge Execution**: Extracts robust 512-dimensional L2-normalized face embeddings using RetinaFace detection with Haar cascade fallback.
- **Calibrated Cosine Similarity**: Threshold ($\ge 0.55$) optimized for low-resolution, laminated, or weathered ID photos matched against live passenger webcams.

### 5. Composite Threat Scorer & SOC Dashboard
- Synthesizes all 4 threat vectors into an explainable **0–100 Composite Threat Score**:
  - **Score < 30**: `STATUS: PASS / CLEARED` (Green)
  - **Score 30–65**: `STATUS: MANUAL REVIEW` (Amber)
  - **Score > 65**: `STATUS: REJECT / FORGERY ALERT` (Red)
- **Zero-PII Compliance**: Exported inspection files enforce permanent privacy masking regardless of officer UI toggles.
- **Cryptographic Audit Ledger**: Computes a deterministic **SHA-256 digital hash** across the canonical dossier to prevent post-clearance ledger tampering.
- One-click export of official **MHA Checkpoint Clearance Slips (.txt)** and **JSON Audit Dossiers**.

---

## 🛠️ Installation & Setup (Local Environment)

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/codeanimator/AI-Based-Screening-System.git
cd AI-Based-Screening-System

python -m venv venv
.\venv\Scripts\activate      # Windows PowerShell
# source venv/bin/activate   # Linux/macOS
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install paddlepaddle paddleocr
```

### 3. Run Verification Test Suite
```bash
python test_pipeline.py
```

### 4. Launch Checkpoint Terminal
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Included Demonstration Scenarios
The dashboard includes instant-loading test scenarios in the sidebar:
- **Scenario 1: Authentic Passport** $\rightarrow$ Valid MRZ, clean EXIF, matching facial biometrics $\rightarrow$ `PASS / CLEARED`
- **Scenario 2: Tampered ID** $\rightarrow$ Photoshop EXIF signature, spliced photo border, altered ID $\rightarrow$ `REJECT / FORGERY ALERT`
- **Scenario 3: Demographic Contradiction** $\rightarrow$ Child DOB with adult facial photograph $\rightarrow$ `REJECT / FORGERY ALERT`
- **Scenario 4: Expired Passport & Impostor** $\rightarrow$ Expired validity window + Biometric mismatch $\rightarrow$ `REJECT / FORGERY ALERT`

---

## 📜 Compliance & Architecture Notes
- **Zero-PII Architecture**: Adheres to UIDAI Circular No. 11020/205/2017 regarding Aadhaar storage and masking.
- **Lightweight Edge Deployment**: Fully compatible with Linux and Windows edge boxes, optimized for resource-constrained environments (~2.5GB RAM ceiling).
- **Developed for**: Ministry of Home Affairs / Sashastra Seema Bal under Smart India Hackathon (Problem Statement 26188).
