# 🛡️ AI-Based Fake Identity & Document Screening System
### Ministry of Home Affairs / Sashastra Seema Bal (SSB) Checkpoint Terminal
**Smart India Hackathon — Problem Statement 26188**

A production-ready, dark-mode Security Operations Center (SOC) Streamlit application engineered for remote border security checkpoints (SSB / Bureau of Immigration). Operates **100% offline and air-gapped on standard edge hardware (Intel CPU / 8GB RAM)** without any external cloud API dependencies.

Integrates edge OCR extraction, multi-tier pixel/metadata forensics, demographic rule integrity auditing, and 1:1 facial biometric matching into an explainable **Composite Threat Risk Index (0–100)**.

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
                                • DeepFace FaceNet Model
                                • Cosine Distance Metric (< 0.45)
                                • Dynamic Sensitivity Calibration
                                                │
                                                ▼
                              [Composite Threat Engine (0–100)]
                                • PASS / CLEARED (< 30)
                                • MANUAL REVIEW (30–65)
                                • REJECT / FORGERY ALERT (> 65)
                                • Official MHA Clearance Slip (.txt)
                                • Structured JSON Audit Dossier
```

1. **Module 1 (Edge OCR Extraction & Privacy Masking)**:
   - **PaddleOCR Engine**: Runs locally on CPU via optimized ONNX/PIR pipelines without any cloud API calls.
   - **ICAO 9303 MRZ Decoding**: Extracts standardized machine-readable travel document lines (Passports, Visas) directly into legal full names, document numbers, and dates.
   - **Privacy Masking**: Enforces UIDAI guidelines for Aadhaar (`XXXX-XXXX-1234`) and Indian Passports (`Z****204`) with an authorized officer unmask toggle.

2. **Module 2 (Rule & Standards Validation)**:
   - Audits ICAO 9303 and national identity standards (Aadhaar, Passport, PAN, Driving License, Voter ID).
   - Verifies expiration dates, validity durations, and chronological validity.
   - Flags demographic contradictions (e.g., child DOB with adult photograph).

3. **Module 3 (Digital Forensics & Tamper Check)**:
   - **OpenCV Error Level Analysis (ELA)**: Detects localized resaving compression anomalies, clone-stamping, and spliced portrait borders at JPEG Q=90 in-memory difference matrices.
   - **Edge Vision Forensics (Moondream2 1.8B SLM)**: Quantized local SLM reasoning to identify typography alterations, font mismatches, and physical border discontinuities.
   - **Metadata Forensics**: Scans EXIF headers for desktop image editing software signatures (`Photoshop`, `Canva`, `GIMP`, `MS Paint`).
   - **Chat-App Transit Profiling**: Distinguishes benign WhatsApp/Telegram compression from malicious localized digital forgery.

4. **Module 4 (Biometric Face Verification)**:
   - 1:1 facial biometric matching using `DeepFace` with the `Facenet` model.
   - Calibrated cosine distance threshold ($< 0.45$) with adjustable sensitivity for laminated ID cards.
   - Live checkpoint webcam capture or photo upload.

5. **Composite Threat Scorer & SOC Dashboard**:
   - Synthesizes all 4 threat vectors into a normalized 0–100 risk score:
     - **Score < 30**: `STATUS: PASS / CLEARED` (Green)
     - **Score 30–65**: `STATUS: MANUAL REVIEW` (Amber)
     - **Score > 65**: `STATUS: REJECT / FORGERY ALERT` (Red)
   - One-click export of official **MHA Checkpoint Inspection Slips (.txt)** and **JSON Audit Dossiers**.

---

## 🛠️ Installation & Setup (Local Air-Gapped)

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone <your-repository-url>
cd ssb_doc_screening

python -m venv venv
.\venv\Scripts\activate      # Windows PowerShell
# source venv/bin/activate   # Linux/macOS
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install paddlepaddle paddleocr
```

### 3. Run Automated Pipeline Test Suite
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
The dashboard includes ready-to-test scenarios in the sidebar:
- **Scenario 1: Authentic Passport** $\rightarrow$ Clean EXIF, valid validity, matching face $\rightarrow$ `PASS / CLEARED`
- **Scenario 2: Tampered ID** $\rightarrow$ Photoshop EXIF, spliced borders, altered ID $\rightarrow$ `REJECT / FORGERY ALERT`
- **Scenario 3: Demographic Contradiction** $\rightarrow$ Child DOB with Adult Photo $\rightarrow$ `REJECT / FORGERY ALERT`
- **Scenario 4: Expired Passport & Impostor** $\rightarrow$ Expired validity + Face mismatch $\rightarrow$ `REJECT / FORGERY ALERT`

---

## 📜 License
Developed for the Ministry of Home Affairs / Sashastra Seema Bal under Smart India Hackathon.
