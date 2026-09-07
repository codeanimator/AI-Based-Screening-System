# 🛡️ AI-Based Fake Identity & Document Screening System
### Ministry of Home Affairs / Sashastra Seema Bal (SSB) Checkpoint Terminal
**Smart India Hackathon — Problem Statement 26188**

A production-ready, dark-mode Security Operations Center (SOC) Streamlit application designed for border security checkpoints (SSB / Bureau of Immigration). Integrates multimodal GenAI vision forensics, physical EXIF analysis, rule-based chronological integrity auditing, and DeepFace 1:1 facial biometric matching into an explainable **Composite Threat Risk Index (0–100)**.

---

## 🚀 Key Architectural Modules

```
                        ┌──────────────────────────────────────────────┐
                        │      Checkpoint Ingestion & Acquisition      │
                        │   (Document Ingestion + Live Webcam Face)    │
                        └──────────────────────┬───────────────────────┘
                                               │
               ┌───────────────────────────────┼───────────────────────────────┐
               ▼                               ▼                               ▼
     [Module 1: OCR Extraction]     [Module 2: Rule Auditing]      [Module 3: Forensics]
       • Full Legal Name              • Expiry & 6-month alert       • EXIF editing scan
       • Document Number              • DOB & issue date sanity        (Photoshop, Canva)
       • Nationality & Gender         • Demographic contradiction    • Chat-app compression
       • Safe PII Masking               (Adult photo vs child DOB)     vs malicious tamper
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

1. **Module 1 (OCR Extraction & PII Masking)**:
   - Powered by Google GenAI SDK (`google.genai`) targeting `gemini-3.6-flash` with an automated fallback chain (`gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-1.5-flash`).
   - Implements strict Pydantic JSON schema output.
   - Masks sensitive personal identifiers per UIDAI and government guidelines (Aadhaar: `XXXX-XXXX-0004`, Passport: `Z****204`) with an authorized officer unmask toggle.

2. **Module 2 (Rule & Standards Validation)**:
   - Audits ICAO 9303 and national format standards.
   - Automatically checks expiration, validity duration, and chronological sanity.
   - Flags demographic contradictions (e.g., child DOB with adult photo).
   - Differentiates original enrollment dates from subsequent photo updates on Indian Aadhaar cards.

3. **Module 3 (Digital Forensics & Tamper Check)**:
   - **Metadata Forensics**: Uses `PIL.ExifTags` to scan for digital manipulation software traces (`Photoshop`, `Canva`, `GIMP`, `MS Paint`).
   - **Compression vs. Tamper Distinction**: Explicit prompt design instructs Gemini to distinguish everyday chat-app compression artifacts (uniform 8x8 DCT blocks from WhatsApp/Telegram) from localized digital fraud (spliced portrait borders, font alterations).

4. **Module 4 (Biometric Face Verification)**:
   - 1:1 facial biometric matching using `DeepFace` with the `Facenet` model.
   - Calibrated cosine distance threshold ($< 0.45$) with adjustable sensitivity for physical laminated ID cards.
   - Guarantees temporary file cleanup in `try...finally` blocks.

5. **Composite Threat Scorer & SOC Dashboard**:
   - Synthesizes all 4 threat vectors into a 0–100 risk score:
     - **Score < 30**: `STATUS: PASS / CLEARED` (Green)
     - **Score 30–65**: `STATUS: MANUAL REVIEW` (Amber)
     - **Score > 65**: `STATUS: REJECT / FORGERY ALERT` (Red)
   - One-click export of official **MHA Checkpoint Inspection Slips (.txt)** and **JSON Audit Dossiers**.

---

## 🛠️ Installation & Setup

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone <your-repository-url>
cd ssb_doc_screening

python -m venv venv
.\venv\Scripts\activate      # Windows
# source venv/bin/activate   # Linux/macOS
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Run Automated Test Suite
```bash
python test_pipeline.py
```

### 5. Launch Checkpoint Terminal
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Included Demonstration Scenarios
The dashboard includes ready-to-test scenarios in the sidebar:
- **Scenario 1: Authentic Passport** $\rightarrow$ Clean EXIF, valid validity, matching face $\rightarrow$ `PASS / CLEARED`
- **Scenario 2: Tampered ID** $\rightarrow$ Photoshop EXIF, spliced borders, altered ID $\rightarrow$ `REJECT / FORGERY ALERT`
- **Scenario 3: Demographic Contradiction** $\rightarrow$ Child DOB (2021) with Adult Photo $\rightarrow$ `REJECT / FORGERY ALERT`
- **Scenario 4: Expired Passport & Impostor** $\rightarrow$ Expired validity + Face mismatch $\rightarrow$ `REJECT / FORGERY ALERT`

---

## 📜 License
Developed for the Ministry of Home Affairs / Sashastra Seema Bal under Smart India Hackathon.
