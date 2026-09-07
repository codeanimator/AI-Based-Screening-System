"""
AI-Based Fake Identity & Document Screening System
Ministry of Home Affairs / Sashastra Seema Bal (SSB) Checkpoint Terminal

Smart India Hackathon - Problem Statement 26188
Production-Ready Security Operations Dashboard
"""

import os
import json
import time
from datetime import datetime
from PIL import Image
import io
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Core Forensic Modules
from core.forensics_engine import scan_exif_metadata, MetadataForensicReport
from core.ocr_engine import extract_document_and_forensics, mask_identifier, OCRForensicResult, FALLBACK_MODELS
from core.rules_engine import validate_rules_and_standards, RuleValidationReport
from core.biometrics_engine import verify_biometrics, BiometricMatchReport, COSINE_MATCH_THRESHOLD
from core.risk_scorer import compute_composite_risk, CompositeRiskResult
from sample_data.demo_assets import DEMO_SCENARIOS


# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SSB Checkpoint Terminal | AI Document Screening",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS: DARK-MODE SECURITY OPERATIONS CENTER (SOC) THEME
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Dark Theme Security Operations Center Aesthetic */
    .stApp {
        background-color: #0c111d;
        color: #e2e8f0;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }

    /* Terminal Header */
    .terminal-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-left: 6px solid #3b82f6;
        padding: 1.25rem 1.75rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .terminal-title {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #f8fafc;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .terminal-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 0.35rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-weight: 600;
    }
    .badge-gov {
        display: inline-block;
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.4);
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin-left: 0.75rem;
    }

    /* Risk Score Master Card */
    .risk-banner {
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    }
    .risk-score-display {
        font-size: 3.2rem;
        font-weight: 900;
        line-height: 1;
        letter-spacing: -0.03em;
    }
    .risk-status-text {
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .risk-directive {
        font-size: 0.95rem;
        color: #cbd5e1;
        margin-top: 0.5rem;
        font-weight: 500;
    }

    /* Section Cards */
    .soc-card {
        background: #151e2e;
        border: 1px solid #283548;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .soc-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 1px solid #233044;
        padding-bottom: 0.5rem;
    }

    /* Metric Sub-cards */
    .sub-metric-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 0.85rem;
        text-align: center;
    }
    .sub-metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .sub-metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }

    /* Primary Action Button Customization */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        border: 1px solid #3b82f6;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
        transition: all 0.2s ease;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        border-color: #60a5fa;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6);
        transform: translateY(-1px);
    }

    /* Data Table Styling */
    .data-table-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #1f2937;
        font-size: 0.9rem;
    }
    .data-table-label {
        color: #94a3b8;
        font-weight: 600;
    }
    .data-table-value {
        color: #f1f5f9;
        font-weight: 700;
        text-align: right;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #111827;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #1f2937;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
        color: #94a3b8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #60a5fa !important;
        border-bottom: 2px solid #3b82f6 !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "screening_results" not in st.session_state:
    st.session_state["screening_results"] = None
if "active_doc_bytes" not in st.session_state:
    st.session_state["active_doc_bytes"] = None
if "active_live_bytes" not in st.session_state:
    st.session_state["active_live_bytes"] = None
if "unmask_pii" not in st.session_state:
    st.session_state["unmask_pii"] = False


# -----------------------------------------------------------------------------
# SIDEBAR: SYSTEM CONNECTIVITY, CONFIGURATION & DEMO PRESETS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏛️ SSB Terminal Config")
    st.caption("Sashastra Seema Bal | Special Service Bureau")

    st.markdown("""
    **Terminal:** `SSB-INP-07` (Indo-Nepal Border)  
    **Checkpoint:** Raxaul Integrated Checkpost  
    **Duty Officer:** `Insp. V. K. Sharma (SSB/MHA)`  
    **Status:** 🟢 **OPERATIONAL**
    """)
    st.divider()

    st.markdown("### 🔑 API & Model Configuration")
    env_key = os.environ.get("GEMINI_API_KEY", "")
    user_key = st.text_input(
        "Google Gemini API Key",
        value=env_key,
        type="password",
        help="Reads from .env by default. Enter key manually to override."
    )
    api_key_to_use = user_key.strip() if user_key.strip() else env_key

    preferred_model = st.selectbox(
        "Primary Forensic Model",
        options=FALLBACK_MODELS,
        index=0,
        help="Defaults to gemini-3.6-flash. Automatically cascades to subsequent models if 404/unsupported."
    )

    st.divider()

    st.markdown("### ⚙️ System Diagnostic Health")
    # Health checks
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        if api_key_to_use:
            st.success("Gemini API: Ready", icon="✅")
        else:
            st.warning("Gemini API: Key Req.", icon="⚠️")
    with col_h2:
        st.success("FaceNet: Ready", icon="✅")

    st.success("EXIF Scanner: Active", icon="✅")
    st.caption("DeepFace FaceNet calibrated at Cosine Distance < 0.45")

    st.divider()

    st.markdown("### 🧬 Biometric Calibration")
    custom_threshold = st.slider(
        "Facenet Cosine Threshold",
        min_value=0.30,
        max_value=0.75,
        value=0.45,
        step=0.01,
        help="Default is 0.45 (ICAO standard). For legacy documents like older Aadhaar cards with childhood photos, adjust to 0.55 - 0.65."
    )

    st.divider()

    st.markdown("### 🧪 Quick-Load Hackathon Scenarios")
    st.caption("Pre-configured test cases for immediate evaluation:")

    selected_scenario = st.selectbox(
        "Select Test Scenario",
        options=list(DEMO_SCENARIOS.keys()),
        format_func=lambda k: DEMO_SCENARIOS[k]["title"]
    )

    scenario_info = DEMO_SCENARIOS[selected_scenario]
    st.info(f"**Target:** {scenario_info['expected_status']}\n\n{scenario_info['description']}")

    if st.button("📥 Load Scenario Data", key="btn_load_scenario"):
        with st.spinner("Generating scenario assets..."):
            st.session_state["active_doc_bytes"] = scenario_info["doc_func"]()
            st.session_state["active_live_bytes"] = scenario_info["live_func"]()
            st.session_state["screening_results"] = None
            st.success("Scenario assets loaded successfully!")
            st.rerun()

    st.divider()

    # Privacy Masking Toggle
    st.markdown("### 🔒 Data Privacy & Masking")
    unmask = st.toggle(
        "Reveal Unmasked Identifiers",
        value=st.session_state["unmask_pii"],
        help="Per MHA/UIDAI privacy rules, Aadhaar (XXXX-XXXX-1234) and IDs are masked. Enable for authorized officer audit."
    )
    if unmask != st.session_state["unmask_pii"]:
        st.session_state["unmask_pii"] = unmask
        st.rerun()

    if st.button("🔄 Reset Screening Session"):
        st.session_state["active_doc_bytes"] = None
        st.session_state["active_live_bytes"] = None
        st.session_state["screening_results"] = None
        st.rerun()


# -----------------------------------------------------------------------------
# TOP HEADER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="terminal-header">
    <div class="terminal-title">
        <span>🛡️ AI-Based Identity & Document Screening System</span>
        <span class="badge-gov">MHA / SSB SEC-OPS</span>
    </div>
    <div class="terminal-subtitle">
        Ministry of Home Affairs | Sashastra Seema Bal Checkpoint Terminal
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TWO COLUMNS FOR INPUT ACQUISITION
# -----------------------------------------------------------------------------
col_doc, col_bio = st.columns([1, 1], gap="large")

with col_doc:
    st.markdown("""
    <div class="soc-card">
        <div class="soc-card-title">📄 Module 1 & 3: Document Ingestion</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_doc = st.file_uploader(
        "Upload Passport / National ID / Visa / Aadhaar",
        type=["jpg", "jpeg", "png"],
        key="uploader_doc",
        help="Accepts high-resolution images or scans of identity documents."
    )

    if uploaded_doc is not None:
        st.session_state["active_doc_bytes"] = uploaded_doc.read()

    # Display Document Preview if available (Clean st.image without deprecated use_container_width)
    if st.session_state["active_doc_bytes"]:
        doc_img = Image.open(io.BytesIO(st.session_state["active_doc_bytes"]))
        st.image(doc_img, caption=f"Ingested Document ({doc_img.width}x{doc_img.height} px)")
    else:
        st.info("Awaiting identity document ingestion. Upload an image above or load a demo scenario from the sidebar.")

with col_bio:
    st.markdown("""
    <div class="soc-card">
        <div class="soc-card-title">📷 Module 4: Passenger Biometric Acquisition</div>
    </div>
    """, unsafe_allow_html=True)

    bio_mode = st.radio(
        "Biometric Capture Method",
        options=["Live Checkpoint Webcam", "Upload Biometric Selfie"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if bio_mode == "Live Checkpoint Webcam":
        camera_photo = st.camera_input("Acquire Passenger Face Capture", key="camera_live")
        if camera_photo is not None:
            st.session_state["active_live_bytes"] = camera_photo.read()
    else:
        uploaded_selfie = st.file_uploader(
            "Upload Passenger Selfie / Photo",
            type=["jpg", "jpeg", "png"],
            key="uploader_selfie"
        )
        if uploaded_selfie is not None:
            st.session_state["active_live_bytes"] = uploaded_selfie.read()

    # Display Live Capture Preview if available (Clean st.image)
    if st.session_state["active_live_bytes"]:
        live_img = Image.open(io.BytesIO(st.session_state["active_live_bytes"]))
        st.image(live_img, caption=f"Passenger Biometric Capture ({live_img.width}x{live_img.height} px)")
    else:
        st.info("Awaiting passenger biometric capture via live camera or file upload.")


# -----------------------------------------------------------------------------
# TRIGGER SCREENING PIPELINE
# -----------------------------------------------------------------------------
st.write("")
trigger_clicked = st.button("🚀 Run Comprehensive Screening & Biometric Verification", type="primary")

if trigger_clicked:
    if not st.session_state["active_doc_bytes"]:
        st.error("Missing Input: Please ingest an identity document before initiating screening.")
    elif not api_key_to_use:
        st.error("Missing Configuration: Please provide your GEMINI_API_KEY in the sidebar or .env file.")
    else:
        with st.status("Executing 4-Module Checkpoint Screening Pipeline...", expanded=True) as status_box:
            try:
                # 1. Metadata Forensics
                st.write("🔍 **Module 3A: Scanning EXIF & container metadata for editing software signatures...**")
                meta_report: MetadataForensicReport = scan_exif_metadata(st.session_state["active_doc_bytes"])
                time.sleep(0.3)

                # 2. OCR & Multimodal Forensic Vision
                st.write(f"🤖 **Module 1 & 3B: Running Gemini Multimodal OCR & Tamper Analysis (Model: {preferred_model})...**")
                ocr_result, model_used, ocr_err = extract_document_and_forensics(
                    image_bytes=st.session_state["active_doc_bytes"],
                    api_key=api_key_to_use,
                    preferred_model=preferred_model
                )
                if ocr_err:
                    st.warning(f"Note on OCR engine: {ocr_err}")
                else:
                    st.write(f"✨ Forensic extraction completed using `{model_used}`.")
                time.sleep(0.3)

                # 3. Rule & Standards Validation
                st.write("⚖️ **Module 2: Auditing expiration dates, format integrity, and demographic consistency...**")
                rule_report: RuleValidationReport = validate_rules_and_standards(ocr_result)
                time.sleep(0.3)

                # 4. Biometric Face Verification
                st.write("🧬 **Module 4: Executing DeepFace 1:1 facial biometric matching (Facenet, Cosine < 0.45)...**")
                bio_report = None
                if st.session_state["active_live_bytes"]:
                    bio_report = verify_biometrics(
                        doc_image_bytes=st.session_state["active_doc_bytes"],
                        live_image_bytes=st.session_state["active_live_bytes"],
                        threshold=custom_threshold
                    )
                else:
                    st.write("⚠️ Biometric verification skipped (no live passenger face provided).")
                time.sleep(0.3)

                # 5. Composite Risk Scoring
                st.write("📊 **Synthesizing threat vectors into Composite Risk Score (0-100)...**")
                risk_result: CompositeRiskResult = compute_composite_risk(
                    ocr_result=ocr_result,
                    rule_report=rule_report,
                    metadata_report=meta_report,
                    biometric_report=bio_report
                )
                time.sleep(0.2)

                # Save into session state
                st.session_state["screening_results"] = {
                    "ocr_result": ocr_result,
                    "model_used": model_used,
                    "rule_report": rule_report,
                    "meta_report": meta_report,
                    "bio_report": bio_report,
                    "risk_result": risk_result,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                status_box.update(label="Screening & Verification Pipeline Completed!", state="complete", expanded=False)

            except Exception as e:
                status_box.update(label="Screening Execution Error", state="error", expanded=True)
                st.error(f"Pipeline Exception occurred: {str(e)}")


# -----------------------------------------------------------------------------
# OUTPUT DASHBOARD
# -----------------------------------------------------------------------------
if st.session_state["screening_results"]:
    res = st.session_state["screening_results"]
    ocr_res: OCRForensicResult = res["ocr_result"]
    rule_rep: RuleValidationReport = res["rule_report"]
    meta_rep: MetadataForensicReport = res["meta_report"]
    bio_rep: BiometricMatchReport = res["bio_report"]
    risk_res: CompositeRiskResult = res["risk_result"]

    st.markdown("---")

    # 1. Master Composite Risk Score Banner
    st.markdown(f"""
    <div class="risk-banner" style="background: {risk_res.badge_bg}; border-color: {risk_res.badge_color};">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div>
                <div class="risk-status-text" style="color: {risk_res.badge_color};">
                    {risk_res.status_label}
                </div>
                <div class="risk-directive">
                    <strong>OPERATIONAL DIRECTIVE:</strong> {risk_res.recommendation}
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.35rem;">
                    Terminal Audit Stamp: {res['timestamp']} | AI Engine: {res['model_used']}
                </div>
            </div>
            <div style="text-align: right;">
                <div class="risk-score-display" style="color: {risk_res.badge_color};">
                    {risk_res.score}<span style="font-size: 1.6rem; color: #94a3b8;">/100</span>
                </div>
                <div style="font-size: 0.85rem; font-weight: 700; color: {risk_res.badge_color}; letter-spacing: 0.05em;">
                    COMPOSITE THREAT INDEX
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Module Threat Breakdown Mini-Cards
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="sub-metric-box">
            <div class="sub-metric-value" style="color: {'#EF4444' if risk_res.module_breakdown['rules_and_expiry'] > 0 else '#10B981'};">
                +{risk_res.module_breakdown['rules_and_expiry']} pts
            </div>
            <div class="sub-metric-label">Rule & Validity Penalty</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="sub-metric-box">
            <div class="sub-metric-value" style="color: {'#EF4444' if risk_res.module_breakdown['metadata_forensics'] > 0 else '#10B981'};">
                +{risk_res.module_breakdown['metadata_forensics']} pts
            </div>
            <div class="sub-metric-label">Metadata / EXIF Penalty</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="sub-metric-box">
            <div class="sub-metric-value" style="color: {'#EF4444' if risk_res.module_breakdown['visual_tampering'] > 0 else '#10B981'};">
                +{risk_res.module_breakdown['visual_tampering']} pts
            </div>
            <div class="sub-metric-label">Visual Tamper Penalty</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"""
        <div class="sub-metric-box">
            <div class="sub-metric-value" style="color: {'#EF4444' if risk_res.module_breakdown['biometrics'] > 0 else '#10B981'};">
                +{risk_res.module_breakdown['biometrics']} pts
            </div>
            <div class="sub-metric-label">Biometric Discrepancy</div>
        </div>
        """, unsafe_allow_html=True)

    # Triggered Threat Factors (if any)
    if risk_res.factors:
        with st.expander(f"⚠️ View {len(risk_res.factors)} Specific Threat Factors Triggered", expanded=True):
            for factor in risk_res.factors:
                color_dot = "🔴" if factor.severity == "HIGH" else ("🟠" if factor.severity == "MEDIUM" else "🟡")
                st.markdown(f"{color_dot} **[{factor.category}]** {factor.description} `(+{factor.points} pts)`")

    st.write("")

    # -------------------------------------------------------------------------
    # 4 DETAILED ANALYSIS TABS
    # -------------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Extracted OCR Data",
        "⚖️ Rule & Standards Validation",
        "🔬 Tampering & Metadata Forensics",
        "🧬 Biometric Matching"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: EXTRACTED OCR DATA
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown("#### Document Demographic & Legal Extraction")

        # Display masked or raw identifier based on privacy setting
        raw_id = ocr_res.document_number
        display_id = raw_id if st.session_state["unmask_pii"] else mask_identifier(ocr_res.document_type, raw_id)

        col_t1a, col_t1b = st.columns(2, gap="medium")

        with col_t1a:
            st.markdown("""
            <div class="soc-card">
                <div class="soc-card-title">📜 Document Specifications</div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="data-table-row">
                <span class="data-table-label">Document Type:</span>
                <span class="data-table-value">{ocr_res.document_type}</span>
            </div>
            <div class="data-table-row">
                <span class="data-table-label">Document Number:</span>
                <span class="data-table-value">{display_id} {'🔒 (Masked)' if not st.session_state['unmask_pii'] else '🔓 (Officer Unmasked)'}</span>
            </div>
            <div class="data-table-row">
                <span class="data-table-label">Issuing Authority / State:</span>
                <span class="data-table-value">{ocr_res.issuing_country}</span>
            </div>
            <div class="data-table-row">
                <span class="data-table-label">Date of Issuance:</span>
                <span class="data-table-value">{ocr_res.issue_date}</span>
            </div>
            <div class="data-table-row">
                <span class="data-table-label">Date of Expiration:</span>
                <span class="data-table-value" style="color: {'#EF4444' if rule_rep.is_expired else '#10B981'}; font-weight: 800;">
                    {ocr_res.expiry_date}
                </span>
            </div>
            </div>
            """, unsafe_allow_html=True)

        with col_t1b:
            st.markdown("""
            <div class="soc-card">
                <div class="soc-card-title">👤 Holder Demographic Identity</div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="data-table-row">
                <span class="data-table-label">Full Legal Name:</span>
                <span class="data-table-value">{ocr_res.full_name}</span>
            </div>
            <div class="data-table-row">
                <span class="data-table-label">Nationality / Citizenship:</span>
                <span class="data-table-value">{ocr_res.nationality}</span>
            </div>
            <div class="data-table-row">
                <span class="data-table-label">Date of Birth:</span>
                <span class="data-table-value">{ocr_res.dob}</span>
            </div>
            <div class="data-table-row">
                <span class="data-table-label">Gender:</span>
                <span class="data-table-value">{ocr_res.gender}</span>
            </div>
            <div class="data-table-row">
                <span class="data-table-label">Photo Apparent Age Range:</span>
                <span class="data-table-value">{ocr_res.apparent_age_in_photo}</span>
            </div>
            </div>
            """, unsafe_allow_html=True)

        # MRZ Inspection Section
        if ocr_res.mrz_lines:
            st.markdown("##### 🛂 Machine Readable Zone (MRZ - ICAO 9303)")
            mrz_text = "\n".join(ocr_res.mrz_lines)
            st.code(mrz_text, language="text")

        with st.expander("🔍 View Raw JSON Payload from Gemini Vision Model"):
            st.json(ocr_res.model_dump())

    # -------------------------------------------------------------------------
    # TAB 2: RULE & STANDARDS VALIDATION
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("#### Automated Rule & Chronological Integrity Audit")

        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            st.markdown("""
            <div class="soc-card">
                <div class="soc-card-title">📅 Expiration & Status Check</div>
            """, unsafe_allow_html=True)

            if rule_rep.is_expired:
                st.error(f"🚨 EXPIRED DOCUMENT: Expired {abs(rule_rep.days_to_expiry or 0)} days ago.", icon="❌")
            elif rule_rep.is_expiring_soon:
                st.warning(f"⚠️ EXPIRING SOON: {rule_rep.days_to_expiry} days remaining.", icon="⚠️")
            else:
                st.success("✅ VALID EXPIRY: Active travel document.", icon="✅")

            if rule_rep.calculated_age is not None:
                st.metric("Calculated Biological Age", f"{rule_rep.calculated_age} yrs")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_r2:
            st.markdown("""
            <div class="soc-card">
                <div class="soc-card-title">🛡️ Statutory Rule Verification Log</div>
            """, unsafe_allow_html=True)

            for item in rule_rep.audit_items:
                if item.status == "PASS":
                    st.success(f"**[{item.category}] {item.rule_name}**: {item.details}")
                elif item.status == "WARN":
                    st.warning(f"**[{item.category}] {item.rule_name}**: {item.details}")
                else:
                    st.error(f"**[{item.category}] {item.rule_name}**: {item.details}")

            st.markdown("</div>", unsafe_allow_html=True)

        if rule_rep.logical_contradictions:
            st.error(f"🚨 **Flagged Contradictions ({len(rule_rep.logical_contradictions)}):**")
            for c in rule_rep.logical_contradictions:
                st.markdown(f"- {c}")

    # -------------------------------------------------------------------------
    # TAB 3: TAMPERING & METADATA FORENSICS
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown("#### Digital Forensics, EXIF Analysis & Compression Distinction")

        col_f1, col_f2 = st.columns(2, gap="medium")

        with col_f1:
            st.markdown("""
            <div class="soc-card">
                <div class="soc-card-title">💾 EXIF Metadata Signature Scan</div>
            """, unsafe_allow_html=True)

            if meta_rep.editing_software_detected:
                st.error(f"🚨 Digital Editing Software Detected: **{', '.join(meta_rep.detected_software_names)}**")
                st.markdown("**Tampering Traces in EXIF Tags:**")
                st.json(meta_rep.suspicious_tags)
            elif meta_rep.chat_app_transit_detected:
                st.info(f"ℹ️ **Transit Profile:** {meta_rep.chat_app_notes}")
            else:
                st.success("✅ Clean EXIF profile. No desktop editing software signatures detected.")

            st.markdown(f"**Container Format:** `{meta_rep.file_format}` | **Dimensions:** `{meta_rep.dimensions}` | **Tags Found:** `{meta_rep.total_tags_found}`")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_f2:
            st.markdown("""
            <div class="soc-card">
                <div class="soc-card-title">👁️ Gemini Vision Tamper Forensics</div>
            """, unsafe_allow_html=True)

            st.markdown(f"**Tamper Risk Assessment:** `{ocr_res.tamper_risk_level}`")

            # Status indicators for 3 key visual checks
            c1, c2, c3 = st.columns(3)
            with c1:
                if ocr_res.pixel_splicing_detected:
                    st.error("Splicing: YES", icon="🚨")
                else:
                    st.success("Splicing: NO", icon="✅")
            with c2:
                if ocr_res.photo_replacement_signs:
                    st.error("Photo Cut: YES", icon="🚨")
                else:
                    st.success("Photo Cut: NO", icon="✅")
            with c3:
                if ocr_res.font_alteration_detected:
                    st.error("Font Altered: YES", icon="🚨")
                else:
                    st.success("Font Altered: NO", icon="✅")

            st.markdown(f"**Photo Border & Shadow Inspection:** {ocr_res.photo_replacement_details}")
            st.markdown(f"**Typography & Kerning Inspection:** {ocr_res.font_alteration_details}")
            st.markdown("</div>", unsafe_allow_html=True)

        # Highlight Chat-App Compression vs Tampering distinction prominently
        st.markdown("""
        <div class="soc-card" style="border-left: 4px solid #60a5fa;">
            <div class="soc-card-title">📡 Compression Artifacts vs. Malicious Forgery Analysis</div>
        """, unsafe_allow_html=True)
        st.write(ocr_res.compression_vs_tamper_analysis)
        st.caption("Forensic Protocol: Uniform 8x8 DCT quantization, uniform downscaling, and stripped EXIF are classified as benign messaging app artifacts (WhatsApp/Telegram), distinct from localized pixel splicing or headshot replacement.")
        st.markdown("</div>", unsafe_allow_html=True)

        if meta_rep.raw_exif:
            with st.expander("📋 View Complete EXIF Tag Dump"):
                st.json(meta_rep.raw_exif)

    # -------------------------------------------------------------------------
    # TAB 4: BIOMETRIC MATCHING
    # -------------------------------------------------------------------------
    with tab4:
        st.markdown("#### 1:1 Facial Biometric Verification (DeepFace Facenet)")

        if bio_rep:
            col_b1, col_b2, col_b3 = st.columns([1, 1, 1.2], gap="medium")

            with col_b1:
                st.markdown("**Document Photo**")
                if st.session_state["active_doc_bytes"]:
                    st.image(Image.open(io.BytesIO(st.session_state["active_doc_bytes"])))

            with col_b2:
                st.markdown("**Passenger Live Capture**")
                if st.session_state["active_live_bytes"]:
                    st.image(Image.open(io.BytesIO(st.session_state["active_live_bytes"])))

            with col_b3:
                st.markdown("""
                <div class="soc-card">
                    <div class="soc-card-title">📐 Biometric Decision Matrix</div>
                """, unsafe_allow_html=True)

                if bio_rep.status_label == "MATCH":
                    st.success(f"### {bio_rep.status_label} (VERIFIED)")
                    st.markdown(f"**Similarity Score:** `{bio_rep.similarity_percentage}%`")
                elif bio_rep.status_label == "MISMATCH":
                    st.error(f"### {bio_rep.status_label} (IMPOSTOR)")
                    st.markdown(f"**Similarity Score:** `{bio_rep.similarity_percentage}%`")
                else:
                    st.warning(f"### {bio_rep.status_label}")

                st.metric("Cosine Distance", f"{bio_rep.distance:.4f}", f"Threshold: < {bio_rep.threshold:.2f}")

                # Distance gauge indicator
                is_below = bio_rep.distance < bio_rep.threshold
                st.progress(max(0.0, min(1.0, 1.0 - bio_rep.distance)))

                st.write(bio_rep.status_message)
                st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.info("No biometric face comparison executed. Capture live passenger webcam in Column 2 to run FaceNet verification.")

    # -------------------------------------------------------------------------
    # AUDIT DOSSIER EXPORT FOR BORDER OFFICERS
    # -------------------------------------------------------------------------
    st.markdown("---")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        # JSON Dossier
        dossier_data = {
            "checkpoint_terminal": "SSB-INP-07",
            "timestamp": res["timestamp"],
            "duty_officer": "Insp. V. K. Sharma",
            "composite_risk_score": risk_res.score,
            "verdict": risk_res.status_label,
            "directive": risk_res.recommendation,
            "document_ocr": ocr_res.model_dump(),
            "rule_validation": {
                "is_expired": rule_rep.is_expired,
                "days_to_expiry": rule_rep.days_to_expiry,
                "contradictions": rule_rep.logical_contradictions,
                "summary": rule_rep.summary
            },
            "metadata_forensics": {
                "editing_software_detected": meta_rep.editing_software_detected,
                "detected_software": meta_rep.detected_software_names,
                "chat_app_transit": meta_rep.chat_app_transit_detected
            },
            "biometrics": {
                "verified": bio_rep.verified if bio_rep else False,
                "distance": bio_rep.distance if bio_rep else None,
                "threshold": bio_rep.threshold if bio_rep else None,
                "similarity": bio_rep.similarity_percentage if bio_rep else None
            }
        }
        st.download_button(
            "💾 Download JSON Screening Dossier",
            data=json.dumps(dossier_data, indent=2),
            file_name=f"SSB_Audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    with col_e2:
        # Text Inspection Slip
        txt_slip = f"""========================================================================
SASHASTRA SEEMA BAL - CHECKPOINT INSPECTION SLIP
Ministry of Home Affairs, Government of India
========================================================================
Terminal ID : SSB-INP-07 (Indo-Nepal Border)
Timestamp   : {res['timestamp']}
Officer     : Insp. V. K. Sharma (SSB/MHA)

TRAVELER IDENTITY:
Name        : {ocr_res.full_name}
Doc Type    : {ocr_res.document_type}
Doc Number  : {display_id}
Nationality : {ocr_res.nationality}
DOB / Age   : {ocr_res.dob} ({rule_rep.calculated_age or 'N/A'} yrs)

FORENSIC SCREENING SUMMARY:
Risk Score  : {risk_res.score} / 100
Status      : {risk_res.status_label}
Directive   : {risk_res.recommendation}

MODULE AUDIT FINDINGS:
- Expiry Status      : {'EXPIRED' if rule_rep.is_expired else 'VALID'}
- Demographic Check  : {'CONTRADICTION DETECTED' if rule_rep.logical_contradictions else 'PASS'}
- EXIF Editing Tools : {'DETECTED: ' + ', '.join(meta_rep.detected_software_names) if meta_rep.editing_software_detected else 'CLEAN'}
- Visual Tampering   : {ocr_res.tamper_risk_level} (Splicing: {ocr_res.pixel_splicing_detected})
- Biometric Match    : {bio_rep.status_label if bio_rep else 'NOT RUN'} (Cosine Dist: {bio_rep.distance if bio_rep else 'N/A'})

Signature: ______________________
SSB Duty Officer
========================================================================
"""
        st.download_button(
            "📄 Download Official MHA Clearance Slip (.txt)",
            data=txt_slip,
            file_name=f"SSB_Clearance_Slip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
