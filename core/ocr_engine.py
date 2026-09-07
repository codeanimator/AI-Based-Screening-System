"""
OCR & Multimodal Forensic Tamper Inspection Engine (Module 1 & Module 3 - Part B)
SSB Checkpoint Terminal / Ministry of Home Affairs

Uses Google GenAI SDK (`from google import genai; from google.genai import types`)
with a prioritized model fallback chain:
1. gemini-3.6-flash
2. gemini-2.5-flash
3. gemini-2.0-flash
4. gemini-1.5-flash

Includes:
- Strict JSON structured output extraction.
- Demographic & MRZ extraction.
- Forensics prompt explicitly differentiating chat-app compression artifacts from malicious tampering.
- Safe identifier masking (e.g. Aadhaar XXXX-XXXX-1234, Passport masking).
"""

import json
import re
import io
import os
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from PIL import Image

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


FALLBACK_MODELS = [
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]


class OCRForensicResult(BaseModel):
    # Module 1: Demographic & Document Data
    document_type: str = Field(
        default="UNKNOWN",
        description="Type of document (e.g. PASSPORT, AADHAAR, DRIVING_LICENSE, VISA, NATIONAL_ID, VOTER_ID, UNKNOWN)"
    )
    issuing_country: str = Field(
        default="IND",
        description="Three-letter ISO code or country name of issuing state (e.g. IND, USA, GBR, NPL)"
    )
    full_name: str = Field(
        default="UNREADABLE",
        description="Full legal name of the document holder exactly as printed"
    )
    document_number: str = Field(
        default="UNKNOWN",
        description="Document or identity number (Passport Number, 12-digit Aadhaar, License No, etc.)"
    )
    nationality: str = Field(
        default="UNKNOWN",
        description="Stated nationality or citizenship of the holder"
    )
    dob: str = Field(
        default="UNKNOWN",
        description="Date of Birth in YYYY-MM-DD or DD/MM/YYYY format"
    )
    expiry_date: str = Field(
        default="UNKNOWN",
        description="Document expiry date in YYYY-MM-DD, DD/MM/YYYY, or 'LIFELONG'/'NOT_APPLICABLE'"
    )
    issue_date: str = Field(
        default="UNKNOWN",
        description="Date of issuance in YYYY-MM-DD or DD/MM/YYYY format"
    )
    gender: str = Field(
        default="UNKNOWN",
        description="Gender stated on document (M, F, OTHER, UNKNOWN)"
    )
    mrz_lines: List[str] = Field(
        default_factory=list,
        description="Machine Readable Zone lines if a passport or visa, else empty list"
    )
    apparent_age_in_photo: str = Field(
        default="UNKNOWN",
        description="Estimated visual age range of the person in the document photo (e.g. 'ADULT_25_35', 'CHILD_4_8', 'SENIOR_60_PLUS')"
    )
    face_detected_in_document: bool = Field(
        default=True,
        description="Whether a human face/portrait is clearly visible on the document"
    )

    # Module 3: Multimodal Forensic Inspection
    pixel_splicing_detected: bool = Field(
        default=False,
        description="True if pixel cloning, stamp tool, or localized splicing is detected"
    )
    splicing_details: str = Field(
        default="No localized pixel splicing detected.",
        description="Details of spliced zones or reasoning"
    )
    photo_replacement_signs: bool = Field(
        default=False,
        description="True if photo replacement, altered border cuts, mismatched halo, or paper paste-over is observed"
    )
    photo_replacement_details: str = Field(
        default="Portrait photograph appears original and naturally integrated into security substrate.",
        description="Specific findings regarding photo border integration, shadow continuity, and grain consistency"
    )
    font_alteration_detected: bool = Field(
        default=False,
        description="True if numbers or text display mismatched fonts, differing DPI, irregular baseline jumps, or altered kerning"
    )
    font_alteration_details: str = Field(
        default="Typography, font weights, and ink alignments appear consistent across all printed fields.",
        description="Analysis of font uniformity, ink bleed, and character alignment"
    )
    compression_vs_tamper_analysis: str = Field(
        default="Image displays uniform compression without localized digital tampering.",
        description="Detailed forensic distinction explaining whether visual artifacts are normal chat-app downscaling (WhatsApp/Telegram) or deliberate document manipulation"
    )
    tamper_risk_level: str = Field(
        default="LOW",
        description="Tamper risk level: 'LOW', 'MEDIUM', or 'HIGH'"
    )
    tamper_evidence_points: List[str] = Field(
        default_factory=list,
        description="Specific itemized forensic observations and evidence bullets"
    )
    forensic_verdict_summary: str = Field(
        default="Document appears visually authentic under forensic examination.",
        description="Concise summary for immigration/border checkpoint officers"
    )


def mask_identifier(doc_type: str, raw_number: str) -> str:
    """
    Masks personal identifiers to protect privacy per official government guidelines.
    - Aadhaar (12 digits): XXXX-XXXX-1234
    - Passports (Alpha + 7-8 digits): First char and last 3 visible (e.g., A****567)
    - Driving Licenses / PAN: Middle characters masked
    - General: Masks all but last 4 characters
    """
    if not raw_number or raw_number.upper() in ["UNKNOWN", "UNREADABLE", ""]:
        return "UNKNOWN"

    clean = re.sub(r"[\s\-_]", "", raw_number)

    # Aadhaar check (12 continuous digits)
    if clean.isdigit() and len(clean) == 12:
        return f"XXXX-XXXX-{clean[-4:]}"

    # Indian Passport style (1 letter + 7 digits)
    if re.match(r"^[A-Z][0-9]{7,8}$", clean.upper()):
        upper = clean.upper()
        return f"{upper[0]}{'*' * (len(upper) - 4)}{upper[-3:]}"

    # General masking for other identities
    if len(clean) > 5:
        keep_front = 1 if len(clean) < 8 else 2
        keep_back = min(4, len(clean) - keep_front - 2)
        mask_len = len(clean) - keep_front - keep_back
        return f"{clean[:keep_front]}{'*' * mask_len}{clean[-keep_back:]}"

    return clean


SYSTEM_INSTRUCTION = """
You are an elite forensic document examiner and biometrics screening specialist for the Sashastra Seema Bal (SSB), 
Ministry of Home Affairs (MHA), Government of India.

Your task is to analyze the provided identity document (Passport, Aadhaar Card, Driving License, National ID, or Visa) 
for border checkpoint screening.

You must perform two concurrent forensic tasks:
TASK 1: ACCURATE OCR EXTRACTION
- Extract all demographic data (Full Name, Document Number, Nationality, DOB, Expiry Date, Issue Date, Gender).
- Identify Document Type and Issuing Country (e.g. IND for India, NPL for Nepal, etc.).
- Extract MRZ lines if present on a passport or travel visa.
- Assess the visual apparent age of the subject in the photo (e.g. 'ADULT_25_35', 'CHILD_4_8', 'SENIOR_60_PLUS') for logical validation.

TASK 2: RIGOROUS FORENSIC TAMPER INSPECTION
- Check for pixel splicing, clone stamping, or patched overlays.
- Inspect the portrait photograph for signs of cut-and-paste replacement:
  * Irregular cut-out borders, white halos, or sharp edges cutting through the background security pattern.
  * Inconsistent lighting angle or shadows between face and document background.
  * Differing camera noise/grain pattern on the portrait compared to the document body.
- Inspect text fields (especially Document Number, Name, DOB, Expiry Date) for font alterations:
  * Inconsistent typography, mismatched font family or weight.
  * Differing digital resolution/blurriness between specific text and adjacent printed text.
  * Baseline misalignment, unnatural character spacing (kerning) or digital insertion.

CRITICAL FORENSIC MANDATE:
1. MESSAGING COMPRESSION VS FORGERY:
You MUST explicitly distinguish everyday chat-app compression artifacts (such as WhatsApp, Telegram, or Signal downscaling, 
uniform 8x8 DCT JPEG macroblocking, global noise quantization, or stripped EXIF) from actual malicious tampering.
* Chat-app compression affects the ENTIRE image uniformly.
* Malicious tampering is LOCALIZED (e.g., razor-sharp pasted text on a noisy compressed ID, or a sharp face pasted over a blurred card).
Do NOT classify uniform WhatsApp/Telegram compression as tampering. Highlight this distinction in your compression_vs_tamper_analysis.

2. SYNTHETIC & TEST SAMPLES:
If evaluating a digital mock-up or test card, inspect whether there is localized fraudulent alteration (such as pasted numbers, 
spliced photo borders, or mismatched field typography). If the card layout is consistent and has no fraudulent alteration, 
classify tamper_risk_level as 'LOW'.

Output your findings strictly conforming to the requested JSON schema.
"""


def extract_document_and_forensics(
    image_bytes: bytes,
    api_key: Optional[str] = None,
    preferred_model: str = "gemini-3.6-flash"
) -> Tuple[OCRForensicResult, str, Optional[str]]:
    """
    Executes Module 1 (OCR) and Module 3 (Forensic Tamper Inspection) using Google GenAI SDK.
    Follows a prioritized model fallback chain:
      gemini-3.6-flash -> gemini-2.5-flash -> gemini-2.0-flash -> gemini-1.5-flash

    Returns:
      (OCRForensicResult, model_used, error_message)
    """
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return (
            OCRForensicResult(
                full_name="GEMINI_API_KEY NOT CONFIGURED",
                document_number="MISSING_KEY",
                forensic_verdict_summary="API Key missing. Please provide a valid GEMINI_API_KEY in .env or the sidebar."
            ),
            "NONE",
            "GEMINI_API_KEY is not configured. Please supply an API key in the sidebar or in .env."
        )

    if genai is None or types is None:
        return (
            OCRForensicResult(
                full_name="GOOGLE-GENAI NOT INSTALLED",
                forensic_verdict_summary="google-genai library could not be imported."
            ),
            "NONE",
            "The google-genai library is missing."
        )

    client = genai.Client(api_key=key)

    # Build fallback list with preferred_model first
    candidate_models = [preferred_model]
    for m in FALLBACK_MODELS:
        if m not in candidate_models:
            candidate_models.append(m)

    # Prepare image part
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        fmt = pil_img.format or "JPEG"
        mime = f"image/{fmt.lower()}"
        if mime == "image/jpg":
            mime = "image/jpeg"
    except Exception:
        mime = "image/jpeg"

    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime)

    prompt_content = (
        "Perform full OCR extraction and rigorous forensic tamper inspection on this government identity document. "
        "Distinguish chat-app compression from localized digital tampering. Output strictly as JSON."
    )

    last_error = None

    for model_name in candidate_models:
        try:
            # Configure structured output
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=OCRForensicResult,
                temperature=0.1,
            )

            response = client.models.generate_content(
                model=model_name,
                contents=[image_part, prompt_content],
                config=config,
            )

            if response and response.text:
                raw_text = response.text.strip()
                # Parse JSON
                try:
                    data = json.loads(raw_text)
                    result = OCRForensicResult.model_validate(data)
                    return result, model_name, None
                except Exception:
                    # Attempt regex json block extraction if surrounded by backticks
                    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                    if match:
                        data = json.loads(match.group(0))
                        result = OCRForensicResult.model_validate(data)
                        return result, model_name, None

        except Exception as e:
            err_str = str(e)
            last_error = f"{model_name}: {err_str}"
            # Check if it's a 404, not found, or unsupported model error; continue to next fallback
            continue

    # If all models in the fallback chain failed:
    return (
        OCRForensicResult(
            full_name="SCAN FAILED",
            document_number="ERROR",
            tamper_risk_level="HIGH",
            forensic_verdict_summary=f"Automated forensic scan failed across all candidate models: {last_error}"
        ),
        "FAILED",
        f"All models in fallback chain failed: {last_error}"
    )
