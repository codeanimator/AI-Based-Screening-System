"""
Biometric Face Verification Engine (Module 4)
SSB Checkpoint Terminal / Ministry of Home Affairs

Performs 1:1 biometric facial comparison between the document portrait
and the live checkpoint passenger capture (webcam) using DeepFace.
- Model: Facenet
- Metric: Cosine
- enforce_detection: False
- Calibrated match threshold: distance < 0.45
- Safe temporary file cleanup in try...finally block.
"""

from dataclasses import dataclass
from typing import Optional
import tempfile
import os
import io
from PIL import Image

try:
    from deepface import DeepFace
except ImportError:
    DeepFace = None


COSINE_MATCH_THRESHOLD = 0.45


@dataclass
class BiometricMatchReport:
    verified: bool = False
    distance: float = 1.0
    threshold: float = COSINE_MATCH_THRESHOLD
    similarity_percentage: float = 0.0
    model_name: str = "Facenet"
    metric: str = "cosine"
    status_label: str = "PENDING"  # "MATCH", "MISMATCH", "ERROR", "SKIPPED"
    status_message: str = ""
    risk_score_contribution: int = 0  # 0 to 50 points
    error: Optional[str] = None


def verify_biometrics(
    doc_image_bytes: bytes,
    live_image_bytes: bytes,
    threshold: float = COSINE_MATCH_THRESHOLD
) -> BiometricMatchReport:
    """
    Compares the face on the identity document with the passenger's live webcam capture.
    Safely writes temporary files and cleans them up unconditionally in a try...finally block.
    """
    report = BiometricMatchReport(threshold=threshold)

    if not doc_image_bytes or not live_image_bytes:
        report.status_label = "SKIPPED"
        report.status_message = "Both document image and live biometric capture are required for verification."
        report.risk_score_contribution = 15
        return report

    if DeepFace is None:
        report.status_label = "ERROR"
        report.status_message = "DeepFace library is not available in the current environment."
        report.error = "Module deepface not imported."
        report.risk_score_contribution = 20
        return report

    tmp_doc_path = None
    tmp_live_path = None

    try:
        # Create safe temporary files
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f_doc:
            tmp_doc_path = f_doc.name
            doc_img = Image.open(io.BytesIO(doc_image_bytes)).convert("RGB")
            doc_img.save(f_doc, format="JPEG", quality=95)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f_live:
            tmp_live_path = f_live.name
            live_img = Image.open(io.BytesIO(live_image_bytes)).convert("RGB")
            live_img.save(f_live, format="JPEG", quality=95)

        # Execute DeepFace 1:1 facial verification
        result = DeepFace.verify(
            img1_path=tmp_doc_path,
            img2_path=tmp_live_path,
            model_name="Facenet",
            detector_backend="opencv",
            distance_metric="cosine",
            enforce_detection=False,
            align=True,
            threshold=threshold
        )

        distance = float(result.get("distance", 1.0))
        report.distance = round(distance, 4)

        # Calibrate match status strictly against the 0.45 cosine threshold
        is_match = distance < threshold
        report.verified = is_match

        # Compute human-readable similarity percentage
        similarity = max(0.0, min(100.0, (1.0 - distance) * 100.0))
        report.similarity_percentage = round(similarity, 1)

        if is_match:
            report.status_label = "MATCH"
            report.status_message = (
                f"✅ Biometric Match Confirmed. Cosine distance: {report.distance:.4f} "
                f"(Threshold: < {threshold:.2f}, Similarity: {report.similarity_percentage:.1f}%)."
            )
            report.risk_score_contribution = 0
        else:
            report.status_label = "MISMATCH"
            report.status_message = (
                f"🚨 Biometric Mismatch! Cosine distance: {report.distance:.4f} exceeds threshold {threshold:.2f} "
                f"(Similarity: {report.similarity_percentage:.1f}%). Subject does NOT match document portrait."
            )
            report.risk_score_contribution = 45

    except Exception as e:
        report.status_label = "ERROR"
        report.error = str(e)
        report.status_message = f"⚠️ Biometric processing exception: {str(e)}"
        report.risk_score_contribution = 25  # Require manual review if biometrics fail

    finally:
        # Guarantee removal of temporary files
        if tmp_doc_path and os.path.exists(tmp_doc_path):
            try:
                os.remove(tmp_doc_path)
            except Exception:
                pass

        if tmp_live_path and os.path.exists(tmp_live_path):
            try:
                os.remove(tmp_live_path)
            except Exception:
                pass

    return report
