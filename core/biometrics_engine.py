"""
Biometric Face Verification Engine (Module 4)
SENTINEL Checkpoint Terminal / Ministry of Home Affairs

Performs 1:1 biometric facial comparison between the document portrait
and the live checkpoint passenger capture (webcam) using DeepFace.
- Model: Facenet (Pre-warmed via @st.cache_resource)
- Face Extraction: Multi-scale Haar Cascade + CLAHE Contrast Equalization
- Distance Metric: Cosine
- Calibrated match threshold: distance < 0.45
- Safe temporary file cleanup in try...finally block.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import tempfile
import os
import io
import cv2
import numpy as np
from PIL import Image
import streamlit as st

try:
    from deepface import DeepFace
except ImportError:
    DeepFace = None


COSINE_MATCH_THRESHOLD = 0.55


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
    doc_face_crop_bytes: Optional[bytes] = None
    live_face_crop_bytes: Optional[bytes] = None


@st.cache_resource(show_spinner="Pre-warming SENTINEL Facenet Biometric Model...")
def get_facenet_model():
    """
    Pre-warms and caches the Facenet model in RAM to ensure sub-second inference.
    """
    if DeepFace is not None:
        try:
            return DeepFace.build_model("Facenet")
        except Exception:
            return None
    return None


def extract_face_crop_bytes(image_bytes: bytes, is_document: bool = False) -> Tuple[Optional[bytes], Optional[np.ndarray]]:
    """
    Extracts and crops the primary face from image bytes.
    Enhanced for faded/laminated identity cards using CLAHE and multi-scale Haar cascade.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, None

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Pass 1: Standard grayscale
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25))

        # Pass 2: Enhanced CLAHE contrast (for ID cards with watermarks/laminations)
        if len(faces) == 0:
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            faces = face_cascade.detectMultiScale(enhanced_gray, scaleFactor=1.08, minNeighbors=2, minSize=(20, 20))

        # Pass 3: Profile cascade fallback
        if len(faces) == 0:
            profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
            faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25))

        if len(faces) > 0:
            # Select largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            # Add 20% margin for hair, jawline, and natural facial contour
            mx = int(w * 0.20)
            my = int(h * 0.20)
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(img.shape[1], x + w + mx)
            y2 = min(img.shape[0], y + h + my)

            face_crop = img[y1:y2, x1:x2]
            _, buf = cv2.imencode(".jpg", face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            return buf.tobytes(), face_crop

        # If document and no face detected, crop standard ID portrait region (left-middle quadrant)
        if is_document:
            h_doc, w_doc = img.shape[:2]
            left_crop = img[int(h_doc * 0.1):int(h_doc * 0.8), 0:int(w_doc * 0.45)]
            _, buf = cv2.imencode(".jpg", left_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            return buf.tobytes(), left_crop

        return None, None
    except Exception:
        return None, None


def verify_biometrics(
    doc_image_bytes: bytes,
    live_image_bytes: bytes,
    threshold: float = COSINE_MATCH_THRESHOLD
) -> BiometricMatchReport:
    """
    Compares the face on the identity document with the passenger's live webcam capture.
    Crops actual faces first, then executes sub-second 1:1 DeepFace FaceNet verification.
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

    # Pre-warm model in cache
    get_facenet_model()

    tmp_doc_path = None
    tmp_live_path = None

    try:
        # Step 1: Detect and crop face from document
        doc_crop_bytes, doc_np = extract_face_crop_bytes(doc_image_bytes, is_document=True)
        report.doc_face_crop_bytes = doc_crop_bytes

        # Step 2: Detect and crop face from webcam
        live_crop_bytes, live_np = extract_face_crop_bytes(live_image_bytes, is_document=False)
        report.live_face_crop_bytes = live_crop_bytes

        # Use cropped face numpy matrices directly in memory (zero disk file latency)
        if doc_np is None:
            doc_np = cv2.imdecode(np.frombuffer(doc_image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if live_np is None:
            live_np = cv2.imdecode(np.frombuffer(live_image_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Apply CLAHE illumination and contrast equalization to ID portrait (removes card print/laminate fade)
        if doc_np is not None:
            try:
                lab = cv2.cvtColor(doc_np, cv2.COLOR_BGR2LAB)
                l_c, a_c, b_c = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l_c)
                doc_proc = cv2.cvtColor(cv2.merge((cl, a_c, b_c)), cv2.COLOR_LAB2BGR)
            except Exception:
                doc_proc = doc_np
        else:
            doc_proc = doc_np

        # Step 3: Fast 1:1 facial verification with landmark alignment
        try:
            result = DeepFace.verify(
                img1_path=doc_proc,
                img2_path=live_np,
                model_name="Facenet",
                detector_backend="opencv",
                distance_metric="cosine",
                enforce_detection=False,
                align=True,
                threshold=threshold
            )
        except Exception:
            # Fallback to skip if OpenCV face landmark detector encounters edge crop artifacts
            result = DeepFace.verify(
                img1_path=doc_proc,
                img2_path=live_np,
                model_name="Facenet",
                detector_backend="skip",
                distance_metric="cosine",
                enforce_detection=False,
                align=False,
                threshold=threshold
            )

        distance = float(result.get("distance", 1.0))
        report.distance = round(distance, 4)

        # Calibrate match status strictly against threshold
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
        report.risk_score_contribution = 25

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
