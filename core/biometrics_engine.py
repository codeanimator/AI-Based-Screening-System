"""
Biometric Face Verification Engine (Module 4)
SENTINEL Checkpoint Terminal / Ministry of Home Affairs

Performs 1:1 biometric facial comparison between the document portrait
and the live checkpoint passenger capture (webcam) using InsightFace.
- Model: buffalo_l (ResNet-100 backbone, ArcFace loss, 512-d embeddings)
- Runtime: ONNX Runtime CPU (ctx_id=-1) — zero TensorFlow dependency
- Distance Metric: Cosine (scipy.spatial.distance.cosine)
- Calibrated match threshold: distance < 0.55
- Face extraction: InsightFace RetinaFace detector (built-in)
- Fallback: OpenCV Haar cascade if InsightFace detector finds nothing
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import os

# Configure env BEFORE any native library is imported to avoid MKL/OMP conflicts
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Force ONNX Runtime to CPU only — prevents GPU allocation attempt that can OOM
os.environ["ORT_DISABLE_TENSORRT"] = "1"

import io
import cv2
import numpy as np
from PIL import Image
import streamlit as st

try:
    from scipy.spatial.distance import cosine as scipy_cosine
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


COSINE_MATCH_THRESHOLD = 0.55


@dataclass
class BiometricMatchReport:
    verified: bool = False
    distance: float = 1.0
    threshold: float = COSINE_MATCH_THRESHOLD
    similarity_percentage: float = 0.0
    model_name: str = "InsightFace-buffalo_l"
    metric: str = "cosine"
    status_label: str = "PENDING"  # "MATCH", "MISMATCH", "ERROR", "SKIPPED"
    status_message: str = ""
    risk_score_contribution: int = 0  # 0 to 50 points
    error: Optional[str] = None
    doc_face_crop_bytes: Optional[bytes] = None
    live_face_crop_bytes: Optional[bytes] = None


# ---------------------------------------------------------------------------
# MODEL INITIALISATION — cached globally so it only loads once per container
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Pre-warming SENTINEL InsightFace Biometric Model...")
def get_insightface_model():
    """
    Initialises and caches the InsightFace FaceAnalysis app in RAM.
    Uses the lightweight buffalo_l model (ONNX Runtime CPU backend).
    Downloads model files automatically to ~/.insightface/models/ on first run.
    Returns None if InsightFace is unavailable (graceful degradation).
    """
    try:
        import insightface
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        # ctx_id=-1 → CPU; det_size must be multiple of 32, 640x640 is optimal
        app.prepare(ctx_id=-1, det_size=(640, 640))
        return app
    except Exception as e:
        return None


# Keep old name as alias so any stale import in app.py won't crash
def get_facenet_model():
    """Backward-compat alias → delegates to InsightFace initialiser."""
    return get_insightface_model()


# ---------------------------------------------------------------------------
# FACE CROP HELPER — used for preview thumbnails in the UI
# ---------------------------------------------------------------------------
def extract_face_crop_bytes(
    image_bytes: bytes, is_document: bool = False
) -> Tuple[Optional[bytes], Optional[np.ndarray]]:
    """
    Extracts and crops the primary face from image bytes for UI preview.
    Tries InsightFace detector first; falls back to OpenCV Haar cascade.
    Enhanced for faded/laminated identity cards via CLAHE contrast equalisation.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, None

        # --- Pass 1: InsightFace RetinaFace detector (most accurate) ---
        face_app = get_insightface_model()
        if face_app is not None:
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = face_app.get(img_rgb)
                if faces:
                    # Pick the largest detected face
                    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    x1, y1, x2, y2 = [int(c) for c in face.bbox]
                    # Add 20% margin
                    mx = int((x2 - x1) * 0.20)
                    my = int((y2 - y1) * 0.20)
                    x1 = max(0, x1 - mx)
                    y1 = max(0, y1 - my)
                    x2 = min(img.shape[1], x2 + mx)
                    y2 = min(img.shape[0], y2 + my)
                    face_crop = img[y1:y2, x1:x2]
                    _, buf = cv2.imencode(".jpg", face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    return buf.tobytes(), face_crop
            except Exception:
                pass  # fall through to OpenCV cascade

        # --- Pass 2: OpenCV Haar cascade (fallback) ---
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25))

        if len(faces) == 0:
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            faces = face_cascade.detectMultiScale(
                enhanced_gray, scaleFactor=1.08, minNeighbors=2, minSize=(20, 20)
            )

        if len(faces) == 0:
            profile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_profileface.xml"
            )
            faces = profile_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25)
            )

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            mx, my = int(w * 0.20), int(h * 0.20)
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(img.shape[1], x + w + mx)
            y2 = min(img.shape[0], y + h + my)
            face_crop = img[y1:y2, x1:x2]
            _, buf = cv2.imencode(".jpg", face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            return buf.tobytes(), face_crop

        # --- Pass 3: Document fallback — standard ID portrait region ---
        if is_document:
            h_doc, w_doc = img.shape[:2]
            left_crop = img[int(h_doc * 0.1) : int(h_doc * 0.8), 0 : int(w_doc * 0.45)]
            _, buf = cv2.imencode(".jpg", left_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            return buf.tobytes(), left_crop

        return None, None
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# EMBEDDING HELPER
# ---------------------------------------------------------------------------
def _get_embedding(face_app, img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Extracts a 512-d L2-normalised ArcFace embedding from a BGR image.
    Returns None if no face is detected.
    """
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        faces = face_app.get(img_rgb)
        if not faces:
            return None
        # Use the largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = face.normed_embedding  # already L2-normalised, shape (512,)
        return emb.astype(np.float32)
    except Exception:
        return None


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine distance between two L2-normalised vectors.
    For unit vectors: distance = 1 − dot(a, b).
    Falls back to scipy if available for numerical stability.
    """
    if _HAS_SCIPY:
        return float(scipy_cosine(a, b))
    # Fast numpy path (valid because InsightFace returns normed_embedding)
    return float(1.0 - np.dot(a, b))


# ---------------------------------------------------------------------------
# PRIMARY PUBLIC INTERFACE — return signature is IDENTICAL to old DeepFace version
# ---------------------------------------------------------------------------
def verify_biometrics(
    doc_image_bytes: bytes,
    live_image_bytes: bytes,
    threshold: float = COSINE_MATCH_THRESHOLD,
) -> BiometricMatchReport:
    """
    Compares the face on the identity document with the passenger's live
    webcam capture using InsightFace buffalo_l (ArcFace, ONNX Runtime CPU).

    Args:
        doc_image_bytes : Raw bytes of the identity document image.
        live_image_bytes: Raw bytes of the live passenger webcam capture.
        threshold       : Cosine distance threshold below which faces are
                          considered a match (default 0.55).

    Returns:
        BiometricMatchReport — identical schema to the previous DeepFace version.
    """
    report = BiometricMatchReport(threshold=threshold)

    if not doc_image_bytes or not live_image_bytes:
        report.status_label = "SKIPPED"
        report.status_message = (
            "Both document image and live biometric capture are required for verification."
        )
        report.risk_score_contribution = 15
        return report

    face_app = get_insightface_model()
    if face_app is None:
        report.status_label = "ERROR"
        report.status_message = "InsightFace model failed to initialise. Check deployment logs."
        report.error = "InsightFace app is None after initialisation."
        report.risk_score_contribution = 20
        return report

    try:
        # --- Decode images ---
        doc_np = cv2.imdecode(np.frombuffer(doc_image_bytes, np.uint8), cv2.IMREAD_COLOR)
        live_np = cv2.imdecode(np.frombuffer(live_image_bytes, np.uint8), cv2.IMREAD_COLOR)

        if doc_np is None or live_np is None:
            report.status_label = "ERROR"
            report.status_message = "Failed to decode one or both images."
            report.error = "cv2.imdecode returned None."
            report.risk_score_contribution = 20
            return report

        # --- CLAHE contrast normalisation on ID portrait (removes laminate/print fade) ---
        try:
            lab = cv2.cvtColor(doc_np, cv2.COLOR_BGR2LAB)
            l_c, a_c, b_c = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            doc_np = cv2.cvtColor(
                cv2.merge((clahe.apply(l_c), a_c, b_c)), cv2.COLOR_LAB2BGR
            )
        except Exception:
            pass  # Continue with original if CLAHE fails

        # --- Extract face crops for UI preview thumbnails ---
        doc_crop_bytes, _ = extract_face_crop_bytes(doc_image_bytes, is_document=True)
        live_crop_bytes, _ = extract_face_crop_bytes(live_image_bytes, is_document=False)
        report.doc_face_crop_bytes = doc_crop_bytes
        report.live_face_crop_bytes = live_crop_bytes

        # --- Extract 512-d ArcFace embeddings ---
        doc_emb = _get_embedding(face_app, doc_np)
        live_emb = _get_embedding(face_app, live_np)

        if doc_emb is None:
            report.status_label = "ERROR"
            report.status_message = (
                "⚠️ No face detected in document image. "
                "Ensure the document photo region is clear and well-lit."
            )
            report.error = "InsightFace: no face detected in document."
            report.risk_score_contribution = 25
            return report

        if live_emb is None:
            report.status_label = "ERROR"
            report.status_message = (
                "⚠️ No face detected in live capture. "
                "Ensure the passenger is facing the camera clearly."
            )
            report.error = "InsightFace: no face detected in live image."
            report.risk_score_contribution = 25
            return report

        # --- Compute cosine distance ---
        distance = _cosine_distance(doc_emb, live_emb)
        # Clamp to [0, 2] — valid range for cosine distance
        distance = max(0.0, min(2.0, distance))
        report.distance = round(distance, 4)

        is_match = distance < threshold
        report.verified = is_match

        # Similarity %: map [0, 1] distance range to [100, 0]%
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
                f"🚨 Biometric Mismatch! Cosine distance: {report.distance:.4f} exceeds "
                f"threshold {threshold:.2f} "
                f"(Similarity: {report.similarity_percentage:.1f}%). "
                "Subject does NOT match document portrait."
            )
            report.risk_score_contribution = 45

    except Exception as e:
        report.status_label = "ERROR"
        report.error = str(e)
        report.status_message = f"⚠️ Biometric processing exception: {str(e)}"
        report.risk_score_contribution = 25

    return report
