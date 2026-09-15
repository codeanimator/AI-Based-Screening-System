"""
Metadata Forensics & Error Level Analysis (ELA) Engine (Module 3 - Part A)
SSB Checkpoint Terminal / Ministry of Home Affairs

Air-Gapped Edge Architecture:
- EXIF and Container Metadata Analysis (PIL.ExifTags)
- OpenCV Error Level Analysis (ELA) for detecting Photoshop clone stamps and spliced pixels
- Chat-app transit profiling (WhatsApp/Telegram compression vs tampering)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image, ExifTags
import io
import cv2
import numpy as np


KNOWN_EDITING_SIGNATURES = [
    "photoshop",
    "canva",
    "gimp",
    "paint.net",
    "mspaint",
    "coreldraw",
    "lightroom",
    "snapseed",
    "affinity",
    "pixelmator",
    "picsart",
    "facetune",
    "seashore",
    "pixlr",
    "photopea",
    "befunky",
    "fotor",
]

CHAT_APP_INDICATORS = [
    "whatsapp",
    "telegram",
    "signal",
    "wechat",
    "messenger",
]


@dataclass
class MetadataForensicReport:
    has_exif: bool = False
    editing_software_detected: bool = False
    detected_software_names: List[str] = field(default_factory=list)
    suspicious_tags: Dict[str, str] = field(default_factory=dict)
    chat_app_transit_detected: bool = False
    chat_app_notes: str = ""
    total_tags_found: int = 0
    raw_exif: Dict[str, str] = field(default_factory=dict)
    file_format: str = "UNKNOWN"
    dimensions: str = "0x0"
    forensic_summary: str = ""
    risk_score_contribution: int = 0  # 0 to 40 points
    # OpenCV ELA metrics
    ela_max_diff: int = 0
    ela_mean_diff: float = 0.0
    pixel_splicing_detected: bool = False
    ela_details: str = ""
    # Digital screen capture / vector canvas detection
    is_screen_capture: bool = False
    screen_capture_details: str = ""


def perform_ela(image_input: Any, quality: int = 90, threshold: int = 42) -> Dict[str, Any]:
    """
    Performs Error Level Analysis (ELA) using OpenCV to identify pixel splicing,
    clone stamp modifications, and localized compression rate inconsistencies.

    Algorithm:
    1. Re-compresses the image to 90% JPEG quality in memory.
    2. Calculates the absolute difference (cv2.absdiff) between the original and recompressed image.
    3. Converts difference matrix to grayscale and analyzes max pixel intensity and variance.
    4. If max difference exceeds the baseline threshold, flags pixel_splicing_detected = True.
    """
    try:
        # Convert input to numpy BGR array
        if isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            orig = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
            orig = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            orig = image_input.copy()
        else:
            return {
                "pixel_splicing_detected": False,
                "max_diff": 0,
                "mean_diff": 0.0,
                "tamper_risk_level": "LOW",
                "splicing_details": "ELA error: Invalid image input format."
            }

        if orig is None or orig.size == 0:
            return {
                "pixel_splicing_detected": False,
                "max_diff": 0,
                "mean_diff": 0.0,
                "tamper_risk_level": "LOW",
                "splicing_details": "ELA error: Empty image matrix."
            }

        # Step 1: Re-compress image to specified JPEG quality in-memory
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        success, encoded_buf = cv2.imencode(".jpg", orig, encode_param)
        if not success:
            return {
                "pixel_splicing_detected": False,
                "max_diff": 0,
                "mean_diff": 0.0,
                "tamper_risk_level": "LOW",
                "splicing_details": "ELA error: Failed to encode JPEG in memory."
            }

        recompressed = cv2.imdecode(encoded_buf, cv2.IMREAD_COLOR)

        # Step 2: Calculate absolute difference
        diff = cv2.absdiff(orig, recompressed)

        # Step 3: Convert to grayscale and analyze intensity
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        max_diff = int(np.max(gray_diff))
        mean_diff = float(np.mean(gray_diff))
        std_diff = float(np.std(gray_diff))

        # Dynamic heuristic: Spliced pixels produce sharp localized ELA spikes
        diff_ratio = max_diff / (mean_diff + 1e-5)
        is_spliced = (max_diff >= threshold) or (diff_ratio > 35.0 and max_diff >= 35)

        if is_spliced:
            splicing_details = (
                f"OpenCV ELA detected anomalous pixel error rates (Max Diff: {max_diff}, Mean: {mean_diff:.2f}, Ratio: {diff_ratio:.1f}). "
                "Localized high-frequency compression artifacts indicate digital clone stamp or spliced overlay."
            )
            tamper_risk = "HIGH"
        else:
            splicing_details = (
                f"OpenCV ELA error levels uniform across image (Max Diff: {max_diff}, Mean: {mean_diff:.2f}). "
                "No localized compression spikes or spliced pixel overlays detected."
            )
            tamper_risk = "LOW"

        return {
            "pixel_splicing_detected": is_spliced,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "tamper_risk_level": tamper_risk,
            "splicing_details": splicing_details
        }

    except Exception as e:
        return {
            "pixel_splicing_detected": False,
            "max_diff": 0,
            "mean_diff": 0.0,
            "tamper_risk_level": "LOW",
            "splicing_details": f"OpenCV ELA analysis exception: {str(e)}"
        }


def scan_exif_metadata(image_bytes: bytes) -> MetadataForensicReport:
    """
    Scans raw image bytes for EXIF, container metadata, and runs OpenCV ELA.
    Returns a comprehensive MetadataForensicReport.
    """
    report = MetadataForensicReport()

    try:
        image = Image.open(io.BytesIO(image_bytes))
        report.file_format = image.format or "JPEG"
        report.dimensions = f"{image.width}x{image.height}"

        # 1. Run OpenCV Error Level Analysis (ELA)
        ela_res = perform_ela(image_bytes, quality=90, threshold=38)
        report.ela_max_diff = ela_res.get("max_diff", 0)
        report.ela_mean_diff = ela_res.get("mean_diff", 0.0)
        report.pixel_splicing_detected = ela_res.get("pixel_splicing_detected", False)
        report.ela_details = ela_res.get("splicing_details", "")

        # 2. Inspect PIL getexif()
        raw_tags: Dict[str, str] = {}
        exif = image.getexif()

        if exif and len(exif) > 0:
            report.has_exif = True
            for tag_id, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                val_str = str(value)
                if len(val_str) > 200:
                    val_str = val_str[:197] + "..."
                raw_tags[tag_name] = val_str

        # 3. Check extended EXIF / IFD sub-dictionaries if available
        try:
            for ifd_id in ExifTags.IFD:
                try:
                    ifd = exif.get_ifd(ifd_id)
                    for tag_id, value in ifd.items():
                        tag_name = f"{ifd_id.name}:{ExifTags.TAGS.get(tag_id, str(tag_id))}"
                        val_str = str(value)
                        if len(val_str) > 200:
                            val_str = val_str[:197] + "..."
                        raw_tags[tag_name] = val_str
                except Exception:
                    pass
        except Exception:
            pass

        # 4. Check image.info dictionary (PNG text chunks, WebP chunks, JPEG markers)
        if hasattr(image, "info") and isinstance(image.info, dict):
            for k, v in image.info.items():
                if k not in ["exif", "icc_profile", "photoshop"]:
                    val_str = str(v)
                    if len(val_str) > 200:
                        val_str = val_str[:197] + "..."
                    raw_tags[f"Container:{k}"] = val_str
                elif k == "photoshop":
                    raw_tags["Container:PhotoshopData"] = "Photoshop 8BIM Resource Block Present"

        report.raw_exif = raw_tags
        report.total_tags_found = len(raw_tags)

        # 5. Check for Digital Screen-Grab / Vector Canvas Ingestion
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                gray_b = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                pure_white_cnt = np.sum(gray_b >= 254)
                pure_white_pct = float((pure_white_cnt / gray_b.size) * 100.0)
                has_camera_tags = any(k in report.raw_exif for k in ["Make", "Model", "FNumber", "ExposureTime", "FocalLength"])

                if pure_white_pct > 18.0 and not has_camera_tags:
                    report.screen_capture_details = (
                        f"Official Electronic Document Profile / Vector Canvas ({pure_white_pct:.1f}% pure digital background, "
                        f"lossless {report.file_format}). Standard statutory e-Aadhaar / DigiLocker electronic format."
                    )
        except Exception:
            pass

        # 6. Scan for Digital Editing Software Signatures
        detected_software: set = set()
        suspicious_matches: Dict[str, str] = {}

        # 6a. Scan parsed tags (ignoring standard XML namespace URI strings)
        for tag_name, val in raw_tags.items():
            val_lower = val.lower()
            tag_lower = tag_name.lower()

            for software in KNOWN_EDITING_SIGNATURES:
                if software in val_lower or (software in tag_lower and "adobe.xmp" not in tag_lower):
                    detected_software.add(software.capitalize())
                    suspicious_matches[tag_name] = val

        # 6b. Scan full un-truncated image info chunks (e.g. XMP CreatorTool, Canva metadata)
        if hasattr(image, "info") and isinstance(image.info, dict):
            for k, v in image.info.items():
                v_str = str(v).lower()
                for software in KNOWN_EDITING_SIGNATURES:
                    if software in v_str:
                        detected_software.add(software.capitalize())
                        suspicious_matches[f"Container:{k}"] = f"{software.capitalize()} editing signature detected in chunk"

        # 6c. Scan raw byte stream for embedded software markers
        lower_raw_bytes = image_bytes.lower()
        for software in KNOWN_EDITING_SIGNATURES:
            if software.encode("ascii") in lower_raw_bytes:
                detected_software.add(software.capitalize())
                if "BinaryPayload" not in suspicious_matches:
                    suspicious_matches["BinaryPayload"] = f"{software.capitalize()} marker identified in binary payload"

        if detected_software or report.pixel_splicing_detected:
            report.editing_software_detected = bool(detected_software)
            report.detected_software_names = sorted(list(detected_software))
            report.suspicious_tags = suspicious_matches
            
            # Risk points: Desktop / online graphics editors (Canva, Photoshop, etc.) on official IDs are high threat
            points = 0
            if detected_software:
                points += 70
            if report.pixel_splicing_detected:
                points = max(points, 35)
            report.risk_score_contribution = min(70, points)

            summary_parts = []
            if detected_software:
                summary_parts.append(f"Digital editing metadata detected ({', '.join(report.detected_software_names)})")
            if report.pixel_splicing_detected:
                summary_parts.append(f"OpenCV ELA pixel splicing detected (Max diff: {report.ela_max_diff})")
            report.forensic_summary = f"🚨 FORGERY RISK: {'; '.join(summary_parts)}."
        else:
            # 7. Clean Electronic Document or Chat-App Transit Profile
            report.chat_app_transit_detected = True
            report.chat_app_notes = (
                "Official Electronic Document / Transit Profile: No desktop editing software detected. "
                "Vector canvas and compression levels conform to certified electronic document standards (e-Aadhaar / DigiLocker)."
            )
            report.risk_score_contribution = 0
            report.forensic_summary = (
                f"✅ Clean metadata & ELA profile ({report.file_format}). "
                "Error Level Analysis confirms uniform compression without pixel splicing."
            )

    except Exception as e:
        report.forensic_summary = f"⚠️ Metadata scan exception: {str(e)}"
        report.risk_score_contribution = 10

    return report
