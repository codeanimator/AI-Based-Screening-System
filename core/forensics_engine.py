"""
Metadata Forensics Engine (Module 3 - Part A)
SSB Checkpoint Terminal / Ministry of Home Affairs

Inspects document image EXIF and container metadata using PIL.ExifTags to detect
traces of digital image manipulation software (Photoshop, GIMP, Canva, MS Paint, etc.),
and isolates benign chat-app compression signatures (WhatsApp, Telegram) from true forgeries.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from PIL import Image, ExifTags
import io


KNOWN_EDITING_SIGNATURES = [
    "photoshop",
    "adobe",
    "gimp",
    "canva",
    "paint.net",
    "mspaint",
    "paint",
    "coreldraw",
    "lightroom",
    "snapseed",
    "affinity",
    "pixelmator",
    "picsart",
    "facetune",
    "seashore",
    "pixlr",
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


def scan_exif_metadata(image_bytes: bytes) -> MetadataForensicReport:
    """
    Scans raw image bytes for EXIF and file container metadata.
    Returns a comprehensive MetadataForensicReport.
    """
    report = MetadataForensicReport()

    try:
        image = Image.open(io.BytesIO(image_bytes))
        report.file_format = image.format or "JPEG"
        report.dimensions = f"{image.width}x{image.height}"

        # 1. Inspect PIL getexif()
        raw_tags: Dict[str, str] = {}
        exif = image.getexif()

        if exif and len(exif) > 0:
            report.has_exif = True
            for tag_id, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                # Truncate overly long binary or bytecode strings for safety
                val_str = str(value)
                if len(val_str) > 200:
                    val_str = val_str[:197] + "..."
                raw_tags[tag_name] = val_str

        # 2. Check for extended EXIF / IFD sub-dictionaries if available
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

        # 3. Check image.info dictionary (PNG text chunks, WebP chunks, JPEG markers)
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

        # 4. Scan for Digital Editing Software Signatures
        detected_software: set = set()
        suspicious_matches: Dict[str, str] = {}

        for tag_name, val in raw_tags.items():
            val_lower = val.lower()
            tag_lower = tag_name.lower()

            for software in KNOWN_EDITING_SIGNATURES:
                if software in val_lower or software in tag_lower:
                    detected_software.add(software.capitalize())
                    suspicious_matches[tag_name] = val

        if detected_software:
            report.editing_software_detected = True
            report.detected_software_names = sorted(list(detected_software))
            report.suspicious_tags = suspicious_matches
            report.risk_score_contribution = 40
            report.forensic_summary = (
                f"🚨 FORGERY RISK: Digital editing metadata detected ({', '.join(report.detected_software_names)}). "
                f"Document has likely been modified using desktop image editing software."
            )
        else:
            # 5. Check if EXIF is completely stripped (Common chat-app compression signature)
            if not report.has_exif:
                report.chat_app_transit_detected = True
                report.chat_app_notes = (
                    "No EXIF metadata detected. This is typical for images transferred via WhatsApp, "
                    "Telegram, or web screenshot tools, where metadata is stripped for privacy and bandwidth. "
                    "This is an everyday transport artifact, not affirmative proof of tampering."
                )
                report.risk_score_contribution = 5  # Small informational baseline
                report.forensic_summary = (
                    "ℹ️ Metadata stripped (Transit / Chat-app profile detected). "
                    "No editing software signatures found in EXIF."
                )
            else:
                report.risk_score_contribution = 0
                report.forensic_summary = (
                    f"✅ Clean metadata profile. {report.total_tags_found} standard camera/scanner tags found. "
                    "No digital editing signatures identified."
                )

    except Exception as e:
        report.forensic_summary = f"⚠️ Metadata scan exception: {str(e)}"
        report.risk_score_contribution = 10

    return report
