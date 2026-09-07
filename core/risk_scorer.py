"""
Composite Risk Scoring Engine
SSB Checkpoint Terminal / Ministry of Home Affairs

Calculates a normalized 0-100 Composite Threat Score synthesizing:
1. Rule & Standards Validation (Expiry, Chronology, Age vs Photo Contradiction)
2. Metadata Forensics (EXIF Editing Software Signatures)
3. Gemini Multimodal Visual Tamper Inspection (Splicing, Headshot Replacement, Font Alteration)
4. Biometric Face Verification (DeepFace Facenet Cosine Distance >= 0.45)

Status Thresholds:
- Score < 30: "STATUS: PASS / CLEARED" (Green)
- Score 30 - 65: "STATUS: MANUAL REVIEW" (Amber)
- Score > 65: "STATUS: REJECT / FORGERY ALERT" (Red)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .forensics_engine import MetadataForensicReport
from .ocr_engine import OCRForensicResult
from .rules_engine import RuleValidationReport
from .biometrics_engine import BiometricMatchReport


@dataclass
class ThreatFactor:
    category: str
    description: str
    points: int
    severity: str  # "HIGH", "MEDIUM", "LOW"


@dataclass
class CompositeRiskResult:
    score: int  # 0 to 100
    status_code: str  # "PASS", "REVIEW", "REJECT"
    status_label: str  # Full formatted badge text
    badge_color: str  # Hex color code (#10B981, #F59E0B, #EF4444)
    badge_bg: str
    threat_level: str  # "LOW", "ELEVATED", "CRITICAL"
    recommendation: str  # Actionable directive for SSB Officer
    factors: List[ThreatFactor] = field(default_factory=list)
    module_breakdown: Dict[str, int] = field(default_factory=dict)


def compute_composite_risk(
    ocr_result: OCRForensicResult,
    rule_report: RuleValidationReport,
    metadata_report: MetadataForensicReport,
    biometric_report: Optional[BiometricMatchReport] = None
) -> CompositeRiskResult:
    """
    Computes explainable composite risk score (0-100) and actionable clearance verdict.
    """
    factors: List[ThreatFactor] = []
    breakdown = {
        "rules_and_expiry": 0,
        "metadata_forensics": 0,
        "visual_tampering": 0,
        "biometrics": 0,
    }

    # 1. Rule & Standards Validation Penalties
    if rule_report.is_expired:
        points = 35
        factors.append(ThreatFactor(
            category="Rule Validation",
            description=f"Document is expired ({rule_report.days_to_expiry or 0} days past expiry).",
            points=points,
            severity="HIGH"
        ))
        breakdown["rules_and_expiry"] += points

    for contradiction in rule_report.logical_contradictions:
        points = 35
        factors.append(ThreatFactor(
            category="Rule Validation",
            description=f"Logical contradiction: {contradiction}",
            points=points,
            severity="HIGH"
        ))
        breakdown["rules_and_expiry"] += points

    if rule_report.is_expiring_soon:
        points = 10
        factors.append(ThreatFactor(
            category="Rule Validation",
            description="Document expiring within 6 months (border travel warning).",
            points=points,
            severity="LOW"
        ))
        breakdown["rules_and_expiry"] += points

    # 2. Metadata Forensics Penalties
    if metadata_report.editing_software_detected:
        points = 40
        software_list = ", ".join(metadata_report.detected_software_names)
        factors.append(ThreatFactor(
            category="Metadata Forensics",
            description=f"Digital editing software traces discovered in EXIF ({software_list}).",
            points=points,
            severity="HIGH"
        ))
        breakdown["metadata_forensics"] += points

    # 3. Gemini Multimodal Visual Tamper Penalties
    if ocr_result.pixel_splicing_detected:
        points = 35
        factors.append(ThreatFactor(
            category="Visual Forensics",
            description=f"Pixel splicing / clone stamp artifacts: {ocr_result.splicing_details}",
            points=points,
            severity="HIGH"
        ))
        breakdown["visual_tampering"] += points

    if ocr_result.photo_replacement_signs:
        points = 40
        factors.append(ThreatFactor(
            category="Visual Forensics",
            description=f"Photo cut-and-paste / replacement markers: {ocr_result.photo_replacement_details}",
            points=points,
            severity="HIGH"
        ))
        breakdown["visual_tampering"] += points

    if ocr_result.font_alteration_detected:
        points = 30
        factors.append(ThreatFactor(
            category="Visual Forensics",
            description=f"Font alteration / typography inconsistency: {ocr_result.font_alteration_details}",
            points=points,
            severity="HIGH"
        ))
        breakdown["visual_tampering"] += points

    # Check overall tamper risk level stated by Gemini
    if ocr_result.tamper_risk_level.upper() == "HIGH" and not (
        ocr_result.pixel_splicing_detected or ocr_result.photo_replacement_signs or ocr_result.font_alteration_detected
    ):
        points = 30
        factors.append(ThreatFactor(
            category="Visual Forensics",
            description=f"Gemini Forensic Vision flagged HIGH tamper probability: {ocr_result.forensic_verdict_summary}",
            points=points,
            severity="HIGH"
        ))
        breakdown["visual_tampering"] += points
    elif ocr_result.tamper_risk_level.upper() == "MEDIUM" and breakdown["visual_tampering"] == 0:
        points = 15
        factors.append(ThreatFactor(
            category="Visual Forensics",
            description=f"Gemini Forensic Vision flagged MEDIUM anomalies: {ocr_result.forensic_verdict_summary}",
            points=points,
            severity="MEDIUM"
        ))
        breakdown["visual_tampering"] += points

    # 4. Biometrics Penalties
    if biometric_report:
        if biometric_report.status_label == "MISMATCH":
            points = 45
            factors.append(ThreatFactor(
                category="Biometrics",
                description=(
                    f"Biometric facial mismatch (Cosine distance: {biometric_report.distance:.4f} >= {biometric_report.threshold:.2f}). "
                    f"Passenger face does not match document."
                ),
                points=points,
                severity="HIGH"
            ))
            breakdown["biometrics"] += points
        elif biometric_report.status_label == "ERROR":
            points = 20
            factors.append(ThreatFactor(
                category="Biometrics",
                description=f"Biometric verification incomplete: {biometric_report.status_message}",
                points=points,
                severity="MEDIUM"
            ))
            breakdown["biometrics"] += points
        elif biometric_report.status_label == "SKIPPED":
            points = 10
            factors.append(ThreatFactor(
                category="Biometrics",
                description="Biometric capture not provided. Passenger identity not biometrically validated.",
                points=points,
                severity="LOW"
            ))
            breakdown["biometrics"] += points

    # Compute Aggregate Score (0 - 100)
    raw_score = sum(breakdown.values())
    final_score = max(0, min(100, raw_score))

    # Determine status according to requirements:
    # Score < 30: "STATUS: PASS / CLEARED" (Green)
    # Score 30-65: "STATUS: MANUAL REVIEW" (Amber)
    # Score > 65: "STATUS: REJECT / FORGERY ALERT" (Red)
    if final_score < 30:
        status_code = "PASS"
        status_label = "STATUS: PASS / CLEARED"
        badge_color = "#10B981"  # Emerald green
        badge_bg = "rgba(16, 185, 129, 0.15)"
        threat_level = "LOW THREAT"
        recommendation = "AUTHORIZE CLEARANCE: Document integrity, demographics, and biometrics confirmed."
    elif final_score <= 65:
        status_code = "REVIEW"
        status_label = "STATUS: MANUAL REVIEW"
        badge_color = "#F59E0B"  # Amber
        badge_bg = "rgba(245, 158, 11, 0.15)"
        threat_level = "ELEVATED RISK"
        recommendation = "HOLD FOR SECONDARY INSPECTION: Physical document verification and manual biometric interview required."
    else:
        status_code = "REJECT"
        status_label = "STATUS: REJECT / FORGERY ALERT"
        badge_color = "#EF4444"  # Red
        badge_bg = "rgba(239, 68, 68, 0.18)"
        threat_level = "CRITICAL THREAT"
        recommendation = "IMMEDIATE REJECTION: Potential forged identity detected. Detain subject and seize document for forensic custody."

    return CompositeRiskResult(
        score=final_score,
        status_code=status_code,
        status_label=status_label,
        badge_color=badge_color,
        badge_bg=badge_bg,
        threat_level=threat_level,
        recommendation=recommendation,
        factors=factors,
        module_breakdown=breakdown
    )
