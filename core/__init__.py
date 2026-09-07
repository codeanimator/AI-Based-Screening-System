"""
Core screening modules for SSB AI-Based Identity & Document Screening System.
Ministry of Home Affairs / Sashastra Seema Bal Checkpoint Terminal.
"""

from .forensics_engine import scan_exif_metadata, MetadataForensicReport
from .ocr_engine import extract_document_and_forensics, mask_identifier, OCRForensicResult
from .rules_engine import validate_rules_and_standards, RuleValidationReport
from .biometrics_engine import verify_biometrics, BiometricMatchReport
from .risk_scorer import compute_composite_risk, CompositeRiskResult

__all__ = [
    "scan_exif_metadata",
    "MetadataForensicReport",
    "extract_document_and_forensics",
    "mask_identifier",
    "OCRForensicResult",
    "validate_rules_and_standards",
    "RuleValidationReport",
    "verify_biometrics",
    "BiometricMatchReport",
    "compute_composite_risk",
    "CompositeRiskResult",
]
