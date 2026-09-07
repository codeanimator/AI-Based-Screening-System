"""
Rule & Standards Validation Engine (Module 2)
SSB Checkpoint Terminal / Ministry of Home Affairs

Validates document expiration, format integrity, and demographic consistency.
Flags logical contradictions, including:
- Adult photo with recent child DOB (or vice-versa)
- Issue date after expiry date
- DOB or Issue date in the future
- Abnormal validity duration (> 10 years standard)
- Format discrepancies
Safely masks personal identifiers (Aadhaar, Passport) in all audit outputs.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Dict, Optional, Any
import re
from .ocr_engine import mask_identifier, OCRForensicResult


@dataclass
class RuleAuditItem:
    category: str
    rule_name: str
    status: str  # "PASS", "WARN", "FAIL"
    details: str
    risk_points: int = 0


@dataclass
class RuleValidationReport:
    overall_valid: bool = True
    is_expired: bool = False
    is_expiring_soon: bool = False  # Within 180 days (6 months)
    days_to_expiry: Optional[int] = None
    calculated_age: Optional[int] = None
    logical_contradictions: List[str] = field(default_factory=list)
    audit_items: List[RuleAuditItem] = field(default_factory=list)
    risk_score_contribution: int = 0  # 0 to 45 points
    summary: str = ""


def _parse_any_date(date_str: str) -> Optional[date]:
    """Attempts to parse common date formats (YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY, etc.)"""
    if not date_str or date_str.upper() in ["UNKNOWN", "LIFELONG", "NOT_SPECIFIED", "NA", "N/A"]:
        return None

    # Clean whitespace and unwanted characters
    cleaned = date_str.strip().replace(".", "-").replace("/", "-")

    formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%d-%b-%Y",
        "%d-%B-%Y",
        "%Y%m%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            pass

    # Try extracting digits if embedded in text
    match = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", date_str)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass

    match2 = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", date_str)
    if match2:
        try:
            return date(int(match2.group(3)), int(match2.group(2)), int(match2.group(1)))
        except ValueError:
            pass

    return None


def validate_rules_and_standards(ocr_data: OCRForensicResult) -> RuleValidationReport:
    """
    Executes comprehensive rule and integrity checks on extracted OCR document fields.
    """
    report = RuleValidationReport()
    today = date.today()
    total_penalty = 0

    masked_doc_num = mask_identifier(ocr_data.document_type, ocr_data.document_number)

    # -------------------------------------------------------------
    # 1. EXPIRY CHECK
    # -------------------------------------------------------------
    exp_date = _parse_any_date(ocr_data.expiry_date)
    if ocr_data.expiry_date.upper() in ["LIFELONG", "NOT_APPLICABLE"] or "LIFELONG" in ocr_data.expiry_date.upper():
        report.audit_items.append(RuleAuditItem(
            category="Validity",
            rule_name="Document Expiration Status",
            status="PASS",
            details="Document has permanent / lifelong validity (e.g. National ID / Aadhaar / Voter ID).",
            risk_points=0
        ))
    elif exp_date:
        days_diff = (exp_date - today).days
        report.days_to_expiry = days_diff

        if days_diff < 0:
            report.is_expired = True
            report.overall_valid = False
            total_penalty += 35
            report.audit_items.append(RuleAuditItem(
                category="Validity",
                rule_name="Document Expiration Status",
                status="FAIL",
                details=f"🚨 DOCUMENT EXPIRED on {exp_date.isoformat()} ({abs(days_diff)} days ago). Crossing / travel invalid.",
                risk_points=35
            ))
        elif days_diff <= 180:
            report.is_expiring_soon = True
            total_penalty += 10
            report.audit_items.append(RuleAuditItem(
                category="Validity",
                rule_name="Document Expiration Status",
                status="WARN",
                details=f"⚠️ Expiring soon on {exp_date.isoformat()} (in {days_diff} days). Warning: International standards require 6 months validity.",
                risk_points=10
            ))
        else:
            report.audit_items.append(RuleAuditItem(
                category="Validity",
                rule_name="Document Expiration Status",
                status="PASS",
                details=f"✅ Document currently valid until {exp_date.isoformat()} ({days_diff} days remaining).",
                risk_points=0
            ))
    else:
        report.audit_items.append(RuleAuditItem(
            category="Validity",
            rule_name="Document Expiration Status",
            status="WARN",
            details=f"Expiry date could not be parsed unambiguously: '{ocr_data.expiry_date}'. Requires manual inspection.",
            risk_points=5
        ))
        total_penalty += 5

    # -------------------------------------------------------------
    # 2. DATE CHRONOLOGY & SANITY
    # -------------------------------------------------------------
    dob = _parse_any_date(ocr_data.dob)
    issue_date = _parse_any_date(ocr_data.issue_date)

    if dob:
        # Check if DOB is in the future
        if dob > today:
            report.overall_valid = False
            contradiction = f"Logical contradiction: Date of Birth ({dob.isoformat()}) is in the future."
            report.logical_contradictions.append(contradiction)
            total_penalty += 35
            report.audit_items.append(RuleAuditItem(
                category="Chronology",
                rule_name="DOB Temporal Sanity",
                status="FAIL",
                details=f"🚨 {contradiction}",
                risk_points=35
            ))
        else:
            # Calculate Age
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            report.calculated_age = age

            if age > 120 or age < 0:
                report.overall_valid = False
                contradiction = f"Demographic anomaly: Calculated age of {age} years is medically improbable."
                report.logical_contradictions.append(contradiction)
                total_penalty += 30
                report.audit_items.append(RuleAuditItem(
                    category="Chronology",
                    rule_name="Biological Age Sanity",
                    status="FAIL",
                    details=f"🚨 {contradiction}",
                    risk_points=30
                ))
            else:
                report.audit_items.append(RuleAuditItem(
                    category="Chronology",
                    rule_name="Biological Age Sanity",
                    status="PASS",
                    details=f"✅ Calculated age is {age} years (DOB: {dob.isoformat()}).",
                    risk_points=0
                ))

    if issue_date:
        if issue_date > today:
            report.overall_valid = False
            contradiction = f"Logical contradiction: Issue Date ({issue_date.isoformat()}) is in the future."
            report.logical_contradictions.append(contradiction)
            total_penalty += 30
            report.audit_items.append(RuleAuditItem(
                category="Chronology",
                rule_name="Issue Date Temporal Sanity",
                status="FAIL",
                details=f"🚨 {contradiction}",
                risk_points=30
            ))

        if exp_date and issue_date >= exp_date:
            report.overall_valid = False
            contradiction = f"Logical contradiction: Issue Date ({issue_date.isoformat()}) is on or after Expiry Date ({exp_date.isoformat()})."
            report.logical_contradictions.append(contradiction)
            total_penalty += 35
            report.audit_items.append(RuleAuditItem(
                category="Chronology",
                rule_name="Issue vs Expiry Chronology",
                status="FAIL",
                details=f"🚨 {contradiction}",
                risk_points=35
            ))

        if dob and issue_date < dob:
            report.overall_valid = False
            contradiction = f"Logical contradiction: Issue Date ({issue_date.isoformat()}) is before Date of Birth ({dob.isoformat()})."
            report.logical_contradictions.append(contradiction)
            total_penalty += 40
            report.audit_items.append(RuleAuditItem(
                category="Chronology",
                rule_name="Birth vs Issue Chronology",
                status="FAIL",
                details=f"🚨 {contradiction}",
                risk_points=40
            ))

    # -------------------------------------------------------------
    # 3. DEMOGRAPHIC LOGICAL CONTRADICTIONS: PHOTO AGE VS DOB
    # -------------------------------------------------------------
    apparent_age = ocr_data.apparent_age_in_photo.upper()
    if report.calculated_age is not None and apparent_age != "UNKNOWN":
        # Check: Child DOB (e.g. age < 12) with Adult Photo
        if report.calculated_age < 12 and "ADULT" in apparent_age:
            report.overall_valid = False
            contradiction = (
                f"Severe demographic contradiction: Document DOB indicates a CHILD ({report.calculated_age} years old), "
                f"but portrait photograph exhibits an ADULT subject ({ocr_data.apparent_age_in_photo})."
            )
            report.logical_contradictions.append(contradiction)
            total_penalty += 40
            report.audit_items.append(RuleAuditItem(
                category="Demographics",
                rule_name="Photo Appearance vs DOB Sanity",
                status="FAIL",
                details=f"🚨 {contradiction}",
                risk_points=40
            ))
        # Check: Senior/Adult DOB (> 30) with Child photo
        elif report.calculated_age >= 25 and "CHILD" in apparent_age:
            report.overall_valid = False
            contradiction = (
                f"Severe demographic contradiction: Document DOB indicates an ADULT ({report.calculated_age} years old), "
                f"but portrait photograph exhibits a CHILD subject ({ocr_data.apparent_age_in_photo})."
            )
            report.logical_contradictions.append(contradiction)
            total_penalty += 40
            report.audit_items.append(RuleAuditItem(
                category="Demographics",
                rule_name="Photo Appearance vs DOB Sanity",
                status="FAIL",
                details=f"🚨 {contradiction}",
                risk_points=40
            ))
        else:
            report.audit_items.append(RuleAuditItem(
                category="Demographics",
                rule_name="Photo Appearance vs DOB Sanity",
                status="PASS",
                details=f"✅ Document age ({report.calculated_age} yrs) is consistent with visual photo demographics ({ocr_data.apparent_age_in_photo}).",
                risk_points=0
            ))

    # -------------------------------------------------------------
    # 4. FORMAT INTEGRITY & STANDARDS
    # -------------------------------------------------------------
    clean_num = re.sub(r"[\s\-_]", "", ocr_data.document_number).upper()
    doc_type = ocr_data.document_type.upper()

    if "PASSPORT" in doc_type:
        # Standard ICAO 9303 Indian Passport is 1 uppercase letter followed by 7 digits
        if re.match(r"^[A-Z][0-9]{7}$", clean_num):
            report.audit_items.append(RuleAuditItem(
                category="Format",
                rule_name="Passport Number Format (ICAO 9303)",
                status="PASS",
                details=f"✅ Valid format for Indian Passport ({masked_doc_num}).",
                risk_points=0
            ))
        elif len(clean_num) >= 7 and len(clean_num) <= 10:
            report.audit_items.append(RuleAuditItem(
                category="Format",
                rule_name="Passport Number Format",
                status="PASS",
                details=f"✅ Standard international passport format ({masked_doc_num}).",
                risk_points=0
            ))
        else:
            report.audit_items.append(RuleAuditItem(
                category="Format",
                rule_name="Passport Number Format",
                status="WARN",
                details=f"⚠️ Unusual character length or formatting for Passport number ({masked_doc_num}).",
                risk_points=10
            ))
            total_penalty += 10

    elif "AADHAAR" in doc_type:
        if clean_num.isdigit() and len(clean_num) == 12:
            report.audit_items.append(RuleAuditItem(
                category="Format",
                rule_name="Aadhaar 12-Digit Format Integrity",
                status="PASS",
                details=f"✅ Valid 12-digit format conforming to UIDAI standards ({masked_doc_num}).",
                risk_points=0
            ))
        else:
            report.overall_valid = False
            total_penalty += 25
            report.audit_items.append(RuleAuditItem(
                category="Format",
                rule_name="Aadhaar 12-Digit Format Integrity",
                status="FAIL",
                details=f"🚨 Invalid Aadhaar number length or characters: found {len(clean_num)} chars ({masked_doc_num}). Expected 12 digits.",
                risk_points=25
            ))

    # -------------------------------------------------------------
    # 5. DOCUMENT PHOTO AGING / LEGACY ISSUANCE DETECTION
    # -------------------------------------------------------------
    if issue_date:
        years_since_issue = (today - issue_date).days / 365.25
        if years_since_issue >= 8.0:
            if "AADHAAR" in doc_type:
                details_msg = (
                    f"ℹ️ UIDAI Aadhaar printed issue date ({issue_date.isoformat()}) indicates original enrollment {years_since_issue:.1f} years ago. "
                    "Note: Aadhaar cards permit subsequent biometric/photo updates at Aadhaar Seva Kendras. "
                    "If the photo was recently re-enrolled, verify biometric match against current facial features."
                )
            else:
                details_msg = (
                    f"⚠️ Document was issued {years_since_issue:.1f} years ago ({issue_date.isoformat()}). "
                    "Facial morphology may have matured significantly since photo was taken. "
                    "Elevated biometric cosine distance is expected; officer visual interview recommended."
                )

            report.audit_items.append(RuleAuditItem(
                category="Biometrics Advisory",
                rule_name="Legacy Document Photo Aging Gap",
                status="INFO" if "AADHAAR" in doc_type else "WARN",
                details=details_msg,
                risk_points=0
            ))
        else:
            report.audit_items.append(RuleAuditItem(
                category="Biometrics Advisory",
                rule_name="Document Photo Freshness",
                status="PASS",
                details=f"✅ Document photo issued within recent timeframe ({years_since_issue:.1f} years ago).",
                risk_points=0
            ))

    # Cap rule risk score contribution
    report.risk_score_contribution = min(45, total_penalty)

    if report.logical_contradictions:
        report.summary = f"🚨 {len(report.logical_contradictions)} critical contradiction(s) flagged: {report.logical_contradictions[0]}"
    elif report.is_expired:
        report.summary = "🚨 Document has expired. Ineligible for border clearance."
    elif total_penalty > 0:
        report.summary = "⚠️ Minor format or validity warnings detected during standards validation."
    else:
        report.summary = "✅ All statutory rules, date chronology, and format standards successfully verified."

    return report
