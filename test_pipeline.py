"""
Automated unit & pipeline verification script for SSB Document Screening System.
Tests each module independently.
"""

import sys
from PIL import Image
import io

from core.forensics_engine import scan_exif_metadata
from core.ocr_engine import mask_identifier, OCRForensicResult
from core.rules_engine import validate_rules_and_standards
from core.biometrics_engine import verify_biometrics
from core.risk_scorer import compute_composite_risk
from sample_data.demo_assets import DEMO_SCENARIOS


def test_masking():
    print("\n--- TEST 1: SAFE IDENTIFIER MASKING ---")
    aadhaar_masked = mask_identifier("AADHAAR", "543298761234")
    assert aadhaar_masked == "XXXX-XXXX-1234", f"Unexpected Aadhaar mask: {aadhaar_masked}"
    print(f"Aadhaar 543298761234 -> {aadhaar_masked} (PASS)")

    passport_masked = mask_identifier("PASSPORT", "Z1234567")
    assert passport_masked.startswith("Z") and passport_masked.endswith("567"), f"Unexpected Passport mask: {passport_masked}"
    print(f"Passport Z1234567 -> {passport_masked} (PASS)")


def test_metadata_forensics():
    print("\n--- TEST 2: METADATA FORENSICS & CHAT-APP TRANSIT ---")
    # 1. Clean image
    clean_bytes = DEMO_SCENARIOS["1_AUTHENTIC_PASS"]["doc_func"]()
    rep1 = scan_exif_metadata(clean_bytes)
    print(f"Authentic Doc EXIF -> has_exif={rep1.has_exif}, editing_detected={rep1.editing_software_detected} (PASS)")

    # 2. Tampered image with Photoshop tag
    tampered_bytes = DEMO_SCENARIOS["2_TAMPERED_FORGERY"]["doc_func"]()
    rep2 = scan_exif_metadata(tampered_bytes)
    assert rep2.editing_software_detected is True, "Failed to detect Photoshop tag"
    assert "Photoshop" in rep2.detected_software_names, f"Photoshop not in detected names: {rep2.detected_software_names}"
    print(f"Tampered Doc EXIF -> editing_detected={rep2.editing_software_detected}, software={rep2.detected_software_names} (PASS)")

    # 3. Stripped chat-app image
    img = Image.new("RGB", (100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    rep3 = scan_exif_metadata(buf.getvalue())
    assert rep3.chat_app_transit_detected is True
    print(f"Stripped Chat-App Doc -> chat_app_transit_detected={rep3.chat_app_transit_detected} (PASS)")


def test_rules_engine():
    print("\n--- TEST 3: RULE & STANDARDS VALIDATION ---")
    # 1. Authentic passport
    auth_ocr = OCRForensicResult(
        document_type="PASSPORT",
        document_number="Z1839204",
        nationality="INDIAN",
        dob="1990-08-12",
        expiry_date="2032-05-18",
        issue_date="2022-05-19",
        apparent_age_in_photo="ADULT_25_35"
    )
    rule_rep1 = validate_rules_and_standards(auth_ocr)
    assert rule_rep1.overall_valid is True, f"Rule validation failed unexpectedly: {rule_rep1.summary}"
    assert rule_rep1.is_expired is False
    print(f"Authentic Doc Rules -> valid={rule_rep1.overall_valid}, age={rule_rep1.calculated_age} (PASS)")

    # 2. Demographic contradiction: Child DOB with Adult Photo
    contradiction_ocr = OCRForensicResult(
        document_type="PASSPORT",
        document_number="Z5521940",
        nationality="INDIAN",
        dob="2021-06-15",  # 3-5 years old
        expiry_date="2031-06-14",
        issue_date="2021-06-15",
        apparent_age_in_photo="ADULT_30_40"
    )
    rule_rep2 = validate_rules_and_standards(contradiction_ocr)
    assert rule_rep2.overall_valid is False
    assert len(rule_rep2.logical_contradictions) > 0
    print(f"Contradiction Doc Rules -> valid={rule_rep2.overall_valid}, contradiction={rule_rep2.logical_contradictions[0]} (PASS)")

    # 3. Expired document
    expired_ocr = OCRForensicResult(
        document_type="PASSPORT",
        document_number="Z7182903",
        nationality="INDIAN",
        dob="1980-01-01",
        expiry_date="2021-01-01",  # Expired
        issue_date="2011-01-02",
        apparent_age_in_photo="ADULT_40_50"
    )
    rule_rep3 = validate_rules_and_standards(expired_ocr)
    assert rule_rep3.is_expired is True
    assert rule_rep3.overall_valid is False
    print(f"Expired Doc Rules -> is_expired={rule_rep3.is_expired} (PASS)")


def test_composite_risk():
    print("\n--- TEST 4: COMPOSITE RISK SCORER ---")
    auth_ocr = OCRForensicResult(
        document_type="PASSPORT",
        document_number="Z1839204",
        nationality="INDIAN",
        dob="1990-08-12",
        expiry_date="2032-05-18",
        apparent_age_in_photo="ADULT_25_35",
        tamper_risk_level="LOW"
    )
    rule_rep = validate_rules_and_standards(auth_ocr)
    meta_rep = scan_exif_metadata(DEMO_SCENARIOS["1_AUTHENTIC_PASS"]["doc_func"]())

    # Risk for clean authentic case
    risk_clean = compute_composite_risk(auth_ocr, rule_rep, meta_rep, None)
    assert risk_clean.score < 30, f"Clean case got high score: {risk_clean.score}"
    assert risk_clean.status_code == "PASS"
    print(f"Clean Case Score -> {risk_clean.score}/100 [{risk_clean.status_label}] (PASS)")

    # Risk for tampered case
    tampered_ocr = OCRForensicResult(
        document_type="PASSPORT",
        document_number="Z9999999",
        nationality="INDIAN",
        dob="1988-03-22",
        expiry_date="2030-01-10",
        pixel_splicing_detected=True,
        photo_replacement_signs=True,
        font_alteration_detected=True,
        tamper_risk_level="HIGH"
    )
    tampered_meta = scan_exif_metadata(DEMO_SCENARIOS["2_TAMPERED_FORGERY"]["doc_func"]())
    risk_tampered = compute_composite_risk(tampered_ocr, rule_rep, tampered_meta, None)
    assert risk_tampered.score > 65, f"Tampered case got low score: {risk_tampered.score}"
    assert risk_tampered.status_code == "REJECT"
    print(f"Tampered Case Score -> {risk_tampered.score}/100 [{risk_tampered.status_label}] (PASS)")


if __name__ == "__main__":
    test_masking()
    test_metadata_forensics()
    test_rules_engine()
    test_composite_risk()
    print("\n[SUCCESS] ALL PIPELINE TESTS PASSED!")
