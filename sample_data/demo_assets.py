"""
Demo Assets and Test Scenario Generator
SSB Checkpoint Terminal / Ministry of Home Affairs

Provides ready-to-test document and biometric test scenarios for Smart India Hackathon evaluation:
1. Authentic Indian Passport (Valid, Clean EXIF, Matching Live Biometric)
2. Tampered Identity Document (Photoshop EXIF, Spliced Portrait, Font Alteration)
3. Demographic Contradiction (Child DOB 2020 with Adult Photograph)
4. WhatsApp Transferred Document (Compressed, Stripped EXIF, Benign Transit)
"""

import io
from PIL import Image, ImageDraw, ImageFont, ExifTags
import piexif
from typing import Tuple, Dict, Any


def _create_synthetic_passport_image(
    name: str = "RAJESH KUMAR SHARMA",
    doc_num: str = "Z3492815",
    nationality: str = "INDIAN",
    dob: str = "1992-05-14",
    expiry: str = "2031-10-20",
    issue: str = "2021-10-21",
    tampered: bool = False,
    software_tag: str = "",
    face_color: Tuple[int, int, int] = (200, 160, 130)
) -> bytes:
    """Generates a realistic synthetic passport document with security Guilloché-like background."""
    w, h = 650, 420
    img = Image.new("RGB", (w, h), color=(240, 243, 246))
    draw = ImageDraw.Draw(img)

    # Draw security guilloché background patterns
    for y in range(0, h, 8):
        draw.line([(0, y), (w, y + 4)], fill=(225, 232, 240), width=1)
    for x in range(0, w, 16):
        draw.line([(x, 0), (x + 10, h)], fill=(230, 236, 242), width=1)

    # Header Security Band
    draw.rectangle([(0, 0), (w, 55)], fill=(18, 38, 58))
    draw.text((25, 12), "REPUBLIC OF INDIA / BHARAT GANARAJYA", fill=(255, 255, 255))
    draw.text((25, 32), "PASSPORT  *  TYPE: P  *  CODE: IND", fill=(200, 215, 235))

    # Emblem & Watermark Placeholder
    draw.ellipse([(w - 90, 8), (w - 30, 48)], outline=(255, 215, 0), width=2)
    draw.text((w - 78, 20), "SSB", fill=(255, 215, 0))

    # Passport Portrait Box
    photo_box = (35, 80, 175, 260)
    draw.rectangle(photo_box, fill=face_color, outline=(100, 120, 140), width=2)

    # Draw stylized facial silhouette
    draw.ellipse([(75, 110), (135, 175)], fill=(230, 185, 150), outline=(90, 60, 40), width=2)
    draw.arc([(85, 140), (125, 165)], start=0, end=180, fill=(180, 50, 50), width=2)
    # Hair
    draw.chord([(75, 105), (135, 140)], start=180, end=360, fill=(35, 25, 20))
    # Shoulders
    draw.chord([(50, 180), (160, 255)], start=180, end=360, fill=(30, 60, 100))

    if tampered:
        # Draw visible jagged splicing artifact around portrait
        draw.rectangle([(32, 77), (178, 263)], outline=(255, 0, 0), width=2)
        draw.text((40, 265), "[SPLICED BORDER]", fill=(200, 0, 0))

    # Text fields on right
    tx = 210
    fields = [
        ("SURNAME", name.split()[-1] if " " in name else name),
        ("GIVEN NAMES", " ".join(name.split()[:-1]) if " " in name else name),
        ("NATIONALITY", nationality),
        ("SEX", "M"),
        ("DATE OF BIRTH", dob),
        ("PASSPORT NO.", doc_num),
        ("DATE OF ISSUE", issue),
        ("DATE OF EXPIRY", expiry),
    ]

    y_pos = 75
    for label, val in fields:
        draw.text((tx, y_pos), label, fill=(100, 110, 125))
        # If tampered, make document number or expiry misaligned and mismatched font/color
        if tampered and "PASSPORT NO" in label:
            draw.text((tx, y_pos + 13), f">> {val} << (ALTERED)", fill=(180, 0, 0))
        else:
            draw.text((tx, y_pos + 13), val, fill=(15, 25, 40))
        y_pos += 33

    # ICAO MRZ Machine Readable Zone
    draw.rectangle([(0, 350), (w, 420)], fill=(245, 245, 245), outline=(200, 205, 215))
    clean_surname = (name.split()[-1] if " " in name else name).upper()
    clean_given = ("<".join(name.split()[:-1]) if " " in name else "RAJ").upper()
    
    # Compute 6-digit YYMMDD for MRZ
    try:
        dob_parts = dob.split("-")
        dob_mrz = f"{dob_parts[0][2:]}{dob_parts[1]}{dob_parts[2]}"
    except Exception:
        dob_mrz = "900812"
        
    try:
        exp_parts = expiry.split("-")
        exp_mrz = f"{exp_parts[0][2:]}{exp_parts[1]}{exp_parts[2]}"
    except Exception:
        exp_mrz = "320518"

    mrz1 = f"P<IND{clean_surname}<<{clean_given}{'<' * 44}"[:44]
    mrz2 = f"{doc_num}<3IND{dob_mrz}1M{exp_mrz}6{'<' * 44}"[:44]
    draw.text((30, 360), mrz1, fill=(20, 20, 20))
    draw.text((30, 385), mrz2, fill=(20, 20, 20))

    buf = io.BytesIO()

    # If software tag is requested (e.g. Adobe Photoshop), inject into EXIF
    if software_tag:
        try:
            zeroth_ifd = {
                piexif.ImageIFD.Make: b"Canon",
                piexif.ImageIFD.Model: b"EOS 80D",
                piexif.ImageIFD.Software: software_tag.encode("utf-8"),
                piexif.ImageIFD.ImageDescription: b"Modified Passport Scan",
            }
            exif_dict = {"0th": zeroth_ifd}
            exif_bytes = piexif.dump(exif_dict)
            img.save(buf, format="JPEG", quality=90, exif=exif_bytes)
        except Exception:
            img.save(buf, format="JPEG", quality=90)
    else:
        img.save(buf, format="JPEG", quality=95)

    return buf.getvalue()


def _create_synthetic_live_capture(
    face_color: Tuple[int, int, int] = (200, 160, 130),
    is_same_person: bool = True
) -> bytes:
    """Generates a synthetic passenger webcam capture."""
    w, h = 480, 480
    img = Image.new("RGB", (w, h), color=(30, 35, 45))
    draw = ImageDraw.Draw(img)

    # Webcam UI overlay border
    draw.rectangle([(10, 10), (w - 10, h - 10)], outline=(0, 255, 170), width=1)
    draw.text((20, 20), "● REC LIVE [SSB CAM-04]", fill=(0, 255, 170))

    # Face Oval
    face_fill = face_color if is_same_person else (140, 100, 80)
    draw.ellipse([(140, 110), (340, 330)], fill=face_fill, outline=(80, 50, 30), width=3)

    # Eyes
    eye_offset = 0 if is_same_person else 15
    draw.ellipse([(185, 180 + eye_offset), (220, 205 + eye_offset)], fill=(255, 255, 255))
    draw.ellipse([(200, 188 + eye_offset), (212, 200 + eye_offset)], fill=(30, 20, 10))

    draw.ellipse([(260, 180 + eye_offset), (295, 205 + eye_offset)], fill=(255, 255, 255))
    draw.ellipse([(268, 188 + eye_offset), (280, 200 + eye_offset)], fill=(30, 20, 10))

    # Nose & Mouth
    draw.line([(240, 210), (235, 245), (250, 245)], fill=(120, 80, 60), width=3)
    draw.arc([(205, 260), (275, 290)], start=0, end=180, fill=(180, 60, 60), width=3)

    # Hair
    hair_fill = (35, 25, 20) if is_same_person else (90, 70, 40)
    draw.chord([(140, 95), (340, 210)], start=180, end=360, fill=hair_fill)

    # Shoulders
    shirt_fill = (40, 80, 140) if is_same_person else (160, 60, 40)
    draw.chord([(60, 340), (420, 520)], start=180, end=360, fill=shirt_fill)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


DEMO_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "1_AUTHENTIC_PASS": {
        "title": "Scenario 1: Authentic Passport (Legitimate Traveler)",
        "description": "Valid Indian passport, authentic typography, clean metadata, and matching live face.",
        "expected_status": "PASS / CLEARED",
        "doc_func": lambda: _create_synthetic_passport_image(
            name="VIKRAM PRATAP SINGH",
            doc_num="Z1839204",
            nationality="INDIAN",
            dob="1990-08-12",
            expiry="2032-05-18",
            issue="2022-05-19",
            tampered=False,
            software_tag=""
        ),
        "live_func": lambda: _create_synthetic_live_capture(is_same_person=True),
    },
    "2_TAMPERED_FORGERY": {
        "title": "Scenario 2: Tampered ID (Photoshop Traces & Splicing)",
        "description": "Document modified in Adobe Photoshop with spliced portrait borders and altered passport number.",
        "expected_status": "REJECT / FORGERY ALERT",
        "doc_func": lambda: _create_synthetic_passport_image(
            name="AMIT KUMAR VERMA",
            doc_num="Z9999999",
            nationality="INDIAN",
            dob="1988-03-22",
            expiry="2030-01-10",
            issue="2020-01-11",
            tampered=True,
            software_tag="Adobe Photoshop 2024 (Windows)"
        ),
        "live_func": lambda: _create_synthetic_live_capture(is_same_person=False),
    },
    "3_CHILD_DOB_CONTRADICTION": {
        "title": "Scenario 3: Demographic Contradiction (Child DOB + Adult Photo)",
        "description": "Document states DOB as 2021 (Age 3-5 years), but portrait depicts an adult.",
        "expected_status": "REJECT / FORGERY ALERT",
        "doc_func": lambda: _create_synthetic_passport_image(
            name="ROHIT SHARMA",
            doc_num="Z5521940",
            nationality="INDIAN",
            dob="2021-06-15",  # Child DOB
            expiry="2031-06-14",
            issue="2021-06-15",
            tampered=False,
            software_tag=""
        ),
        "live_func": lambda: _create_synthetic_live_capture(is_same_person=True),
    },
    "4_EXPIRED_MISMATCH": {
        "title": "Scenario 4: Expired Passport & Biometric Mismatch (Impostor)",
        "description": "Passport expired in 2022, and passenger presenting the ID does not match the portrait.",
        "expected_status": "REJECT / FORGERY ALERT",
        "doc_func": lambda: _create_synthetic_passport_image(
            name="SURESH CHANDRA DAS",
            doc_num="Z7182903",
            nationality="INDIAN",
            dob="1975-11-04",
            expiry="2022-04-10",  # Expired
            issue="2012-04-11",
            tampered=False,
            software_tag=""
        ),
        "live_func": lambda: _create_synthetic_live_capture(is_same_person=False),
    },
}
