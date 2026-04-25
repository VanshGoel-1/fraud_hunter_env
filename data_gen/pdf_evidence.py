"""
CMS-1500-style scanned claim form generator (Pillow-only).

Every output is a single-image PDF — the same thing a real whistleblower
hands over: a photocopy of a physician's paper claim. This means the agent
*must* OCR. There is no embedded text layer in any mode.

Tier gates OCR *difficulty*, not format:
  * tier ≤ 2 : clean scan — upright, high contrast, crisp font.
  * tier == 3: light degradation — mild noise + 0.5° skew.
  * tier >= 4: heavy degradation — gaussian noise, 2-3° skew, JPEG artifacts,
               faint coffee-stain ellipse, lower contrast. Tesseract typically
               lands 60-80% token recall on this tier, which is what we want.

No third-party PDF lib is required — `Image.save(..., format="PDF")` works
because cpu-basic HF Spaces ship Pillow.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont


# 8.5" x 11" @ 150 DPI
PAGE_W, PAGE_H = 1275, 1650
MARGIN = 60


@dataclass
class ClaimEvidence:
    """Data that gets rendered onto the paper form AND returned so the
    compiler can seed `evidence_documents.expected_fields`."""
    claim_id: str
    beneficiary_id: str
    beneficiary_dob: str
    beneficiary_dod: str | None
    provider_name: str
    provider_npi: str
    service_date: str
    hcpcs_code: str
    icd9_code: str
    amount: float
    diagnosis_text: str = "Essential hypertension, unspecified"


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try a few common system fonts; fall back to default bitmap."""
    for name in ("arial.ttf", "calibri.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _draw_form(img: Image.Image, ev: ClaimEvidence) -> None:
    draw = ImageDraw.Draw(img)
    title_font = _load_font(32)
    h_font = _load_font(20)
    body_font = _load_font(18)
    small_font = _load_font(14)

    # Header
    draw.rectangle((MARGIN, MARGIN, PAGE_W - MARGIN, MARGIN + 80), outline="black", width=2)
    draw.text((MARGIN + 20, MARGIN + 14), "HEALTH INSURANCE CLAIM FORM", fill="black", font=title_font)
    draw.text((MARGIN + 20, MARGIN + 56), "APPROVED BY NATIONAL UNIFORM CLAIM COMMITTEE  (CMS-1500)",
              fill="black", font=small_font)

    y = MARGIN + 110

    # Section 1 — patient block
    def field(label: str, value: str, yy: int, width: int = 560) -> None:
        draw.text((MARGIN, yy), label, fill="black", font=h_font)
        draw.rectangle((MARGIN, yy + 24, MARGIN + width, yy + 58), outline="black", width=1)
        draw.text((MARGIN + 10, yy + 30), value, fill="black", font=body_font)

    field("1a. INSURED'S I.D. NUMBER (MEDICARE)", ev.beneficiary_id, y)
    field("3. PATIENT'S BIRTH DATE", ev.beneficiary_dob, y, width=360)
    # right-side DOB box needs to not overlap; shift
    draw.rectangle((MARGIN + 600, y + 24, MARGIN + 960, y + 58), outline="black", width=1)
    draw.text((MARGIN + 600, y), "3a. DATE OF DEATH (IF ANY)", fill="black", font=h_font)
    draw.text((MARGIN + 610, y + 30),
              ev.beneficiary_dod if ev.beneficiary_dod else "—", fill="black", font=body_font)

    y += 90
    field("24A. DATE OF SERVICE", ev.service_date, y, width=360)
    draw.text((MARGIN + 400, y), "24D. PROCEDURES, SERVICES (HCPCS/CPT)", fill="black", font=h_font)
    draw.rectangle((MARGIN + 400, y + 24, MARGIN + 700, y + 58), outline="black", width=1)
    draw.text((MARGIN + 410, y + 30), ev.hcpcs_code, fill="black", font=body_font)

    draw.text((MARGIN + 740, y), "21. DIAGNOSIS (ICD-9)", fill="black", font=h_font)
    draw.rectangle((MARGIN + 740, y + 24, MARGIN + 1000, y + 58), outline="black", width=1)
    draw.text((MARGIN + 750, y + 30), ev.icd9_code, fill="black", font=body_font)

    y += 90
    field(f"21a. DIAGNOSIS NARRATIVE", ev.diagnosis_text, y, width=900)

    y += 90
    field("24F. CHARGES (USD)", f"${ev.amount:,.2f}", y, width=360)
    draw.text((MARGIN + 400, y), "CLAIM ID", fill="black", font=h_font)
    draw.rectangle((MARGIN + 400, y + 24, MARGIN + 780, y + 58), outline="black", width=1)
    draw.text((MARGIN + 410, y + 30), ev.claim_id, fill="black", font=body_font)

    y += 90
    field("31. SIGNATURE OF PHYSICIAN", ev.provider_name, y, width=600)
    draw.text((MARGIN + 640, y), "33a. PROVIDER NPI", fill="black", font=h_font)
    draw.rectangle((MARGIN + 640, y + 24, MARGIN + 960, y + 58), outline="black", width=1)
    draw.text((MARGIN + 650, y + 30), ev.provider_npi, fill="black", font=body_font)

    # Footer
    draw.text((MARGIN, PAGE_H - MARGIN - 20),
              f"FORM CMS-1500 (02-12)   Page 1 of 1   Claim {ev.claim_id}",
              fill="black", font=small_font)


def _apply_degradation(img: Image.Image, tier: int, rng: random.Random) -> Image.Image:
    """Tier-gated realism pass: skew, noise, contrast, coffee stains."""
    if tier <= 2:
        return img  # clean scan

    # Slight random skew
    max_deg = 0.5 if tier == 3 else (2.0 if tier == 4 else 3.0)
    angle = rng.uniform(-max_deg, max_deg)
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor="white", expand=False)

    # Gaussian blur to simulate photocopy / scanner defocus
    if tier >= 4:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))

    # Salt-and-pepper noise
    px = img.load()
    if px is not None:
        noise_rate = {3: 0.003, 4: 0.010, 5: 0.020}.get(tier, 0.005)
        n_pixels = PAGE_W * PAGE_H
        n_noise = int(n_pixels * noise_rate)
        for _ in range(n_noise):
            x = rng.randint(0, PAGE_W - 1)
            y = rng.randint(0, PAGE_H - 1)
            px[x, y] = 0 if rng.random() < 0.5 else 255

    # Coffee stain (tier 5 only) — faint brown ellipse
    if tier >= 5:
        stain = Image.new("RGBA", img.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(stain)
        cx = rng.randint(200, PAGE_W - 200)
        cy = rng.randint(200, PAGE_H - 200)
        r = rng.randint(80, 160)
        sd.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(120, 80, 40, 40))
        stain = stain.filter(ImageFilter.GaussianBlur(radius=12))
        img = Image.alpha_composite(img.convert("RGBA"), stain).convert("RGB")

    # Lower contrast on heavy tiers via JPEG round-trip
    if tier >= 4:
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=55)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    return img


def render_claim_pdf(
    out_path: Path,
    ev: ClaimEvidence,
    tier: int,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """
    Render a single-page CMS-1500 scan to `out_path` (PDF) and return a dict
    of the fields the agent is expected to extract.
    """
    rng = rng or random.Random()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    _draw_form(img, ev)
    img = _apply_degradation(img, tier, rng)
    img.save(out_path, format="PDF", resolution=150.0)

    return {
        "claim_id": ev.claim_id,
        "beneficiary_id": ev.beneficiary_id,
        "service_date": ev.service_date,
        "hcpcs_code": ev.hcpcs_code,
        "icd9_code": ev.icd9_code,
        "amount": ev.amount,
        "provider_npi": ev.provider_npi,
        "provider_name": ev.provider_name,
    }
