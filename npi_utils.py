"""NPI (National Provider Identifier) helpers.

Single source of truth for the Luhn check used by the grader and the synthetic
case generator. Both layers must agree byte-for-byte: if one drifts, valid
generated NPIs would start failing the grader's validation.

NPI standard: a 10-digit identifier whose check digit is the Luhn checksum of
``"80840" + first 9 digits``.
"""

from __future__ import annotations

import random


__all__ = ["validate_npi_luhn", "generate_valid_npi"]


def validate_npi_luhn(npi: str) -> bool:
    """Simplified NPI Luhn check (prefix 80840 + 9 digits with check digit)."""
    if not npi or len(npi) != 10 or not npi.isdigit():
        return False
    # Standard Luhn on "80840" + NPI[:-1]
    full = "80840" + npi[:-1]
    total = 0
    for i, ch in enumerate(reversed(full)):
        n = int(ch)
        if i % 2 == 0:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    check = (10 - (total % 10)) % 10
    return check == int(npi[-1])


def generate_valid_npi(rng: random.Random) -> str:
    """Generate a valid 10-digit NPI with Luhn check digit (prefix 80840)."""
    base = "".join([str(rng.randint(0, 9)) for _ in range(9)])
    full = "80840" + base
    total = 0
    for i, ch in enumerate(reversed(full)):
        n = int(ch)
        if i % 2 == 0:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    check = (10 - (total % 10)) % 10
    return base + str(check)
