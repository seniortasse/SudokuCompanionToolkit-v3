from __future__ import annotations


def validate_mask81(mask81: str) -> str:
    value = str(mask81).strip()
    if len(value) != 81:
        raise ValueError(f"mask81 must be exactly 81 characters long, got {len(value)}")
    invalid = {ch for ch in value if ch not in {"0", "1"}}
    if invalid:
        raise ValueError(f"mask81 contains invalid symbols: {sorted(invalid)}")
    return value


def build_canonical_mask_signature(mask81: str) -> str:
    """
    For now, canonical mask identity is the exact normalized 81-bit mask string.

    This deliberately leaves room for future equivalence expansion
    (e.g. symmetry-normalized pattern identity), without changing callers.
    """
    return validate_mask81(mask81)


def build_variant_code(*, family_name: str | None, ordinal: int | None = None) -> str:
    family = str(family_name or "base").strip().lower().replace(" ", "-")
    if ordinal is None:
        return family
    return f"{family}-v{int(ordinal):02d}"