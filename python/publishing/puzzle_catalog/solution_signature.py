from __future__ import annotations


def _validate_solution81(solution81: str) -> str:
    value = str(solution81).strip()
    if len(value) != 81:
        raise ValueError(f"solution81 must be exactly 81 characters long, got {len(value)}")
    return value


def build_solution_signature(solution81: str) -> str:
    """
    Canonicalize a solved grid by first-seen symbol remapping.

    Example:
    - exact same solved grid -> same signature
    - any digit-substitution-equivalent solved grid -> same signature

    For classic 9x9 Sudoku, solutions are expected to contain symbols from the charset,
    but this canonicalizer intentionally only depends on position-wise symbol identity.
    """
    value = _validate_solution81(solution81)

    symbol_map: dict[str, str] = {}
    next_digit = 1
    out: list[str] = []

    for ch in value:
        if ch not in symbol_map:
            if next_digit > 9:
                raise ValueError(
                    "solution81 contains more than 9 distinct symbols; cannot build classic9 signature"
                )
            symbol_map[ch] = str(next_digit)
            next_digit += 1
        out.append(symbol_map[ch])

    return "".join(out)