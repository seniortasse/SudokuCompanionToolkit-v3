from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List


MaskBuilder = Callable[[int], str]


@dataclass(frozen=True)
class FormulaPatternSpec:
    family_id: str
    family_name: str
    base_name: str
    description_template: str
    tags: List[str]
    builder: MaskBuilder


def _rows_to_mask(rows: List[str]) -> str:
    if len(rows) != 9 or any(len(row) != 9 for row in rows):
        raise ValueError("Formula pattern rows must be exactly 9 rows of length 9")
    return "".join(rows)


def _mirror_h(rows: List[str]) -> List[str]:
    return [row[::-1] for row in rows]


def _mirror_v(rows: List[str]) -> List[str]:
    return list(rows[::-1])


def _rotate_180(rows: List[str]) -> List[str]:
    return [row[::-1] for row in rows[::-1]]


def _apply_variant(rows: List[str], variant: int) -> List[str]:
    mode = int(variant) % 4
    if mode == 0:
        return rows
    if mode == 1:
        return _mirror_h(rows)
    if mode == 2:
        return _mirror_v(rows)
    return _rotate_180(rows)


def _diamond_rows() -> List[str]:
    return [
        "000010000",
        "000111000",
        "001111100",
        "011111110",
        "111111111",
        "011111110",
        "001111100",
        "000111000",
        "000010000",
    ]


def _hourglass_rows() -> List[str]:
    return [
        "111000111",
        "011101110",
        "001111100",
        "000111000",
        "000010000",
        "000111000",
        "001111100",
        "011101110",
        "111000111",
    ]


def _crosswind_rows() -> List[str]:
    return [
        "100010001",
        "110101011",
        "011111110",
        "001111100",
        "111111111",
        "001111100",
        "011111110",
        "110101011",
        "100010001",
    ]


def _lantern_rows() -> List[str]:
    return [
        "000111000",
        "001111100",
        "011010110",
        "111010111",
        "111111111",
        "111010111",
        "011010110",
        "001111100",
        "000111000",
    ]


def _orbit_rows() -> List[str]:
    return [
        "001111100",
        "011000110",
        "110000011",
        "100111001",
        "100101001",
        "100111001",
        "110000011",
        "011000110",
        "001111100",
    ]


def _build_from_rows(base_rows: List[str], variant: int) -> str:
    return _rows_to_mask(_apply_variant(base_rows, variant))


FORMULA_FAMILIES: Dict[str, FormulaPatternSpec] = {
    "diamond": FormulaPatternSpec(
        family_id="diamond",
        family_name="Diamond",
        base_name="Diamond Bloom",
        description_template="A centered diamond silhouette with a clean, geometric rhythm.",
        tags=["generated", "geometric", "diamond", "balanced"],
        builder=lambda variant: _build_from_rows(_diamond_rows(), variant),
    ),
    "hourglass": FormulaPatternSpec(
        family_id="hourglass",
        family_name="Hourglass",
        base_name="Hourglass Gate",
        description_template="A tapering hourglass profile that pulls the eye toward the center.",
        tags=["generated", "geometric", "hourglass", "balanced"],
        builder=lambda variant: _build_from_rows(_hourglass_rows(), variant),
    ),
    "crosswind": FormulaPatternSpec(
        family_id="crosswind",
        family_name="Crosswind",
        base_name="Crosswind Lattice",
        description_template="A wind-swept lattice with a strong center and tapered diagonals.",
        tags=["generated", "geometric", "lattice", "balanced"],
        builder=lambda variant: _build_from_rows(_crosswind_rows(), variant),
    ),
    "lantern": FormulaPatternSpec(
        family_id="lantern",
        family_name="Lantern",
        base_name="Lantern Core",
        description_template="A glowing lantern-like core framed by symmetrical outer contours.",
        tags=["generated", "decorative", "lantern", "symmetry"],
        builder=lambda variant: _build_from_rows(_lantern_rows(), variant),
    ),
    "orbit": FormulaPatternSpec(
        family_id="orbit",
        family_name="Orbit",
        base_name="Orbit Ring",
        description_template="A ringed orbital motif with a strong silhouette and central chamber.",
        tags=["generated", "decorative", "orbit", "ring"],
        builder=lambda variant: _build_from_rows(_orbit_rows(), variant),
    ),
}


def available_formula_families() -> List[str]:
    return sorted(FORMULA_FAMILIES.keys())


def build_formula_pattern(family_id: str, variant: int) -> dict:
    key = str(family_id).strip().lower()
    if key not in FORMULA_FAMILIES:
        raise KeyError(f"Unknown formula family: {family_id}")

    spec = FORMULA_FAMILIES[key]
    mask81 = spec.builder(int(variant))

    return {
        "family_id": spec.family_id,
        "family_name": spec.family_name,
        "name": f"{spec.base_name} {int(variant) + 1}",
        "description": spec.description_template,
        "mask81": mask81,
        "tags": list(spec.tags),
        "source_ref": "pattern_generator_formulas_v1",
    }