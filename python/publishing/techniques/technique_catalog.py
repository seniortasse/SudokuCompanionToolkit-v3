from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass(frozen=True)
class TechniqueCatalogEntry:
    engine_id: str
    canonical_id: str
    public_name: str
    public_name_plural: str
    family: str
    commonness: str
    is_standard_sudoku: bool
    is_project_specific: bool
    short_description: str


def normalize_technique_id(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _norm(value: str) -> str:
    return normalize_technique_id(value)


TECHNIQUE_CATALOG: Dict[str, TechniqueCatalogEntry] = {
    "singles_1": TechniqueCatalogEntry(
        engine_id="singles_1",
        canonical_id="full_house",
        public_name="Full House",
        public_name_plural="Full Houses",
        family="singles",
        commonness="standard",
        is_standard_sudoku=True,
        is_project_specific=False,
        short_description="A house has only one empty cell, so the missing digit is forced.",
    ),
    "singles_2": TechniqueCatalogEntry(
        engine_id="singles_2",
        canonical_id="hidden_single",
        public_name="Hidden Single",
        public_name_plural="Hidden Singles",
        family="singles",
        commonness="standard",
        is_standard_sudoku=True,
        is_project_specific=False,
        short_description="A digit has only one possible cell in a row, column, or box.",
    ),
    "singles_3": TechniqueCatalogEntry(
        engine_id="singles_3",
        canonical_id="hidden_single",
        public_name="Hidden Single",
        public_name_plural="Hidden Singles",
        family="singles",
        commonness="standard",
        is_standard_sudoku=True,
        is_project_specific=False,
        short_description="A digit has only one possible cell in a row, column, or box.",
    ),
    "singles_naked_2": TechniqueCatalogEntry(
        engine_id="singles_naked_2",
        canonical_id="naked_single",
        public_name="Naked Single",
        public_name_plural="Naked Singles",
        family="singles",
        commonness="standard",
        is_standard_sudoku=True,
        is_project_specific=False,
        short_description="A cell has only one possible candidate left.",
    ),
    "singles_naked_3": TechniqueCatalogEntry(
        engine_id="singles_naked_3",
        canonical_id="naked_single",
        public_name="Naked Single",
        public_name_plural="Naked Singles",
        family="singles",
        commonness="standard",
        is_standard_sudoku=True,
        is_project_specific=False,
        short_description="A cell has only one possible candidate left.",
    ),
    "doubles_naked": TechniqueCatalogEntry(
        engine_id="doubles_naked",
        canonical_id="naked_pair",
        public_name="Naked Pair",
        public_name_plural="Naked Pairs",
        family="naked_subsets",
        commonness="standard",
        is_standard_sudoku=True,
        is_project_specific=False,
        short_description="Two cells in a house contain only the same two candidates between them.",
    ),
    "triplets_naked": TechniqueCatalogEntry(
        engine_id="triplets_naked",
        canonical_id="naked_triple",
        public_name="Naked Triple",
        public_name_plural="Naked Triples",
        family="naked_subsets",
        commonness="standard",
        is_standard_sudoku=True,
        is_project_specific=False,
        short_description="Three cells in a house contain only three candidates between them.",
    ),
    "quads_naked": TechniqueCatalogEntry(
        engine_id="quads_naked",
        canonical_id="naked_quad",
        public_name="Naked Quad",
        public_name_plural="Naked Quads",
        family="naked_subsets",
        commonness="standard",
        is_standard_sudoku=True,
        is_project_specific=False,
        short_description="Four cells in a house contain only four candidates between them.",
    ),
    "doubles": TechniqueCatalogEntry(
        engine_id="doubles",
        canonical_id="hidden_pair",
        public_name="Hidden Pair",
        public_name_plural="Hidden Pairs",
        family="hidden_subsets",
        commonness="standard",
        is_standard_sudoku=True,
        is_project_specific=False,
        short_description="Two candidates appear only in the same two cells of a house.",
    ),
    "triplets": TechniqueCatalogEntry(
        engine_id="triplets",
        canonical_id="hidden_triple",
        public_name="Hidden Triple",
        public_name_plural="Hidden Triples",
        family="hidden_subsets",
        commonness="standard",
        is_standard_sudoku=True,
        is_project_specific=False,
        short_description="Three candidates appear only in the same three cells of a house.",
    ),
    "quads": TechniqueCatalogEntry(
        engine_id="quads",
        canonical_id="hidden_quad",
        public_name="Hidden Quad",
        public_name_plural="Hidden Quads",
        family="hidden_subsets",
        commonness="standard",
        is_standard_sudoku=True,
        is_project_specific=False,
        short_description="Four candidates appear only in the same four cells of a house.",
    ),
    "singles_pointing": TechniqueCatalogEntry(
        engine_id="singles_pointing",
        canonical_id="locked_candidates_pointing",
        public_name="Pointing Pair / Triple",
        public_name_plural="Pointing Pairs / Triples",
        family="locked_candidates",
        commonness="standard",
        is_standard_sudoku=True,
        is_project_specific=False,
        short_description="A digit is confined to one row or column inside a box, eliminating it from the rest of that line.",
    ),
    "singles_boxed": TechniqueCatalogEntry(
        engine_id="singles_boxed",
        canonical_id="locked_candidates_claiming",
        public_name="Claiming Pair / Triple",
        public_name_plural="Claiming Pairs / Triples",
        family="locked_candidates",
        commonness="standard",
        is_standard_sudoku=True,
        is_project_specific=False,
        short_description="A digit is confined to one box inside a row or column, eliminating it from the rest of that box.",
    ),
    "x_wings": TechniqueCatalogEntry(
        engine_id="x_wings",
        canonical_id="x_wing",
        public_name="X-Wing",
        public_name_plural="X-Wings",
        family="fish",
        commonness="standard",
        is_standard_sudoku=True,
        is_project_specific=False,
        short_description="A size-2 fish pattern that eliminates a candidate from matching rows or columns.",
    ),
    "x_wings_3": TechniqueCatalogEntry(
        engine_id="x_wings_3",
        canonical_id="swordfish",
        public_name="Swordfish",
        public_name_plural="Swordfish",
        family="fish",
        commonness="standard",
        is_standard_sudoku=True,
        is_project_specific=False,
        short_description="A size-3 fish pattern that generalizes the X-Wing.",
    ),
    "x_wings_4": TechniqueCatalogEntry(
        engine_id="x_wings_4",
        canonical_id="jellyfish",
        public_name="Jellyfish",
        public_name_plural="Jellyfish",
        family="fish",
        commonness="standard",
        is_standard_sudoku=True,
        is_project_specific=False,
        short_description="A size-4 fish pattern that generalizes X-Wing and Swordfish.",
    ),
    "y_wings": TechniqueCatalogEntry(
        engine_id="y_wings",
        canonical_id="xy_wing",
        public_name="XY-Wing",
        public_name_plural="XY-Wings",
        family="wings",
        commonness="standard",
        is_standard_sudoku=True,
        is_project_specific=False,
        short_description="A three-cell bivalue pattern using a pivot and two wings to eliminate a shared candidate.",
    ),
    "remote_pairs": TechniqueCatalogEntry(
        engine_id="remote_pairs",
        canonical_id="remote_pairs",
        public_name="Remote Pairs",
        public_name_plural="Remote Pairs",
        family="chains",
        commonness="standard",
        is_standard_sudoku=True,
        is_project_specific=False,
        short_description="A chain of bivalue cells with the same pair of candidates that forces eliminations.",
    ),
    "ab_chains": TechniqueCatalogEntry(
        engine_id="ab_chains",
        canonical_id="xy_chain",
        public_name="XY-Chain",
        public_name_plural="XY-Chains",
        family="chains",
        commonness="standard",
        is_standard_sudoku=True,
        is_project_specific=False,
        short_description="A chain through bivalue cells that forces a candidate placement or elimination.",
    ),
    "ab_rings": TechniqueCatalogEntry(
        engine_id="ab_rings",
        canonical_id="xy_loop",
        public_name="XY-Loop",
        public_name_plural="XY-Loops",
        family="chains",
        commonness="semi_standard",
        is_standard_sudoku=True,
        is_project_specific=False,
        short_description="A closed bivalue chain that creates eliminations around the loop.",
    ),
    "boxed_doubles": TechniqueCatalogEntry(
        engine_id="boxed_doubles",
        canonical_id="boxed_pair",
        public_name="Boxed Pair",
        public_name_plural="Boxed Pairs",
        family="box_patterns",
        commonness="project_specific",
        is_standard_sudoku=True,
        is_project_specific=True,
        short_description="A Sudoku Companion box-pattern technique that forces a two-candidate target cell through box and line constraints.",
    ),
    "boxed_triplets": TechniqueCatalogEntry(
        engine_id="boxed_triplets",
        canonical_id="boxed_triple",
        public_name="Boxed Triple",
        public_name_plural="Boxed Triples",
        family="box_patterns",
        commonness="project_specific",
        is_standard_sudoku=True,
        is_project_specific=True,
        short_description="A Sudoku Companion box-pattern technique that forces a three-candidate target cell through box and line constraints.",
    ),
    "boxed_quads": TechniqueCatalogEntry(
        engine_id="boxed_quads",
        canonical_id="boxed_quad",
        public_name="Boxed Quad",
        public_name_plural="Boxed Quads",
        family="box_patterns",
        commonness="project_specific",
        is_standard_sudoku=True,
        is_project_specific=True,
        short_description="A Sudoku Companion box-pattern technique that forces a four-candidate target cell through box and line constraints.",
    ),
    "boxed_wings": TechniqueCatalogEntry(
        engine_id="boxed_wings",
        canonical_id="box_wing",
        public_name="Box-Wing",
        public_name_plural="Box-Wings",
        family="box_patterns",
        commonness="project_specific",
        is_standard_sudoku=True,
        is_project_specific=True,
        short_description="A Sudoku Companion wing pattern where two outside wings interact with a box to remove a candidate.",
    ),
    "boxed_rays": TechniqueCatalogEntry(
        engine_id="boxed_rays",
        canonical_id="box_ray",
        public_name="Box-Ray",
        public_name_plural="Box-Rays",
        family="box_patterns",
        commonness="project_specific",
        is_standard_sudoku=True,
        is_project_specific=True,
        short_description="A Sudoku Companion box pattern using a bent ray across boxes to remove a candidate.",
    ),
}


for i in range(1, 10):
    TECHNIQUE_CATALOG[f"leftovers_{i}"] = TechniqueCatalogEntry(
        engine_id=f"leftovers_{i}",
        canonical_id="law_of_leftovers",
        public_name="Law of Leftovers",
        public_name_plural="Law of Leftovers",
        family="custom_layout_logic",
        commonness="specialized",
        is_standard_sudoku=False,
        is_project_specific=False,
        short_description="A custom-box Sudoku technique based on balancing inside and outside regions.",
    )


def get_technique_entry(engine_id: str) -> TechniqueCatalogEntry | None:
    return TECHNIQUE_CATALOG.get(_norm(engine_id))


def get_public_technique_name(engine_id: str, *, plural: bool = False) -> str:
    entry = get_technique_entry(engine_id)
    if not entry:
        return str(engine_id)
    return entry.public_name_plural if plural else entry.public_name


def collapse_to_public_names(engine_ids: Iterable[str], *, plural: bool = True) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for raw_id in engine_ids:
        entry = get_technique_entry(raw_id)
        if not entry:
            name = str(raw_id)
            key = _norm(name)
        else:
            name = entry.public_name_plural if plural else entry.public_name
            key = entry.canonical_id
        if key not in seen:
            seen.add(key)
            names.append(name)
    return names


def public_combo_label(value: str, *, plural: bool = True, separator: str = " + ") -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""

    parts = [part.strip() for part in raw.split("+") if part.strip()]
    if len(parts) <= 1:
        return get_public_technique_name(raw, plural=plural)

    return separator.join(collapse_to_public_names(parts, plural=plural))