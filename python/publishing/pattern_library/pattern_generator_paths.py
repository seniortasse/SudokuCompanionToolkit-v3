from __future__ import annotations

from typing import Dict, List


def _rows_to_mask(rows: List[str]) -> str:
    if len(rows) != 9 or any(len(row) != 9 for row in rows):
        raise ValueError("Path pattern rows must be exactly 9 rows of length 9")
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


PATH_FAMILIES: Dict[str, dict] = {
    "serpentine": {
        "family_id": "serpentine",
        "family_name": "Serpentine",
        "base_name": "Double Serpent",
        "description": "Two serpentine bands sweep across the grid like mirrored labyrinth walkers.",
        "tags": ["generated", "artistic", "worm", "maze", "serpentine"],
        "rows": [
            "111000111",
            "011101110",
            "001111100",
            "000111000",
            "001111100",
            "011101110",
            "111000111",
            "011101110",
            "001111100",
        ],
    },
    "twin-rivers": {
        "family_id": "twin-rivers",
        "family_name": "Twin Rivers",
        "base_name": "Twin Rivers",
        "description": "Two flowing corridors meander through the board like paired rivers.",
        "tags": ["generated", "artistic", "river", "corridor", "flow"],
        "rows": [
            "110000011",
            "111000111",
            "011101110",
            "001111100",
            "000111000",
            "001111100",
            "011101110",
            "111000111",
            "110000011",
        ],
    },
    "woven-corridor": {
        "family_id": "woven-corridor",
        "family_name": "Woven Corridor",
        "base_name": "Woven Corridor",
        "description": "A woven corridor motif with alternating bands crossing through the middle.",
        "tags": ["generated", "artistic", "woven", "corridor", "maze"],
        "rows": [
            "101111101",
            "001111100",
            "011010110",
            "111000111",
            "111111111",
            "111000111",
            "011010110",
            "001111100",
            "101111101",
        ],
    },
    "castle-walk": {
        "family_id": "castle-walk",
        "family_name": "Castle Walk",
        "base_name": "Castle Walk",
        "description": "A battlement-like pathway suggesting courtyards, walls, and narrow passages.",
        "tags": ["generated", "artistic", "castle", "walk", "courtyard"],
        "rows": [
            "111010111",
            "111010111",
            "001111100",
            "001010100",
            "111111111",
            "001010100",
            "001111100",
            "111010111",
            "111010111",
        ],
    },
    "spiral": {
        "family_id": "spiral",
        "family_name": "Spiral",
        "base_name": "Spiral Gate",
        "description": "A spiral-like gateway motif that draws the eye inward through staged turns.",
        "tags": ["generated", "artistic", "spiral", "gate", "labyrinth"],
        "rows": [
            "111111000",
            "100001000",
            "101111110",
            "101000010",
            "101011010",
            "101010010",
            "101011110",
            "100000000",
            "111111111",
        ],
    },
}


def available_path_families() -> List[str]:
    return sorted(PATH_FAMILIES.keys())


def build_path_pattern(family_id: str, variant: int) -> dict:
    key = str(family_id).strip().lower()
    if key not in PATH_FAMILIES:
        raise KeyError(f"Unknown path family: {family_id}")

    spec = PATH_FAMILIES[key]
    rows = _apply_variant(spec["rows"], int(variant))
    mask81 = _rows_to_mask(rows)

    return {
        "family_id": spec["family_id"],
        "family_name": spec["family_name"],
        "name": f"{spec['base_name']} {int(variant) + 1}",
        "description": spec["description"],
        "mask81": mask81,
        "tags": list(spec["tags"]),
        "source_ref": "pattern_generator_paths_v1",
    }