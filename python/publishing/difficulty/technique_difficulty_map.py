from __future__ import annotations

from python.publishing.techniques.technique_catalog import normalize_technique_id

DIFFICULTY_VERSION = "technique-profile-v3"

EASY = "easy"
MEDIUM = "medium"
HARD = "hard"
VERY_HARD = "very_hard"
EXPERT = "expert"
GENIUS = "genius"


def normalize_technique_name(value: str) -> str:
    return normalize_technique_id(value)


TECHNIQUE_DIFFICULTY_MAP = {
    # easy
    "singles_1": EASY,
    "singles_2": EASY,
    "singles_3": EASY,

    # medium
    "singles_naked_2": MEDIUM,
    "singles_naked_3": MEDIUM,

    # hard
    "doubles_naked": HARD,
    "singles_pointing": HARD,
    "singles_boxed": HARD,
    "doubles": HARD,
    "triplets": HARD,
    "triplets_naked": HARD,
    

    # very hard
    "x_wings": VERY_HARD,
    "y_wings": VERY_HARD,
    "remote_pairs": VERY_HARD,
    "boxed_doubles": VERY_HARD,
    

    # expert
    "quads_naked": EXPERT,
    "quads": EXPERT,
    "boxed_triplets": EXPERT,
    "boxed_quads": EXPERT,
    "boxed_wings": EXPERT,
    "boxed_rays": EXPERT,
    "ab_rings": EXPERT,
    "ab_chains": EXPERT,
    "x_wings_3": EXPERT,
    "x_wings_4": EXPERT,
}


def get_technique_difficulty(technique_name: str) -> str | None:
    key = normalize_technique_name(technique_name)
    if not key:
        return None
    return TECHNIQUE_DIFFICULTY_MAP.get(key)


def list_known_technique_difficulties() -> dict[str, str]:
    return dict(TECHNIQUE_DIFFICULTY_MAP)