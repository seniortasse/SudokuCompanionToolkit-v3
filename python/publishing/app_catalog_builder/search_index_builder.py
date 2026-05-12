from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

from python.publishing.schemas.models import PuzzleRecord
from python.publishing.techniques.technique_catalog import get_public_technique_name


def _sorted_unique(values: Iterable[str]) -> List[str]:
    return sorted({str(v) for v in values if str(v).strip()})


def build_puzzles_by_technique_index(puzzle_records: Iterable[PuzzleRecord]) -> Dict[str, List[str]]:
    index = defaultdict(list)

    for record in puzzle_records:
        for technique in record.techniques_used:
            # Keep this index customer/app-facing by using public technique names.
            # Internal engine IDs remain inside puzzle records themselves.
            public_name = get_public_technique_name(str(technique), plural=True)
            index[public_name].append(record.puzzle_uid)

    return {key: _sorted_unique(values) for key, values in sorted(index.items())}


def build_puzzles_by_weight_band_index(puzzle_records: Iterable[PuzzleRecord]) -> Dict[str, List[str]]:
    index = defaultdict(list)

    for record in puzzle_records:
        if record.difficulty_label:
            index[record.difficulty_label].append(record.puzzle_uid)
        if record.difficulty_band_code:
            index[str(record.difficulty_band_code).lower()].append(record.puzzle_uid)

    return {key: _sorted_unique(values) for key, values in sorted(index.items())}


def build_puzzles_by_pattern_index(puzzle_records: Iterable[PuzzleRecord]) -> Dict[str, List[str]]:
    index = defaultdict(list)

    for record in puzzle_records:
        if record.pattern_id:
            index[record.pattern_id].append(record.puzzle_uid)

    return {key: _sorted_unique(values) for key, values in sorted(index.items())}


def build_book_by_title_index(puzzle_records: Iterable[PuzzleRecord]) -> Dict[str, str]:
    index: Dict[str, str] = {}

    for record in puzzle_records:
        if not record.book_id:
            continue

        # MVP placeholder: until full BookManifest exists, map book_id as its own lookup title.
        normalized_title = str(record.book_id).strip().lower()
        index[normalized_title] = record.book_id

    return dict(sorted(index.items()))