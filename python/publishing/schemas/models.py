from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


def _clone_list(value: Optional[List[Any]]) -> List[Any]:
    return list(value) if value is not None else []


def _clone_dict(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(value) if value is not None else {}


@dataclass
class LibraryManifest:
    library_id: str
    slug: str
    title: str
    subtitle: str
    description: str
    layout_type: str
    grid_size: int
    charset: str
    box_schema: str
    status: str
    aisle_ids: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LibraryManifest":
        return cls(
            library_id=str(data["library_id"]),
            slug=str(data["slug"]),
            title=str(data["title"]),
            subtitle=str(data.get("subtitle", "")),
            description=str(data.get("description", "")),
            layout_type=str(data["layout_type"]),
            grid_size=int(data["grid_size"]),
            charset=str(data["charset"]),
            box_schema=str(data.get("box_schema", "")),
            status=str(data.get("status", "draft")),
            aisle_ids=_clone_list(data.get("aisle_ids")),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass
class AisleManifest:
    aisle_id: str
    library_id: str
    slug: str
    title: str
    description: str
    sort_order: int
    organization_principle: str
    book_ids: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AisleManifest":
        return cls(
            aisle_id=str(data["aisle_id"]),
            library_id=str(data["library_id"]),
            slug=str(data["slug"]),
            title=str(data["title"]),
            description=str(data.get("description", "")),
            sort_order=int(data.get("sort_order", 0)),
            organization_principle=str(data.get("organization_principle", "")),
            book_ids=_clone_list(data.get("book_ids")),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass
class BookManifest:
    book_id: str
    library_id: str
    aisle_id: str
    title: str
    subtitle: str
    series_name: str
    volume_number: Optional[int]
    isbn: Optional[str]
    description: str
    target_audience: str
    trim_size: str
    puzzles_per_page: int
    page_layout_profile: str
    solution_section_policy: str
    cover_theme: str
    layout_type: str
    grid_size: int
    section_ids: List[str] = field(default_factory=list)
    puzzle_count: int = 0
    publication_status: str = "draft"
    search_tags: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BookManifest":
        return cls(
            book_id=str(data["book_id"]),
            library_id=str(data["library_id"]),
            aisle_id=str(data["aisle_id"]),
            title=str(data["title"]),
            subtitle=str(data.get("subtitle", "")),
            series_name=str(data.get("series_name", "")),
            volume_number=data.get("volume_number"),
            isbn=data.get("isbn"),
            description=str(data.get("description", "")),
            target_audience=str(data.get("target_audience", "general")),
            trim_size=str(data["trim_size"]),
            puzzles_per_page=int(data["puzzles_per_page"]),
            page_layout_profile=str(data["page_layout_profile"]),
            solution_section_policy=str(data.get("solution_section_policy", "appendix")),
            cover_theme=str(data.get("cover_theme", "")),
            layout_type=str(data["layout_type"]),
            grid_size=int(data["grid_size"]),
            section_ids=_clone_list(data.get("section_ids")),
            puzzle_count=int(data.get("puzzle_count", 0)),
            publication_status=str(data.get("publication_status", "draft")),
            search_tags=_clone_list(data.get("search_tags")),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )





@dataclass
class SectionCriteria:
    weight_min: Optional[int] = None
    weight_max: Optional[int] = None
    clue_count_min: Optional[int] = None
    clue_count_max: Optional[int] = None
    technique_count_min: Optional[int] = None
    technique_count_max: Optional[int] = None

    puzzle_difficulty: Optional[str] = None
    puzzle_difficulty_in: List[str] = field(default_factory=list)

    required_techniques: List[str] = field(default_factory=list)
    required_any_techniques: List[str] = field(default_factory=list)
    excluded_techniques: List[str] = field(default_factory=list)

    featured_techniques: List[str] = field(default_factory=list)

    pattern_ids: List[str] = field(default_factory=list)
    pattern_names: List[str] = field(default_factory=list)
    pattern_family_ids: List[str] = field(default_factory=list)
    excluded_pattern_ids: List[str] = field(default_factory=list)

    required_pattern_tags_any: List[str] = field(default_factory=list)
    excluded_pattern_tags: List[str] = field(default_factory=list)

    technique_histogram_ranges: Dict[str, Dict[str, int]] = field(default_factory=dict)

    min_distinct_patterns: Optional[int] = None
    max_distinct_patterns: Optional[int] = None
    min_distinct_pattern_families: Optional[int] = None
    max_distinct_pattern_families: Optional[int] = None

    pattern_occurrence_caps: Dict[str, int] = field(default_factory=dict)
    pattern_family_occurrence_caps: Dict[str, int] = field(default_factory=dict)

    min_pattern_generation_attempts: Optional[int] = None
    min_pattern_success_rate: Optional[float] = None
    min_pattern_unique_rate: Optional[float] = None
    min_pattern_human_solvable_rate: Optional[float] = None

    random_seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "SectionCriteria":
        data = data or {}
        return cls(
            weight_min=data.get("weight_min"),
            weight_max=data.get("weight_max"),
            clue_count_min=data.get("clue_count_min"),
            clue_count_max=data.get("clue_count_max"),
            technique_count_min=data.get("technique_count_min"),
            technique_count_max=data.get("technique_count_max"),
            puzzle_difficulty=data.get("puzzle_difficulty"),
            puzzle_difficulty_in=_clone_list(data.get("puzzle_difficulty_in")),
            required_techniques=_clone_list(data.get("required_techniques")),
            required_any_techniques=_clone_list(data.get("required_any_techniques")),
            excluded_techniques=_clone_list(data.get("excluded_techniques")),
            featured_techniques=_clone_list(data.get("featured_techniques")),
            pattern_ids=_clone_list(data.get("pattern_ids")),
            pattern_names=_clone_list(data.get("pattern_names")),
            pattern_family_ids=_clone_list(data.get("pattern_family_ids")),
            excluded_pattern_ids=_clone_list(data.get("excluded_pattern_ids")),
            required_pattern_tags_any=_clone_list(data.get("required_pattern_tags_any")),
            excluded_pattern_tags=_clone_list(data.get("excluded_pattern_tags")),
            technique_histogram_ranges=_clone_dict(data.get("technique_histogram_ranges")),
            min_distinct_patterns=data.get("min_distinct_patterns"),
            max_distinct_patterns=data.get("max_distinct_patterns"),
            min_distinct_pattern_families=data.get("min_distinct_pattern_families"),
            max_distinct_pattern_families=data.get("max_distinct_pattern_families"),
            pattern_occurrence_caps=_clone_dict(data.get("pattern_occurrence_caps")),
            pattern_family_occurrence_caps=_clone_dict(data.get("pattern_family_occurrence_caps")),
            min_pattern_generation_attempts=data.get("min_pattern_generation_attempts"),
            min_pattern_success_rate=data.get("min_pattern_success_rate"),
            min_pattern_unique_rate=data.get("min_pattern_unique_rate"),
            min_pattern_human_solvable_rate=data.get("min_pattern_human_solvable_rate"),
            random_seed=data.get("random_seed"),
        )


@dataclass
class SectionManifest:
    section_id: str
    book_id: str
    section_code: str
    title: str
    subtitle: str
    section_order: int
    criteria: SectionCriteria = field(default_factory=SectionCriteria)
    difficulty_label_hint: Optional[str] = None
    puzzle_count: int = 0
    puzzle_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["criteria"] = self.criteria.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SectionManifest":
        return cls(
            section_id=str(data["section_id"]),
            book_id=str(data["book_id"]),
            section_code=str(data["section_code"]),
            title=str(data["title"]),
            subtitle=str(data.get("subtitle", "")),
            section_order=int(data.get("section_order", 0)),
            criteria=SectionCriteria.from_dict(data.get("criteria")),
            difficulty_label_hint=data.get("difficulty_label_hint"),
            puzzle_count=int(data.get("puzzle_count", 0)),
            puzzle_ids=_clone_list(data.get("puzzle_ids")),
        )


@dataclass
class PatternRecord:
    pattern_id: str
    library_id: str
    name: str
    slug: str
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    grid_size: int = 9
    layout_type: str = "classic9x9"
    mask81: str = ""
    canonical_mask_signature: str = ""
    clue_count: int = 0
    symmetry_type: str = "none"
    visual_family: str = "uncategorized"
    family_id: Optional[str] = None
    family_name: Optional[str] = None
    variant_code: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    status: str = "active"
    source_type: str = "manual"
    source_ref: Optional[str] = None
    author: Optional[str] = None
    notes: Optional[str] = None
    is_verified: bool = False
    print_score: Optional[float] = None
    legibility_score: Optional[float] = None
    aesthetic_score: Optional[float] = None
    production_stats: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PatternRecord":
        canonical_mask_signature = data.get("canonical_mask_signature") or str(data.get("mask81", ""))
        family_name = data.get("family_name") or data.get("visual_family")
        family_id = data.get("family_id")
        if family_id is None and family_name is not None:
            family_id = str(family_name).strip().lower().replace(" ", "-")
        return cls(
            pattern_id=str(data["pattern_id"]),
            library_id=str(data["library_id"]),
            name=str(data["name"]),
            slug=str(data["slug"]),
            aliases=_clone_list(data.get("aliases")),
            description=str(data.get("description", "")),
            grid_size=int(data.get("grid_size", 9)),
            layout_type=str(data.get("layout_type", "classic9x9")),
            mask81=str(data["mask81"]),
            canonical_mask_signature=str(canonical_mask_signature),
            clue_count=int(data["clue_count"]),
            symmetry_type=str(data.get("symmetry_type", "none")),
            visual_family=str(data.get("visual_family", "uncategorized")),
            family_id=str(family_id) if family_id is not None else None,
            family_name=str(family_name) if family_name is not None else None,
            variant_code=data.get("variant_code"),
            tags=_clone_list(data.get("tags")),
            status=str(data.get("status", "active")),
            source_type=str(data.get("source_type", "manual")),
            source_ref=data.get("source_ref"),
            author=data.get("author"),
            notes=data.get("notes"),
            is_verified=bool(data.get("is_verified", False)),
            print_score=data.get("print_score"),
            legibility_score=data.get("legibility_score"),
            aesthetic_score=data.get("aesthetic_score"),
            production_stats=_clone_dict(data.get("production_stats")),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass
class PrintHeader:
    display_code: str
    difficulty_label: str
    effort_label: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "PrintHeader":
        data = data or {}
        return cls(
            display_code=str(data.get("display_code", "")),
            difficulty_label=str(data.get("difficulty_label", "")),
            effort_label=str(data.get("effort_label", "")),
        )


@dataclass
class PuzzleRecord:
    record_id: str
    candidate_status: str
    solution_signature: str
    library_id: str
    aisle_id: Optional[str]
    book_id: Optional[str]
    section_id: Optional[str]
    section_code: Optional[str]
    local_puzzle_code: Optional[str]
    friendly_puzzle_id: Optional[str]
    puzzle_uid: Optional[str]
    title: str
    subtitle: str
    layout_type: str
    grid_size: int
    charset: str
    givens81: str
    solution81: str
    
    pattern_id: Optional[str]
    pattern_name: Optional[str]
    pattern_family_id: Optional[str]
    pattern_family_name: Optional[str]
    clue_count: int


    symmetry_type: Optional[str]
    is_unique: bool
    is_human_solvable: bool
    generation_method: str
    generation_seed: Optional[int]
    generator_version: Optional[str]
    weight: int
    difficulty_label: str
    difficulty_band_code: Optional[str]
    techniques_difficulty: List[str] = field(default_factory=list)
    puzzle_difficulty: str = "easy"
    difficulty_version: Optional[str] = None
    technique_count: int = 0
    techniques_used: List[str] = field(default_factory=list)
    technique_histogram: Dict[str, int] = field(default_factory=dict)
    featured_technique: Optional[str] = None
    technique_prominence_score: Optional[float] = None
    app_search_tags: List[str] = field(default_factory=list)
    book_page_number: Optional[int] = None
    position_in_section: Optional[int] = None
    position_in_book: Optional[int] = None
    print_header: PrintHeader = field(default_factory=lambda: PrintHeader("", "", ""))
    publication_status: str = "draft"
    reuse_policy: str = "book_exclusive"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["print_header"] = self.print_header.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PuzzleRecord":
        legacy_record_id = data.get("record_id") or data.get("puzzle_uid") or ""
        legacy_signature = data.get("solution_signature") or str(data.get("solution81", ""))
        return cls(
            record_id=str(legacy_record_id),
            candidate_status=str(data.get("candidate_status", "available")),
            solution_signature=str(legacy_signature),
            library_id=str(data["library_id"]),
            aisle_id=data.get("aisle_id"),
            book_id=data.get("book_id"),
            section_id=data.get("section_id"),
            section_code=data.get("section_code"),
            local_puzzle_code=data.get("local_puzzle_code"),
            friendly_puzzle_id=data.get("friendly_puzzle_id"),
            puzzle_uid=data.get("puzzle_uid"),
            title=str(data.get("title", "")),
            subtitle=str(data.get("subtitle", "")),
            layout_type=str(data["layout_type"]),
            grid_size=int(data["grid_size"]),
            charset=str(data["charset"]),
            givens81=str(data["givens81"]),
            solution81=str(data["solution81"]),
            
            pattern_id=data.get("pattern_id"),
            pattern_name=data.get("pattern_name"),
            pattern_family_id=data.get("pattern_family_id"),
            pattern_family_name=data.get("pattern_family_name"),
            clue_count=int(data["clue_count"]),


            symmetry_type=data.get("symmetry_type"),
            is_unique=bool(data["is_unique"]),
            is_human_solvable=bool(data["is_human_solvable"]),
            generation_method=str(data.get("generation_method", "")),
            generation_seed=data.get("generation_seed"),
            generator_version=data.get("generator_version"),
            weight=int(data.get("weight", 0)),
            difficulty_label=str(data.get("difficulty_label", "")),
            difficulty_band_code=data.get("difficulty_band_code"),
            techniques_difficulty=_clone_list(data.get("techniques_difficulty")),
            puzzle_difficulty=str(data.get("puzzle_difficulty", "easy")),
            difficulty_version=data.get("difficulty_version"),
            technique_count=int(data.get("technique_count", 0)),
            techniques_used=_clone_list(data.get("techniques_used")),
            technique_histogram=_clone_dict(data.get("technique_histogram")),
            featured_technique=data.get("featured_technique"),
            technique_prominence_score=data.get("technique_prominence_score"),
            app_search_tags=_clone_list(data.get("app_search_tags")),
            book_page_number=data.get("book_page_number"),
            position_in_section=data.get("position_in_section"),
            position_in_book=data.get("position_in_book"),
            print_header=PrintHeader.from_dict(data.get("print_header")),
            publication_status=str(data.get("publication_status", "draft")),
            reuse_policy=str(data.get("reuse_policy", "book_exclusive")),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass
class BookSummary:
    book_id: str
    title: str
    subtitle: str
    aisle_id: str
    puzzle_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BookSummary":
        return cls(
            book_id=str(data["book_id"]),
            title=str(data["title"]),
            subtitle=str(data.get("subtitle", "")),
            aisle_id=str(data["aisle_id"]),
            puzzle_count=int(data.get("puzzle_count", 0)),
        )


@dataclass
class CatalogManifest:
    catalog_version: str
    generated_at: str
    library_ids: List[str] = field(default_factory=list)
    libraries: List[LibraryManifest] = field(default_factory=list)
    book_summaries: List[BookSummary] = field(default_factory=list)
    index_files: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "catalog_version": self.catalog_version,
            "generated_at": self.generated_at,
            "library_ids": list(self.library_ids),
            "libraries": [library.to_dict() for library in self.libraries],
            "book_summaries": [summary.to_dict() for summary in self.book_summaries],
            "index_files": dict(self.index_files),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CatalogManifest":
        return cls(
            catalog_version=str(data["catalog_version"]),
            generated_at=str(data["generated_at"]),
            library_ids=_clone_list(data.get("library_ids")),
            libraries=[LibraryManifest.from_dict(x) for x in data.get("libraries", [])],
            book_summaries=[BookSummary.from_dict(x) for x in data.get("book_summaries", [])],
            index_files=_clone_dict(data.get("index_files")),
        )
    


@dataclass
class PrintFormatSpec:
    format_id: str
    vendor: str
    binding_type: str
    trim_width_in: float
    trim_height_in: float
    bleed_in: float
    safe_margin_in: float
    inside_margin_in: float
    outside_margin_in: float
    top_margin_in: float
    bottom_margin_in: float
    supports_spine: bool = True
    supports_isbn: bool = True
    paper_options: List[str] = field(default_factory=list)
    color_options: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PrintFormatSpec":
        return cls(
            format_id=str(data["format_id"]),
            vendor=str(data["vendor"]),
            binding_type=str(data["binding_type"]),
            trim_width_in=float(data["trim_width_in"]),
            trim_height_in=float(data["trim_height_in"]),
            bleed_in=float(data.get("bleed_in", 0.0)),
            safe_margin_in=float(data.get("safe_margin_in", 0.0)),
            inside_margin_in=float(data.get("inside_margin_in", data.get("safe_margin_in", 0.0))),
            outside_margin_in=float(data.get("outside_margin_in", data.get("safe_margin_in", 0.0))),
            top_margin_in=float(data.get("top_margin_in", data.get("safe_margin_in", 0.0))),
            bottom_margin_in=float(data.get("bottom_margin_in", data.get("safe_margin_in", 0.0))),
            supports_spine=bool(data.get("supports_spine", True)),
            supports_isbn=bool(data.get("supports_isbn", True)),
            paper_options=_clone_list(data.get("paper_options")),
            color_options=_clone_list(data.get("color_options")),
            description=str(data.get("description", "")),
        )


@dataclass
class ChannelPreset:
    channel_id: str
    vendor: str
    binding_type: str
    description: str = ""
    default_format_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChannelPreset":
        return cls(
            channel_id=str(data["channel_id"]),
            vendor=str(data["vendor"]),
            binding_type=str(data["binding_type"]),
            description=str(data.get("description", "")),
            default_format_ids=_clone_list(data.get("default_format_ids")),
        )


@dataclass
class MarginProfile:
    profile_id: str
    mirrored: bool
    inside_margin_in: float
    outside_margin_in: float
    top_margin_in: float
    bottom_margin_in: float
    bleed_in: float
    safe_margin_in: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MarginProfile":
        return cls(
            profile_id=str(data["profile_id"]),
            mirrored=bool(data.get("mirrored", False)),
            inside_margin_in=float(data.get("inside_margin_in", 0.0)),
            outside_margin_in=float(data.get("outside_margin_in", 0.0)),
            top_margin_in=float(data.get("top_margin_in", 0.0)),
            bottom_margin_in=float(data.get("bottom_margin_in", 0.0)),
            bleed_in=float(data.get("bleed_in", 0.0)),
            safe_margin_in=float(data.get("safe_margin_in", 0.0)),
        )


@dataclass
class PageBlock:
    page_type: str
    template_id: str
    section_id: Optional[str] = None
    page_index: Optional[int] = None
    physical_page_number: Optional[int] = None
    logical_page_number: Optional[int] = None
    show_page_number: bool = True
    page_number_style: Optional[str] = "arabic"
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_type": self.page_type,
            "template_id": self.template_id,
            "section_id": self.section_id,
            "page_index": self.page_index,
            "physical_page_number": self.physical_page_number,
            "logical_page_number": self.logical_page_number,
            "show_page_number": self.show_page_number,
            "page_number_style": self.page_number_style,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PageBlock":
        return cls(
            page_type=str(data["page_type"]),
            template_id=str(data["template_id"]),
            section_id=data.get("section_id"),
            page_index=data.get("page_index"),
            physical_page_number=data.get("physical_page_number"),
            logical_page_number=data.get("logical_page_number"),
            show_page_number=bool(data.get("show_page_number", True)),
            page_number_style=data.get("page_number_style", "arabic"),
            payload=_clone_dict(data.get("payload")),
        )


@dataclass
class InteriorPlan:
    page_blocks: List[PageBlock] = field(default_factory=list)
    estimated_page_count: int = 0
    requires_blank_page_adjustment: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_blocks": [block.to_dict() for block in self.page_blocks],
            "estimated_page_count": self.estimated_page_count,
            "requires_blank_page_adjustment": self.requires_blank_page_adjustment,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InteriorPlan":
        return cls(
            page_blocks=[PageBlock.from_dict(x) for x in data.get("page_blocks", [])],
            estimated_page_count=int(data.get("estimated_page_count", 0)),
            requires_blank_page_adjustment=bool(data.get("requires_blank_page_adjustment", False)),
            notes=_clone_list(data.get("notes")),
        )


@dataclass
class PublicationLayoutConfig:
    puzzles_per_page: Optional[int] = None
    rows: Optional[int] = None
    cols: Optional[int] = None

    # Optional independent answer-key layout.
    # Example: puzzle pages 4up, solution pages 12up.
    solution_puzzles_per_page: Optional[int] = None
    solution_rows: Optional[int] = None
    solution_cols: Optional[int] = None

    inner_margin_in: Optional[float] = None
    outer_margin_in: Optional[float] = None
    top_margin_in: Optional[float] = None
    bottom_margin_in: Optional[float] = None
    header_height_in: Optional[float] = None
    footer_height_in: Optional[float] = None
    gutter_x_in: Optional[float] = None
    gutter_y_in: Optional[float] = None
    tile_slot_padding_in: Optional[float] = None
    tile_header_band_height_in: Optional[float] = None
    tile_gap_below_header_in: Optional[float] = None
    tile_bottom_padding_in: Optional[float] = None
    font_family: Optional[str] = None
    given_digit_scale: Optional[float] = None
    solution_digit_scale: Optional[float] = None
    given_digit_size_preset: Optional[str] = None
    solution_digit_size_preset: Optional[str] = None
    language: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "PublicationLayoutConfig":
        data = data or {}
        return cls(
            puzzles_per_page=data.get("puzzles_per_page"),
            rows=data.get("rows"),
            cols=data.get("cols"),
            solution_puzzles_per_page=data.get("solution_puzzles_per_page"),
            solution_rows=data.get("solution_rows"),
            solution_cols=data.get("solution_cols"),
            inner_margin_in=data.get("inner_margin_in"),
            outer_margin_in=data.get("outer_margin_in"),
            top_margin_in=data.get("top_margin_in"),
            bottom_margin_in=data.get("bottom_margin_in"),
            header_height_in=data.get("header_height_in"),
            footer_height_in=data.get("footer_height_in"),
            gutter_x_in=data.get("gutter_x_in"),
            gutter_y_in=data.get("gutter_y_in"),
            tile_slot_padding_in=data.get("tile_slot_padding_in"),
            tile_header_band_height_in=data.get("tile_header_band_height_in"),
            tile_gap_below_header_in=data.get("tile_gap_below_header_in"),
            tile_bottom_padding_in=data.get("tile_bottom_padding_in"),
            font_family=data.get("font_family"),
            given_digit_scale=data.get("given_digit_scale"),
            solution_digit_scale=data.get("solution_digit_scale"),
            given_digit_size_preset=data.get("given_digit_size_preset"),
            solution_digit_size_preset=data.get("solution_digit_size_preset"),
            language=data.get("language"),
        )


@dataclass
class PublicationSpec:
    publication_id: str
    book_id: str
    channel_id: str
    format_id: str
    include_cover: bool = True
    include_solutions: bool = True
    mirror_margins: bool = True
    front_matter_profile: str = "minimal_front_matter"
    end_matter_profile: str = "none"
    section_separator_policy: str = "section_openers"
    blank_page_policy: str = "none"
    page_numbering_policy: str = "physical_all_suppress_blank_only"
    puzzle_page_template: str = "classic_4up_clean"
    solution_page_template: str = "solution_4up_basic"
    cover_template: str = "basic_full_wrap"
    paper_type: str = "white_bw"

    # Interior export policy:
    # - "no_bleed": export trim-size interior only.
    # - "bleed": export KDP/vendor bleed-size interior only.
    # - "both": export both interior_no_bleed.pdf and interior_bleed.pdf,
    #           plus interior.pdf as the preferred marketplace-ready file.
    interior_bleed_mode: str = "both"

    layout_config: PublicationLayoutConfig = field(default_factory=PublicationLayoutConfig)

    # Wave 1 schema additions for richer publication choreography.
    front_matter_sequence: List[str] = field(default_factory=list)
    section_prelude_sequence: List[str] = field(default_factory=list)
    recto_start_policy: Dict[str, Any] = field(default_factory=dict)
    features_page_config: Dict[str, Any] = field(default_factory=dict)
    section_preview_config: Dict[str, Any] = field(default_factory=dict)
    editorial_copy: Dict[str, Any] = field(default_factory=dict)
    ecosystem_config: Dict[str, Any] = field(default_factory=dict)

    # Optional opener/transition page before the answer key.
    # Used for books like B05 where puzzles are large-print 4up
    # but solutions may be compact 12up.
    solution_section_config: Dict[str, Any] = field(default_factory=dict)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "publication_id": self.publication_id,
            "book_id": self.book_id,
            "channel_id": self.channel_id,
            "format_id": self.format_id,
            "include_cover": self.include_cover,
            "include_solutions": self.include_solutions,
            "mirror_margins": self.mirror_margins,
            "front_matter_profile": self.front_matter_profile,
            "end_matter_profile": self.end_matter_profile,
            "section_separator_policy": self.section_separator_policy,
            "blank_page_policy": self.blank_page_policy,
            "page_numbering_policy": self.page_numbering_policy,
            "puzzle_page_template": self.puzzle_page_template,
            "solution_page_template": self.solution_page_template,
            "cover_template": self.cover_template,
            "paper_type": self.paper_type,
            "interior_bleed_mode": self.interior_bleed_mode,
            "layout_config": self.layout_config.to_dict(),
            "front_matter_sequence": list(self.front_matter_sequence),
            "section_prelude_sequence": list(self.section_prelude_sequence),
            "recto_start_policy": dict(self.recto_start_policy),
            "features_page_config": dict(self.features_page_config),
            "section_preview_config": dict(self.section_preview_config),
            "editorial_copy": dict(self.editorial_copy),
            "ecosystem_config": dict(self.ecosystem_config),
            "solution_section_config": dict(self.solution_section_config),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PublicationSpec":
        return cls(
            publication_id=str(data["publication_id"]),
            book_id=str(data["book_id"]),
            channel_id=str(data["channel_id"]),
            format_id=str(data["format_id"]),
            include_cover=bool(data.get("include_cover", True)),
            include_solutions=bool(data.get("include_solutions", True)),
            mirror_margins=bool(data.get("mirror_margins", True)),
            front_matter_profile=str(data.get("front_matter_profile", "minimal_front_matter")),
            end_matter_profile=str(data.get("end_matter_profile", "none")),
            section_separator_policy=str(data.get("section_separator_policy", "section_openers")),
            blank_page_policy=str(data.get("blank_page_policy", "none")),
            page_numbering_policy=str(
                data.get("page_numbering_policy", "physical_all_suppress_blank_only")
            ),
            puzzle_page_template=str(data.get("puzzle_page_template", "classic_4up_clean")),
            solution_page_template=str(data.get("solution_page_template", "solution_4up_basic")),
            cover_template=str(data.get("cover_template", "basic_full_wrap")),
            paper_type=str(data.get("paper_type", "white_bw")),
            interior_bleed_mode=str(data.get("interior_bleed_mode", "both")),
            layout_config=PublicationLayoutConfig.from_dict(data.get("layout_config")),
            front_matter_sequence=_clone_list(data.get("front_matter_sequence")),
            section_prelude_sequence=_clone_list(data.get("section_prelude_sequence")),
            recto_start_policy=_clone_dict(data.get("recto_start_policy")),
            features_page_config=_clone_dict(data.get("features_page_config")),
            section_preview_config=_clone_dict(data.get("section_preview_config")),
            editorial_copy=_clone_dict(data.get("editorial_copy")),
            ecosystem_config=_clone_dict(data.get("ecosystem_config")),
            solution_section_config=_clone_dict(data.get("solution_section_config")),
            metadata=_clone_dict(data.get("metadata")),
        )


@dataclass
class CoverSpec:
    cover_id: str
    publication_id: str
    format_id: str
    page_count: int
    paper_type: str
    spine_width_in: float
    front_design_asset: Optional[str] = None
    back_design_asset: Optional[str] = None
    spine_text: str = ""
    back_copy: str = ""
    author_imprint: str = ""
    isbn: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CoverSpec":
        return cls(
            cover_id=str(data["cover_id"]),
            publication_id=str(data["publication_id"]),
            format_id=str(data["format_id"]),
            page_count=int(data.get("page_count", 0)),
            paper_type=str(data.get("paper_type", "white_bw")),
            spine_width_in=float(data.get("spine_width_in", 0.0)),
            front_design_asset=data.get("front_design_asset"),
            back_design_asset=data.get("back_design_asset"),
            spine_text=str(data.get("spine_text", "")),
            back_copy=str(data.get("back_copy", "")),
            author_imprint=str(data.get("author_imprint", "")),
            isbn=data.get("isbn"),
        )


@dataclass
class PublicationPackage:
    publication_id: str
    book_id: str
    book_dir: str
    publication_dir: str
    format_id: str
    channel_id: str
    publication_manifest_path: str
    interior_plan_path: str
    cover_spec_path: Optional[str] = None
    generated_at: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PublicationPackage":
        return cls(
            publication_id=str(data["publication_id"]),
            book_id=str(data["book_id"]),
            book_dir=str(data["book_dir"]),
            publication_dir=str(data["publication_dir"]),
            format_id=str(data["format_id"]),
            channel_id=str(data["channel_id"]),
            publication_manifest_path=str(data["publication_manifest_path"]),
            interior_plan_path=str(data["interior_plan_path"]),
            cover_spec_path=data.get("cover_spec_path"),
            generated_at=data.get("generated_at"),
            warnings=_clone_list(data.get("warnings")),
        )