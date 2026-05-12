from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


def _clone_list(value) -> List[Any]:
    return list(value) if value is not None else []


def _clone_dict(value) -> Dict[str, Any]:
    return dict(value) if value is not None else {}


@dataclass
class ImprintMetadata:
    imprint_id: str
    imprint_name: str
    publisher_name: str = ""
    website: str = ""
    contact_email: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ImprintMetadata":
        return cls(
            imprint_id=str(data.get("imprint_id", "")),
            imprint_name=str(data.get("imprint_name", "")),
            publisher_name=str(data.get("publisher_name", "")),
            website=str(data.get("website", "")),
            contact_email=str(data.get("contact_email", "")),
        )


@dataclass
class IsbnAssignment:
    isbn13: str
    isbn10: str = ""
    assignment_name: str = ""
    status: str = "active"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IsbnAssignment":
        return cls(
            isbn13=str(data.get("isbn13", "")),
            isbn10=str(data.get("isbn10", "")),
            assignment_name=str(data.get("assignment_name", "")),
            status=str(data.get("status", "active")),
        )


@dataclass
class MarketplaceMetadata:
    marketplace: str
    language: str = "English"
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    audience: str = ""
    contributor_name: str = ""
    series_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "marketplace": self.marketplace,
            "language": self.language,
            "description": self.description,
            "keywords": list(self.keywords),
            "categories": list(self.categories),
            "audience": self.audience,
            "contributor_name": self.contributor_name,
            "series_name": self.series_name,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MarketplaceMetadata":
        return cls(
            marketplace=str(data.get("marketplace", "")),
            language=str(data.get("language", "English")),
            description=str(data.get("description", "")),
            keywords=_clone_list(data.get("keywords")),
            categories=_clone_list(data.get("categories")),
            audience=str(data.get("audience", "")),
            contributor_name=str(data.get("contributor_name", "")),
            series_name=str(data.get("series_name", "")),
        )


@dataclass
class DistributionPackageMetadata:
    publication_id: str
    book_id: str
    title: str
    subtitle: str = ""
    imprint: Optional[ImprintMetadata] = None
    isbn: Optional[IsbnAssignment] = None
    marketplace: Optional[MarketplaceMetadata] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "publication_id": self.publication_id,
            "book_id": self.book_id,
            "title": self.title,
            "subtitle": self.subtitle,
            "imprint": self.imprint.to_dict() if self.imprint else None,
            "isbn": self.isbn.to_dict() if self.isbn else None,
            "marketplace": self.marketplace.to_dict() if self.marketplace else None,
            "extra": dict(self.extra),
        }