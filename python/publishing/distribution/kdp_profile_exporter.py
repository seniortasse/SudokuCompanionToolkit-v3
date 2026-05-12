from __future__ import annotations

import json
from pathlib import Path

from python.publishing.distribution.metadata_models import (
    DistributionPackageMetadata,
    MarketplaceMetadata,
)


def export_kdp_profile_json(
    *,
    distribution_metadata: DistributionPackageMetadata,
    output_path: Path,
) -> Path:
    marketplace = distribution_metadata.marketplace or MarketplaceMetadata(marketplace="amazon_kdp")

    payload = {
        "platform": "amazon_kdp",
        "publication_id": distribution_metadata.publication_id,
        "book_id": distribution_metadata.book_id,
        "title": distribution_metadata.title,
        "subtitle": distribution_metadata.subtitle,
        "author_name": marketplace.contributor_name,
        "description": marketplace.description,
        "keywords": list(marketplace.keywords),
        "categories": list(marketplace.categories),
        "language": marketplace.language,
        "series_name": marketplace.series_name,
        "audience": marketplace.audience,
        "imprint_name": distribution_metadata.imprint.imprint_name if distribution_metadata.imprint else "",
        "publisher_name": distribution_metadata.imprint.publisher_name if distribution_metadata.imprint else "",
        "isbn13": distribution_metadata.isbn.isbn13 if distribution_metadata.isbn else "",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path