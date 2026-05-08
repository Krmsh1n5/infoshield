"""
pipeline/ingestors — InfoShield cascade ingestors package.

Exposes a unified interface for ingesting propagation graphs from
different platforms into the InfoShield pipeline.

Quick start
-----------
    from pipeline.ingestors import InstagramIngestor

    # Mock mode (no credentials needed):
    ingestor = InstagramIngestor(mock=True, seed=42)
    G = ingestor.get_post_cascade("any_post_id")

    # Real mode:
    ingestor = InstagramIngestor(access_token="IGQV...")
    G = ingestor.get_post_cascade("17858893269000001")

    # Then feed directly into the InfoShield pipeline:
    from pipeline.run_pipeline import simulate_cascade_following
    from pipeline.sbm_fitter import find_root_user
    root = find_root_user(G)
    ...
"""

from pipeline.ingestors.base_ingestor import BaseIngestor, ValidationReport
from pipeline.ingestors.instagram_ingestor import InstagramIngestor

__all__ = [
    "BaseIngestor",
    "ValidationReport",
    "InstagramIngestor",
]
