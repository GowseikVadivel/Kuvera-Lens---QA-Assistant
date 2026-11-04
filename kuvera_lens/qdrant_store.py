"""Shared Qdrant client configuration for ingestion and retrieval."""

from pathlib import Path

from qdrant_client import QdrantClient

from kuvera_lens.config import Settings

ACTIVE_COLLECTION_ALIAS = "kuvera_lens_active"
DEFAULT_LOCAL_QDRANT_PATH = Path("data/qdrant")


def create_qdrant_client(
    settings: Settings | None = None,
    *,
    local_path: Path = DEFAULT_LOCAL_QDRANT_PATH,
) -> QdrantClient:
    """Create the configured Cloud client or an isolated local persistent client.

    Production uses QDRANT_URL and QDRANT_API_KEY. With neither set, development
    uses a local database at ``data/qdrant``; partial Cloud configuration fails.
    """
    settings = settings or Settings.from_environment()
    if settings.qdrant_url or settings.qdrant_api_key:
        url, api_key = settings.require_qdrant_cloud_config()
        return QdrantClient(url=url, api_key=api_key)

    local_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(local_path))
