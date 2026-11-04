"""Publish active chunks and their vectors to a versioned Qdrant collection."""

import argparse
import hashlib
import json
import math
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol
from urllib.parse import urlsplit

import numpy as np
from qdrant_client import models

from kuvera_lens.embedding import (
    DEFAULT_CHUNK_MANIFEST,
    DEFAULT_EMBEDDING_DIRECTORY,
    MODEL_NAME,
    MODEL_PROFILE,
    ChunkRecord,
    load_active_chunks,
)
from kuvera_lens.qdrant_store import ACTIVE_COLLECTION_ALIAS, create_qdrant_client
from kuvera_lens.schemes import SCHEME_IDS
from kuvera_lens.sources import OFFICIAL_HOSTS, ApprovedSource, validate_source

COLLECTION_PREFIX = "kuvera_lens_g_"
UPSERT_BATCH_SIZE = 100
SCROLL_PAGE_SIZE = 256
_REQUIRED_METADATA = (
    "source_id",
    "scheme_id",
    "scheme_name",
    "source_url",
    "source_domain",
    "document_title",
    "document_type",
    "content_hash",
    "document_version",
)


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: str
    point_id: uuid.UUID
    text: str
    metadata: dict[str, str | int | float | bool]
    embedding: np.ndarray


@dataclass(frozen=True)
class IndexStats:
    updated: bool
    chunks_indexed: int
    schemes_indexed: int
    collection_name: str
    generation: str
    collection_alias: str = ACTIVE_COLLECTION_ALIAS


class QdrantClient(Protocol):
    def collection_exists(self, collection_name: str) -> bool:
        ...

    def create_collection(
        self,
        collection_name: str,
        *,
        vectors_config: object,
    ) -> bool:
        ...

    def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        *,
        field_schema: object,
        wait: bool = True,
    ) -> object:
        ...

    def get_collection(self, collection_name: str) -> object:
        ...

    def upsert(
        self,
        collection_name: str,
        *,
        points: list[object],
        wait: bool = True,
    ) -> object:
        ...

    def count(self, collection_name: str, *, exact: bool = True) -> object:
        ...

    def scroll(
        self,
        collection_name: str,
        *,
        limit: int,
        offset: object | None = None,
        with_payload: list[str] | bool = True,
        with_vectors: bool = False,
    ) -> tuple[list[object], object | None]:
        ...

    def get_aliases(self) -> object:
        ...

    def update_collection_aliases(self, *, change_aliases_operations: list[object]) -> bool:
        ...


class _IndexConsistencyError(RuntimeError):
    """A stored Qdrant generation does not match the active inputs."""


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON file {path}: {exc}") from exc


def _resolve_manifest_path(root: Path, relative_path: object, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        raise ValueError(f"{label} must be a non-empty POSIX relative path.")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Unsafe {label}: {relative_path!r}")
    try:
        resolved_root = root.resolve(strict=True)
        candidate = root.joinpath(*relative.parts).resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"Cannot resolve {label} {relative_path!r}: {exc}") from exc
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"{label} escapes its managed directory: {relative_path!r}")
    if not candidate.is_file():
        raise ValueError(f"{label} is not a file: {relative_path!r}")
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunk_id_digest(records: list[ChunkRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.chunk_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _point_id(chunk_id: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"kuvera-lens:chunk:{chunk_id}")


def _validate_metadata(
    metadata: object,
    record: ChunkRecord,
    line_number: int,
) -> dict[str, str | int | float | bool]:
    if not isinstance(metadata, dict):
        raise ValueError(f"Missing chunk metadata at {record.chunk_file}:{line_number}.")
    for key in _REQUIRED_METADATA:
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Required citation metadata {key!r} is missing at "
                f"{record.chunk_file}:{line_number}."
            )
    if (
        metadata["source_id"] != record.source_id
        or metadata["scheme_id"] != record.scheme_id
        or metadata["content_hash"] != record.content_hash
        or record.scheme_id not in SCHEME_IDS
    ):
        raise ValueError(
            f"Chunk provenance mismatch at {record.chunk_file}:{line_number}."
        )
    source_url = metadata["source_url"]
    source_domain = metadata["source_domain"]
    hostname = urlsplit(source_url).hostname
    if not hostname or hostname.lower().rstrip(".") != source_domain.lower().rstrip("."):
        raise ValueError(
            f"Source URL/domain mismatch at {record.chunk_file}:{line_number}."
        )
    organization = next(
        (
            name
            for name, hosts in OFFICIAL_HOSTS.items()
            if hostname.lower().rstrip(".") in hosts
        ),
        None,
    )
    if organization is None:
        raise ValueError(
            f"Source host is not allowlisted at {record.chunk_file}:{line_number}."
        )
    try:
        validate_source(
            ApprovedSource(
                source_id=record.source_id,
                scheme_id=record.scheme_id,
                organization=organization,
                url=source_url,
            )
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid source provenance at {record.chunk_file}:{line_number}: {exc}"
        ) from exc

    result: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"Invalid metadata key at {record.chunk_file}:{line_number}.")
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(
                    f"Non-finite metadata value at {record.chunk_file}:{line_number}."
                )
            result[key] = value
        else:
            raise ValueError(
                f"Unsupported Qdrant payload value for {key!r} at "
                f"{record.chunk_file}:{line_number}."
            )
    checked_at = result.get("last_checked_at")
    if isinstance(checked_at, str):
        result.setdefault("retrieved_at", checked_at)
    return result


def load_index_chunks(
    chunk_manifest: Path = DEFAULT_CHUNK_MANIFEST,
    embedding_manifest: Path = DEFAULT_EMBEDDING_DIRECTORY / "active_embeddings.json",
) -> tuple[list[IndexedChunk], str, int]:
    """Validate and join the active chunks and embeddings in their manifest order."""
    grouped = load_active_chunks(chunk_manifest)
    manifest = _read_json(embedding_manifest)
    if not isinstance(manifest, dict):
        raise ValueError(f"Embedding manifest must contain a JSON object: {embedding_manifest}")
    if (
        manifest.get("model_name") != MODEL_NAME
        or manifest.get("model_profile") != MODEL_PROFILE
        or manifest.get("normalized") is not True
        or manifest.get("dtype") != "float32"
    ):
        raise ValueError("Active embedding manifest has an unsupported model profile.")
    dimensions = manifest.get("dimensions")
    if not isinstance(dimensions, int) or dimensions < 1:
        raise ValueError("Active embedding manifest has invalid vector dimensions.")
    embedding_documents = manifest.get("documents")
    if not isinstance(embedding_documents, list):
        raise ValueError("Active embedding manifest has no documents list.")
    by_source: dict[str, dict[str, object]] = {}
    for item in embedding_documents:
        if not isinstance(item, dict):
            raise ValueError("Active embedding manifest contains an invalid document entry.")
        source_id = item.get("source_id")
        if not isinstance(source_id, str) or source_id in by_source:
            raise ValueError("Active embedding manifest has missing or duplicate source IDs.")
        by_source[source_id] = item
    if set(by_source) != set(grouped):
        raise ValueError("Active chunk and embedding manifests reference different documents.")
    expected_chunk_count = manifest.get("chunks")
    if (
        not isinstance(expected_chunk_count, int)
        or expected_chunk_count != sum(len(records) for records in grouped.values())
    ):
        raise ValueError("Active embedding manifest chunk count does not match active chunks.")

    chunk_root = chunk_manifest.parent
    embedding_root = embedding_manifest.parent
    indexed_chunks: list[IndexedChunk] = []
    fingerprint = hashlib.sha256()
    fingerprint.update(MODEL_PROFILE.encode("utf-8"))
    fingerprint.update(str(dimensions).encode("ascii"))

    for source_id, records in sorted(grouped.items()):
        entry = by_source[source_id]
        first_record = records[0]
        chunk_digest = _chunk_id_digest(records)
        if (
            entry.get("scheme_id") != first_record.scheme_id
            or entry.get("content_hash") != first_record.content_hash
            or entry.get("chunk_id_digest") != chunk_digest
            or entry.get("chunk_file") != first_record.chunk_file
            or entry.get("chunk_count") != len(records)
            or entry.get("model_profile") != MODEL_PROFILE
            or entry.get("dimensions") != dimensions
        ):
            raise ValueError(f"Embedding metadata does not match active chunks for {source_id!r}.")

        chunk_path = _resolve_manifest_path(
            chunk_root, first_record.chunk_file, "chunk file"
        )
        try:
            chunk_rows = [
                json.loads(line)
                for line in chunk_path.read_text(encoding="utf-8").splitlines()
            ]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read chunk records from {chunk_path}: {exc}") from exc
        if len(chunk_rows) != len(records):
            raise ValueError(f"Chunk row count changed for {source_id!r}.")

        vector_path = _resolve_manifest_path(
            embedding_root, entry.get("path"), "embedding file"
        )
        expected_file_hash = entry.get("file_sha256")
        if (
            not isinstance(expected_file_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_file_hash)
            or _sha256_file(vector_path) != expected_file_hash
        ):
            raise ValueError(f"Embedding file checksum mismatch for {source_id!r}.")
        try:
            with np.load(vector_path, allow_pickle=False) as vectors_file:
                vector_ids = vectors_file["chunk_ids"]
                vectors = vectors_file["embeddings"]
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"Cannot read embeddings for {source_id!r}: {exc}") from exc
        expected_ids = [record.chunk_id for record in records]
        if (
            vector_ids.ndim != 1
            or vector_ids.tolist() != expected_ids
            or vectors.ndim != 2
            or vectors.shape != (len(records), dimensions)
            or vectors.dtype != np.float32
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)
        ):
            raise ValueError(f"Invalid, unaligned, or non-normalized vectors for {source_id!r}.")

        for index, (record, row) in enumerate(zip(records, chunk_rows)):
            if not isinstance(row, dict) or row.get("chunk_id") != record.chunk_id:
                raise ValueError(f"Chunk order mismatch for {source_id!r}.")
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty chunk text for {record.chunk_id!r}.")
            metadata = _validate_metadata(row.get("metadata"), record, index + 1)
            metadata["chunk_id"] = record.chunk_id
            metadata["index_generation"] = ""
            indexed_chunks.append(
                IndexedChunk(
                    chunk_id=record.chunk_id,
                    point_id=_point_id(record.chunk_id),
                    text=text,
                    metadata=metadata,
                    embedding=vectors[index].copy(),
                )
            )

        fingerprint.update(source_id.encode("utf-8"))
        fingerprint.update(first_record.scheme_id.encode("utf-8"))
        fingerprint.update(first_record.content_hash.encode("ascii"))
        fingerprint.update(chunk_digest.encode("ascii"))
        fingerprint.update(expected_file_hash.encode("ascii"))

    generation = fingerprint.hexdigest()
    for chunk in indexed_chunks:
        chunk.metadata["index_generation"] = generation
    return indexed_chunks, generation, dimensions


def _active_collection_name(generation: str) -> str:
    return f"{COLLECTION_PREFIX}{generation[:32]}"


def _get_alias_target(client: QdrantClient) -> str | None:
    aliases = client.get_aliases().aliases
    matches = [
        alias.collection_name
        for alias in aliases
        if alias.alias_name == ACTIVE_COLLECTION_ALIAS
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Qdrant returned duplicate {ACTIVE_COLLECTION_ALIAS!r} aliases.")
    return matches[0] if matches else None


def _verify_collection(
    client: QdrantClient,
    collection_name: str,
    expected_chunks: list[IndexedChunk],
    generation: str,
    dimensions: int,
) -> None:
    info = client.get_collection(collection_name=collection_name)
    vectors = info.config.params.vectors
    size = getattr(vectors, "size", None)
    if size != dimensions:
        raise _IndexConsistencyError(
            f"Qdrant vector dimension mismatch: expected {dimensions}, found {size}."
        )
    count = client.count(collection_name=collection_name, exact=True).count
    if count != len(expected_chunks):
        raise _IndexConsistencyError("Qdrant point count does not match active chunks.")
    expected_by_id = {str(chunk.point_id): chunk for chunk in expected_chunks}
    seen_ids: set[str] = set()
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for point in points:
            point_id = str(point.id)
            expected = expected_by_id.get(point_id)
            if expected is None or point_id in seen_ids:
                raise _IndexConsistencyError(
                    "Qdrant point IDs do not match active chunk IDs."
                )
            seen_ids.add(point_id)
            if (
                point.payload is None
                or point.payload.get("index_generation")
                != generation
                or point.payload.get("chunk_id") != expected.chunk_id
                or point.payload.get("text") != expected.text
            ):
                raise _IndexConsistencyError(
                    "Qdrant point payload does not match the active chunk generation."
                )
            for key, value in expected.metadata.items():
                if point.payload.get(key) != value:
                    raise _IndexConsistencyError(
                        f"Qdrant point metadata mismatch for {key!r}."
                    )
            vector = np.asarray(point.vector, dtype=np.float32)
            if (
                vector.shape != (dimensions,)
                or not np.isfinite(vector).all()
                or not np.allclose(vector, expected.embedding, rtol=1e-5, atol=1e-6)
            ):
                raise _IndexConsistencyError(
                    "Qdrant vector does not match the active embedding."
                )
        if next_offset is None:
            break
        offset = next_offset
    if seen_ids != set(expected_by_id):
        raise _IndexConsistencyError("Qdrant point IDs do not match active chunk IDs.")


def _publish_alias(
    client: QdrantClient,
    collection_name: str,
    previous_collection: str | None,
) -> None:
    operations: list[object] = []
    if previous_collection is not None:
        operations.append(
            models.DeleteAliasOperation(
                delete_alias=models.DeleteAlias(alias_name=ACTIVE_COLLECTION_ALIAS)
            )
        )
    operations.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(
                collection_name=collection_name,
                alias_name=ACTIVE_COLLECTION_ALIAS,
            )
        )
    )
    client.update_collection_aliases(change_aliases_operations=operations)
    if _get_alias_target(client) != collection_name:
        raise RuntimeError("Qdrant did not confirm the active collection alias.")


def _ensure_scheme_id_index(client: QdrantClient, collection_name: str) -> None:
    client.create_payload_index(
        collection_name=collection_name,
        field_name="scheme_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True,
    )


def index_active_embeddings(
    client: QdrantClient,
    chunk_manifest: Path = DEFAULT_CHUNK_MANIFEST,
    embedding_manifest: Path = DEFAULT_EMBEDDING_DIRECTORY / "active_embeddings.json",
    *,
    batch_size: int = UPSERT_BATCH_SIZE,
) -> IndexStats:
    """Build and verify a new Qdrant generation before atomically switching alias."""
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero.")
    chunks, generation, dimensions = load_index_chunks(chunk_manifest, embedding_manifest)
    if not chunks:
        raise ValueError("There are no active chunks to index.")
    target_collection = _active_collection_name(generation)
    previous_collection = _get_alias_target(client)

    if previous_collection == target_collection:
        try:
            _ensure_scheme_id_index(client, target_collection)
            _verify_collection(client, target_collection, chunks, generation, dimensions)
            return IndexStats(
                updated=False,
                chunks_indexed=0,
                schemes_indexed=len({str(chunk.metadata["scheme_id"]) for chunk in chunks}),
                collection_name=target_collection,
                generation=generation,
            )
        except _IndexConsistencyError:
            target_collection = f"{target_collection}_{uuid.uuid4().hex[:8]}"

    if client.collection_exists(collection_name=target_collection):
        try:
            _verify_collection(client, target_collection, chunks, generation, dimensions)
        except _IndexConsistencyError:
            target_collection = f"{target_collection}_{uuid.uuid4().hex[:8]}"

    if not client.collection_exists(collection_name=target_collection):
        client.create_collection(
            collection_name=target_collection,
            vectors_config=models.VectorParams(
                size=dimensions,
                distance=models.Distance.COSINE,
            ),
        )
    _ensure_scheme_id_index(client, target_collection)

    points = [
        models.PointStruct(
            id=chunk.point_id,
            vector=chunk.embedding.tolist(),
            payload={**chunk.metadata, "text": chunk.text},
        )
        for chunk in chunks
    ]
    for start in range(0, len(points), batch_size):
        client.upsert(
            collection_name=target_collection,
            points=points[start : start + batch_size],
            wait=True,
        )
    _verify_collection(client, target_collection, chunks, generation, dimensions)
    _publish_alias(client, target_collection, previous_collection)

    return IndexStats(
        updated=True,
        chunks_indexed=len(chunks),
        schemes_indexed=len({str(chunk.metadata["scheme_id"]) for chunk in chunks}),
        collection_name=target_collection,
        generation=generation,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish active chunks and vectors to Qdrant."
    )
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNK_MANIFEST)
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=DEFAULT_EMBEDDING_DIRECTORY / "active_embeddings.json",
    )
    parser.add_argument("--batch-size", type=int, default=UPSERT_BATCH_SIZE)
    args = parser.parse_args(argv)
    try:
        import httpx
        from qdrant_client.http.exceptions import ApiException
    except ImportError as exc:
        print(f"Qdrant indexing failed; install qdrant-client first: {exc}", file=sys.stderr)
        return 1
    try:
        client = create_qdrant_client()
        stats = index_active_embeddings(
            client,
            args.chunks,
            args.embeddings,
            batch_size=args.batch_size,
        )
    except (OSError, ValueError, RuntimeError, ApiException, httpx.HTTPError) as exc:
        print(
            "Qdrant indexing failed. The active collection alias is changed only "
            f"after the new generation is verified: {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"Updated: {stats.updated}")
    print(f"Chunks indexed: {stats.chunks_indexed}")
    print(f"Schemes indexed: {stats.schemes_indexed}")
    print(f"Collection: {stats.collection_name}")
    print(f"Active alias: {stats.collection_alias}")
    print(f"Generation: {stats.generation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
