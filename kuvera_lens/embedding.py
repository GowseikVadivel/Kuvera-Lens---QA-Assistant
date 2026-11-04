"""Generate normalized sentence embeddings for the active chunk set."""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Protocol

import numpy as np

from kuvera_lens.chunking import DEFAULT_OUTPUT_DIRECTORY
from kuvera_lens.schemes import SCHEME_IDS

DEFAULT_CHUNK_MANIFEST = DEFAULT_OUTPUT_DIRECTORY / "active_chunks.json"
DEFAULT_EMBEDDING_DIRECTORY = Path("data/embeddings")
ACTIVE_EMBEDDINGS_FILENAME = "active_embeddings.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_PROFILE = f"{MODEL_NAME}:normalized-float32-v1"
BATCH_SIZE = 32
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,199}$")


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    text: str
    source_id: str
    scheme_id: str
    content_hash: str
    chunk_file: str


@dataclass(frozen=True)
class EmbeddingStats:
    updated: bool
    documents_embedded: int
    chunks_embedded: int
    unchanged_documents: int
    dimensions: int | None
    active_manifest: Path | None


class EmbeddingModel(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        ...


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON file {path}: {exc}") from exc


def _resolve_chunk_file(chunks_directory: Path, relative_path: object) -> tuple[Path, str]:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("Chunk manifest file paths must be non-empty strings.")
    if "\\" in relative_path:
        raise ValueError(f"Chunk manifest path must use POSIX separators: {relative_path!r}")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Unsafe chunk manifest path: {relative_path!r}")
    root = chunks_directory.resolve(strict=True)
    candidate = (chunks_directory / Path(*relative.parts)).resolve(strict=True)
    if candidate.parent != root and root not in candidate.parents:
        raise ValueError(f"Chunk file escapes the chunk directory: {relative_path!r}")
    if not candidate.is_file():
        raise ValueError(f"Chunk file does not exist: {relative_path!r}")
    return candidate, relative.as_posix()


def load_active_chunks(
    chunk_manifest: Path = DEFAULT_CHUNK_MANIFEST,
) -> dict[str, list[ChunkRecord]]:
    """Load and validate only chunk files referenced by active_chunks.json."""
    manifest = _read_json(chunk_manifest)
    if not isinstance(manifest, dict):
        raise ValueError(f"Chunk manifest must contain a JSON object: {chunk_manifest}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Chunk manifest contains no active files.")
    expected_documents = manifest.get("documents")
    expected_chunks = manifest.get("chunks")
    if not isinstance(expected_documents, int) or expected_documents != len(files):
        raise ValueError("Chunk manifest document count does not match its files.")
    if not isinstance(expected_chunks, int) or expected_chunks < 1:
        raise ValueError("Chunk manifest has an invalid active chunk count.")

    chunks_directory = chunk_manifest.parent
    grouped: dict[str, list[ChunkRecord]] = {}
    seen_sources: set[str] = set()
    seen_chunk_ids: set[str] = set()
    total_chunks = 0
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Each chunk manifest file entry must be an object.")
        source_id = item.get("source_id")
        scheme_id = item.get("scheme_id")
        content_hash = item.get("content_hash")
        expected_count = item.get("chunk_count")
        if not isinstance(source_id, str) or not _SAFE_ID.fullmatch(source_id):
            raise ValueError(f"Invalid source ID in chunk manifest: {source_id!r}")
        if source_id in seen_sources:
            raise ValueError(f"Duplicate source ID in chunk manifest: {source_id!r}")
        seen_sources.add(source_id)
        if not isinstance(scheme_id, str) or scheme_id not in SCHEME_IDS:
            raise ValueError(f"Unsupported scheme ID for {source_id!r}: {scheme_id!r}")
        if not isinstance(content_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", content_hash):
            raise ValueError(f"Invalid content hash for {source_id!r}.")
        if not isinstance(expected_count, int) or expected_count < 1:
            raise ValueError(f"Invalid chunk count for {source_id!r}.")

        chunk_path, relative_chunk_path = _resolve_chunk_file(
            chunks_directory, item.get("path")
        )
        try:
            lines = chunk_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"Cannot read chunk file {chunk_path}: {exc}") from exc
        if len(lines) != expected_count:
            raise ValueError(
                f"Chunk count mismatch for {source_id!r}: expected {expected_count}, "
                f"found {len(lines)}."
            )

        records: list[ChunkRecord] = []
        for line_number, line in enumerate(lines, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {chunk_path}, line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Invalid chunk record at {chunk_path}:{line_number}.")
            chunk_id = row.get("chunk_id")
            text = row.get("text")
            metadata = row.get("metadata")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(f"Missing chunk ID at {chunk_path}:{line_number}.")
            if chunk_id in seen_chunk_ids:
                raise ValueError(f"Duplicate active chunk ID: {chunk_id!r}")
            seen_chunk_ids.add(chunk_id)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty chunk text at {chunk_path}:{line_number}.")
            if not isinstance(metadata, dict):
                raise ValueError(f"Missing chunk metadata at {chunk_path}:{line_number}.")
            if (
                metadata.get("source_id") != source_id
                or metadata.get("scheme_id") != scheme_id
                or metadata.get("content_hash") != content_hash
            ):
                raise ValueError(
                    f"Chunk provenance mismatch at {chunk_path}:{line_number}."
                )
            records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    text=text,
                    source_id=source_id,
                    scheme_id=scheme_id,
                    content_hash=content_hash,
                    chunk_file=relative_chunk_path,
                )
            )
        grouped[source_id] = records
        total_chunks += len(records)

    if total_chunks != expected_chunks:
        raise ValueError(
            f"Chunk manifest total mismatch: expected {expected_chunks}, found {total_chunks}."
        )
    return grouped


def _safe_storage_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")
    if not safe:
        raise ValueError("Cannot create storage name from an empty identifier.")
    return safe


def _chunk_id_digest(records: list[ChunkRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.chunk_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_active_embedding_manifest(output_directory: Path) -> dict[str, object] | None:
    path = output_directory / ACTIVE_EMBEDDINGS_FILENAME
    if not path.exists():
        return None
    manifest = _read_json(path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Embedding manifest must contain a JSON object: {path}")
    documents = manifest.get("documents")
    if not isinstance(documents, list):
        raise ValueError(f"Embedding manifest has no documents list: {path}")
    seen_sources: set[str] = set()
    for item in documents:
        if not isinstance(item, dict):
            raise ValueError(f"Embedding manifest has an invalid document entry: {path}")
        source_id = item.get("source_id")
        if not isinstance(source_id, str) or source_id in seen_sources:
            raise ValueError(f"Embedding manifest has a missing or duplicate source ID: {path}")
        seen_sources.add(source_id)
    return manifest


def _load_embedding_file(
    path: Path,
    expected_chunk_ids: list[str],
    expected_dimensions: int | None,
) -> tuple[np.ndarray, int] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            chunk_ids = data["chunk_ids"]
            vectors = data["embeddings"]
    except (OSError, ValueError, KeyError):
        return None
    if chunk_ids.ndim != 1 or vectors.ndim != 2:
        return None
    if chunk_ids.tolist() != expected_chunk_ids or vectors.shape[0] != len(expected_chunk_ids):
        return None
    dimensions = int(vectors.shape[1])
    if dimensions < 1 or (
        expected_dimensions is not None and dimensions != expected_dimensions
    ):
        return None
    if not np.isfinite(vectors).all():
        return None
    norms = np.linalg.norm(vectors, axis=1)
    if len(norms) and not np.allclose(norms, 1.0, atol=1e-3):
        return None
    return vectors, dimensions


def _load_model() -> EmbeddingModel:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required to generate chunk embeddings."
        ) from exc
    return SentenceTransformer(MODEL_NAME)


def _embed_records(
    records: list[ChunkRecord],
    model: EmbeddingModel,
    batch_size: int,
    expected_dimensions: int | None,
) -> tuple[np.ndarray, int]:
    batches: list[np.ndarray] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        encoded = np.asarray(
            model.encode(
                [record.text for record in batch],
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )
        if encoded.ndim != 2 or encoded.shape[0] != len(batch):
            raise ValueError("Embedding model returned an unexpected batch shape.")
        if encoded.shape[1] < 1 or (
            expected_dimensions is not None
            and encoded.shape[1] != expected_dimensions
        ):
            raise ValueError("Embedding model returned an unexpected vector dimension.")
        if not np.isfinite(encoded).all():
            raise ValueError("Embedding model returned NaN or infinite values.")
        norms = np.linalg.norm(encoded, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("Embedding model returned non-normalized vectors.")
        batches.append(encoded)
        expected_dimensions = int(encoded.shape[1])
    if not batches:
        raise ValueError("No chunk text was supplied for embedding.")
    return np.concatenate(batches, axis=0), int(batches[0].shape[1])


def _write_embedding_file(
    path: Path,
    chunk_ids: list[str],
    vectors: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            np.savez_compressed(
                file,
                chunk_ids=np.asarray(chunk_ids, dtype=np.str_),
                embeddings=np.asarray(vectors, dtype=np.float32),
            )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except OSError:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
        raise


def _publish_manifest(output_directory: Path, manifest: dict[str, object]) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    active_path = output_directory / ACTIVE_EMBEDDINGS_FILENAME
    temporary_path: Path | None = None
    content = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_directory,
            prefix=".active_embeddings.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, active_path)
    except OSError:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
        raise
    return active_path


def embed_active_chunks(
    chunk_manifest: Path = DEFAULT_CHUNK_MANIFEST,
    output_directory: Path = DEFAULT_EMBEDDING_DIRECTORY,
    *,
    model: EmbeddingModel | None = None,
    batch_size: int = BATCH_SIZE,
) -> EmbeddingStats:
    """Embed changed active chunk files and atomically publish the aligned vectors."""
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero.")
    grouped_chunks = load_active_chunks(chunk_manifest)
    previous_manifest = _read_active_embedding_manifest(output_directory)
    previous_documents = {}
    expected_dimensions: int | None = None
    if previous_manifest:
        if previous_manifest.get("model_profile") != MODEL_PROFILE:
            previous_documents = {}
        else:
            previous_documents = {
                item.get("source_id"): item
                for item in previous_manifest["documents"]
                if isinstance(item, dict) and isinstance(item.get("source_id"), str)
            }
            dimensions = previous_manifest.get("dimensions")
            if isinstance(dimensions, int) and dimensions > 0:
                expected_dimensions = dimensions

    prepared: list[tuple[str, list[ChunkRecord], Path, dict[str, object] | None]] = []
    unchanged_count = 0
    for source_id, records in sorted(grouped_chunks.items()):
        scheme_id = records[0].scheme_id
        chunk_hash = records[0].content_hash
        if any(
            record.scheme_id != scheme_id or record.content_hash != chunk_hash
            for record in records
        ):
            raise ValueError(f"Inconsistent source metadata for {source_id!r}.")
        chunk_ids = [record.chunk_id for record in records]
        chunk_digest = _chunk_id_digest(records)
        storage_path = (
            Path("files")
            / _safe_storage_id(scheme_id)
            / (
                f"{_safe_storage_id(source_id)}-{chunk_hash[:16]}-"
                f"{chunk_digest[:16]}-{hashlib.sha256(MODEL_PROFILE.encode()).hexdigest()[:10]}.npz"
            )
        )
        destination = output_directory / storage_path
        previous = previous_documents.get(source_id)
        cache_valid = False
        if isinstance(previous, dict):
            cache_valid = (
                previous.get("scheme_id") == scheme_id
                and previous.get("content_hash") == chunk_hash
                and previous.get("chunk_id_digest") == chunk_digest
                and previous.get("chunk_file") == records[0].chunk_file
                and previous.get("chunk_count") == len(records)
                and previous.get("model_profile") == MODEL_PROFILE
            )
            if cache_valid:
                previous_path_value = previous.get("path")
                if not isinstance(previous_path_value, str):
                    cache_valid = False
                else:
                    previous_relative = PurePosixPath(previous_path_value)
                    if previous_relative.is_absolute() or any(
                        part in {"", ".", ".."} for part in previous_relative.parts
                    ):
                        cache_valid = False
                    else:
                        previous_file = output_directory.joinpath(
                            *previous_relative.parts
                        )
                        if previous_file != destination:
                            cache_valid = False
                        else:
                            loaded = _load_embedding_file(
                                destination, chunk_ids, expected_dimensions
                            )
                            if loaded is None:
                                cache_valid = False
                            else:
                                if expected_dimensions is not None and loaded[1] != expected_dimensions:
                                    cache_valid = False
                                else:
                                    expected_dimensions = loaded[1]
                                    cache_valid = (
                                        previous.get("file_sha256")
                                        == _file_sha256(destination)
                                    )
        if cache_valid:
            unchanged_count += 1
            prepared.append((source_id, records, destination, previous))
        else:
            if destination.exists():
                destination = destination.with_name(
                    f"{destination.stem}-{uuid.uuid4().hex[:12]}{destination.suffix}"
                )
            prepared.append((source_id, records, destination, None))

    changed = [item for item in prepared if item[3] is None]
    if (
        not changed
        and previous_manifest
        and set(previous_documents) == set(grouped_chunks)
        and previous_manifest.get("model_profile") == MODEL_PROFILE
        and previous_manifest.get("dimensions") == expected_dimensions
        and previous_manifest.get("chunks")
        == sum(len(records) for records in grouped_chunks.values())
    ):
        return EmbeddingStats(
            updated=False,
            documents_embedded=0,
            chunks_embedded=0,
            unchanged_documents=unchanged_count,
            dimensions=expected_dimensions,
            active_manifest=output_directory / ACTIVE_EMBEDDINGS_FILENAME,
        )

    if changed:
        model = model or _load_model()

    manifest_documents: list[dict[str, object]] = []
    dimensions = expected_dimensions
    chunks_embedded = 0
    for source_id, records, destination, cached in prepared:
        if cached is None:
            if model is None:
                raise RuntimeError("Embedding model was not initialized.")
            vectors, dimensions = _embed_records(
                records,
                model,
                batch_size,
                expected_dimensions=dimensions,
            )
            _write_embedding_file(
                destination,
                [record.chunk_id for record in records],
                vectors,
            )
            chunks_embedded += len(records)
            file_hash = _file_sha256(destination)
        else:
            file_hash_value = cached.get("file_sha256")
            if not isinstance(file_hash_value, str):
                raise ValueError(f"Cached embedding metadata is invalid for {source_id!r}.")
            file_hash = file_hash_value

        first_record = records[0]
        manifest_documents.append(
            {
                "source_id": source_id,
                "scheme_id": first_record.scheme_id,
                "content_hash": first_record.content_hash,
                "chunk_id_digest": _chunk_id_digest(records),
                "chunk_file": first_record.chunk_file,
                "chunk_count": len(records),
                "path": destination.relative_to(output_directory).as_posix(),
                "file_sha256": file_hash,
                "model_profile": MODEL_PROFILE,
                "dimensions": dimensions,
            }
        )

    if dimensions is None:
        raise ValueError("No embedding dimension could be determined.")
    manifest = {
        "model_name": MODEL_NAME,
        "model_profile": MODEL_PROFILE,
        "normalized": True,
        "dtype": "float32",
        "dimensions": dimensions,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "documents": manifest_documents,
        "chunks": sum(int(item["chunk_count"]) for item in manifest_documents),
    }
    active_path = _publish_manifest(output_directory, manifest)
    return EmbeddingStats(
        updated=True,
        documents_embedded=len(changed),
        chunks_embedded=chunks_embedded,
        unchanged_documents=unchanged_count,
        dimensions=dimensions,
        active_manifest=active_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"Embed active chunks with {MODEL_NAME}; does not write to Qdrant."
    )
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNK_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_EMBEDDING_DIRECTORY)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args(argv)
    try:
        stats = embed_active_chunks(
            args.chunks,
            args.output,
            batch_size=args.batch_size,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Embedding failed; previous active embeddings were preserved: {exc}", file=sys.stderr)
        return 1

    print(f"Embedding model: {MODEL_NAME}")
    print(f"Updated: {stats.updated}")
    print(f"Documents embedded: {stats.documents_embedded}")
    print(f"Chunks embedded: {stats.chunks_embedded}")
    print(f"Unchanged documents: {stats.unchanged_documents}")
    print(f"Vector dimensions: {stats.dimensions}")
    print(f"Active manifest: {stats.active_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
