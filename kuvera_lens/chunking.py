"""Split verified loaded documents into deterministic, citation-aware chunks."""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias
from urllib.parse import urlsplit

from kuvera_lens.loaders import DEFAULT_INPUT_DIRECTORY, LoadedDocument, load_documents
from kuvera_lens.sources import ApprovedSource, validate_source

DEFAULT_OUTPUT_DIRECTORY = Path("data/chunks")
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 160
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z“"(\[])')
JsonValue: TypeAlias = str | int


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: dict[str, JsonValue]


def _validate_document(document: LoadedDocument) -> None:
    try:
        validate_source(
            ApprovedSource(
                source_id=document.source_id,
                scheme_id=document.scheme_id,
                organization=document.organization,
                url=document.url,
            )
        )
    except ValueError as exc:
        raise ValueError(
            f"Document {document.source_id!r} has invalid source provenance: {exc}"
        ) from exc
    if not re.fullmatch(r"[0-9a-f]{64}", document.content_hash):
        raise ValueError(f"Document {document.source_id!r} has an invalid content hash.")
    if not document.document_title.strip():
        raise ValueError(f"Document {document.source_id!r} has no document title.")


def _split_oversized_segment(segment: str, max_chars: int) -> list[str]:
    words = segment.split()
    result: list[str] = []
    current: list[str] = []
    current_length = 0
    for word in words:
        if len(word) > max_chars:
            if current:
                result.append(" ".join(current))
                current = []
                current_length = 0
            result.extend(
                word[start : start + max_chars]
                for start in range(0, len(word), max_chars)
            )
            continue
        added_length = len(word) + (1 if current else 0)
        if current and current_length + added_length > max_chars:
            result.append(" ".join(current))
            current = [word]
            current_length = len(word)
        else:
            current.append(word)
            current_length += added_length
    if current:
        result.append(" ".join(current))
    return result


def _segments(text: str, max_chars: int) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    sentence_segments = [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY.split(normalized)
        if sentence.strip()
    ]
    result: list[str] = []
    for segment in sentence_segments:
        if len(segment) <= max_chars:
            result.append(segment)
        else:
            result.extend(_split_oversized_segment(segment, max_chars))
    return result


def _pack_segments(
    segments: list[str],
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    chunks: list[str] = []
    start = 0
    prefix = ""
    while start < len(segments):
        parts = [prefix] if prefix else []
        current_length = len(prefix)
        end = start
        while end < len(segments):
            added_length = len(segments[end]) + (1 if parts else 0)
            if parts and current_length + added_length > max_chars:
                break
            if not parts and added_length > max_chars:
                raise ValueError("Internal chunking error: segment exceeds chunk size.")
            parts.append(segments[end])
            current_length += added_length
            end += 1

        if end == start and prefix:
            available_prefix = max(0, max_chars - len(segments[start]) - 1)
            prefix = prefix[-available_prefix:].lstrip() if available_prefix else ""
            continue

        chunk = " ".join(parts).strip()
        if chunk:
            chunks.append(chunk)
        start = end
        prefix = chunk[-overlap_chars:].lstrip() if start < len(segments) and overlap_chars else ""
    return chunks


def split_text(
    text: str,
    max_chars: int = DEFAULT_CHUNK_SIZE,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text at sentence boundaries, falling back to words for long sentences."""
    if max_chars < 1:
        raise ValueError("max_chars must be greater than zero.")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than max_chars.")
    segments = _segments(text, max_chars)
    return _pack_segments(segments, max_chars, overlap_chars)


def _chunk_id(
    document: LoadedDocument,
    page_number: int | None,
    section: str | None,
    index: int,
    text: str,
) -> str:
    identity = "\0".join(
        (
            document.scheme_id,
            document.source_id,
            document.content_hash,
            str(page_number or 0),
            section or "",
            str(index),
            text,
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def chunk_document(
    document: LoadedDocument,
    max_chars: int = DEFAULT_CHUNK_SIZE,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Chunk a validated loaded document while retaining source/page metadata."""
    _validate_document(document)
    result: list[DocumentChunk] = []
    for extracted in document.content:
        for index, text in enumerate(
            split_text(extracted.text, max_chars, overlap_chars)
        ):
            metadata: dict[str, JsonValue] = {
                "source_id": document.source_id,
                "scheme_id": document.scheme_id,
                "scheme_name": document.scheme_name,
                "source_url": document.url,
                "source_domain": document.source_domain,
                "document_title": document.document_title,
                "document_type": document.document_type,
                "content_hash": document.content_hash,
                "document_version": document.content_hash,
                "chunk_index": index,
            }
            if document.publication_date:
                metadata["publication_date"] = document.publication_date
            if document.last_checked_at:
                metadata["last_checked_at"] = document.last_checked_at
            if extracted.page_number is not None:
                metadata["page_number"] = extracted.page_number
            if extracted.section:
                metadata["section"] = extracted.section
            result.append(
                DocumentChunk(
                    chunk_id=_chunk_id(
                        document,
                        extracted.page_number,
                        extracted.section,
                        index,
                        text,
                    ),
                    text=text,
                    metadata=metadata,
                )
            )
    return result


def chunk_documents(
    documents: list[LoadedDocument],
    max_chars: int = DEFAULT_CHUNK_SIZE,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    documents_by_source: dict[tuple[str, str], LoadedDocument] = {}
    hashes_by_source: dict[tuple[str, str], str] = {}
    documents_by_content: dict[tuple[str, str], LoadedDocument] = {}
    for document in documents:
        _validate_document(document)
        identity = (document.scheme_id, document.source_id)
        previous_hash = hashes_by_source.get(identity)
        if previous_hash is not None and previous_hash != document.content_hash:
            raise ValueError(
                f"Conflicting loaded document versions for scheme {document.scheme_id!r}: "
                f"{document.source_id!r}"
            )
        hashes_by_source[identity] = document.content_hash
        existing_source = documents_by_source.get(identity)
        if existing_source is None or _document_preference(document) > _document_preference(
            existing_source
        ):
            documents_by_source[identity] = document

    for document in documents_by_source.values():
        content_key = (document.scheme_id, document.content_hash)
        existing_content = documents_by_content.get(content_key)
        if existing_content is None or _document_preference(document) > _document_preference(
            existing_content
        ):
            documents_by_content[content_key] = document

    for document in documents_by_content.values():
        chunks.extend(chunk_document(document, max_chars, overlap_chars))
    return chunks


def _document_preference(document: LoadedDocument) -> tuple[int, str, str]:
    is_direct_pdf = int(urlsplit(document.url).path.lower().endswith(".pdf"))
    last_checked = document.last_checked_at or ""
    return is_direct_pdf, last_checked, document.source_id


def _safe_output_name(source_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", source_id).strip("-")
    if not safe:
        raise ValueError("Cannot create output name from empty source ID.")
    return safe


def write_chunks(
    documents: list[LoadedDocument],
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    max_chars: int = DEFAULT_CHUNK_SIZE,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, int]:
    """Write versioned source chunks and atomically publish their active manifest."""
    output_directory.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[DocumentChunk]] = {}
    for chunk in chunk_documents(documents, max_chars, overlap_chars):
        source_id = str(chunk.metadata["source_id"])
        grouped.setdefault(source_id, []).append(chunk)

    config_hash = hashlib.sha256(
        f"{max_chars}:{overlap_chars}".encode("ascii")
    ).hexdigest()[:12]
    counts: dict[str, int] = {}
    active_files: list[dict[str, JsonValue]] = []
    for source_id, chunks in grouped.items():
        scheme_id = str(chunks[0].metadata["scheme_id"])
        content_hash = str(chunks[0].metadata["content_hash"])
        target_directory = output_directory / _safe_output_name(scheme_id)
        target_directory.mkdir(parents=True, exist_ok=True)
        output_path = target_directory / (
            f"{_safe_output_name(source_id)}-{content_hash}-c{config_hash}.jsonl"
        )
        content = "".join(
            json.dumps(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for chunk in chunks
        )
        if output_path.exists() and output_path.read_text(encoding="utf-8") == content:
            counts[source_id] = len(chunks)
        else:
            temp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="\n",
                    dir=target_directory,
                    prefix=f".{output_path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temp_file:
                    temp_file.write(content)
                    temp_path = Path(temp_file.name)
                os.replace(temp_path, output_path)
            except OSError:
                if temp_path and temp_path.exists():
                    temp_path.unlink()
                raise
            counts[source_id] = len(chunks)
        active_files.append(
            {
                "source_id": source_id,
                "scheme_id": scheme_id,
                "content_hash": content_hash,
                "path": output_path.relative_to(output_directory).as_posix(),
                "chunk_count": len(chunks),
            }
        )

    manifest_path = output_directory / "active_chunks.json"
    manifest_content = json.dumps(
        {
            "documents": len(active_files),
            "chunks": sum(counts.values()),
            "chunk_size": max_chars,
            "overlap_chars": overlap_chars,
            "files": sorted(active_files, key=lambda item: str(item["source_id"])),
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if not manifest_path.exists() or manifest_path.read_text(encoding="utf-8") != manifest_content:
        temp_manifest: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=output_directory,
                prefix=".active_chunks.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_file.write(manifest_content)
                temp_manifest = Path(temp_file.name)
            os.replace(temp_manifest, manifest_path)
        except OSError:
            if temp_manifest and temp_manifest.exists():
                temp_manifest.unlink()
            raise
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Split loaded official-source documents into citation-aware chunks."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIRECTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap-chars", type=int, default=DEFAULT_CHUNK_OVERLAP)
    args = parser.parse_args(argv)
    try:
        documents = load_documents(args.input)
        counts = write_chunks(
            documents,
            args.output,
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
        )
    except (OSError, ValueError) as exc:
        print(f"Chunking failed: {exc}", file=sys.stderr)
        return 1

    print(f"Documents chunked: {len(counts)}")
    print(f"Chunks written: {sum(counts.values())}")
    for source_id, count in sorted(counts.items()):
        print(f"{source_id}: {count} chunks")
    print(f"Output directory: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
