"""Load downloaded official HTML and PDF documents with provenance."""

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pymupdf
from bs4 import BeautifulSoup

from kuvera_lens.schemes import SCHEME_BY_ID
from kuvera_lens.sources import ApprovedSource, validate_source

DEFAULT_INPUT_DIRECTORY = Path("data/raw")
SUPPORTED_CONTENT_TYPES = {
    "application/pdf": ".pdf",
    "text/html": ".html",
}
_RISKOMETER_LEVEL = re.compile(
    r"\briskometer\b\s*(?:[:#-]\s*)?\b(?:very\s+high|moderately\s+high|high|moderate|low)\b",
    re.IGNORECASE,
)
_TER_VALUE = re.compile(r"\bTER\b\s*\d+(?:\.\d+)?%?(?![\d.])", re.IGNORECASE)


@dataclass(frozen=True)
class ExtractedText:
    text: str
    page_number: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class LoadedDocument:
    source_id: str
    scheme_id: str
    scheme_name: str
    organization: str
    url: str
    source_domain: str
    content_type: str
    document_title: str
    content_hash: str
    local_path: Path
    content: tuple[ExtractedText, ...]
    document_type: str
    publication_date: str | None
    last_checked_at: str | None


class DocumentLoadError(ValueError):
    """Raised when a downloaded document or its provenance cannot be trusted."""


def has_scheme_page_key_facts(text: str) -> bool:
    """Return whether a scheme-page capture includes a TER and riskometer value."""
    normalized = re.sub(r"\s+", " ", text)
    return bool(_TER_VALUE.search(normalized) and _RISKOMETER_LEVEL.search(normalized))


def _is_hdfc_scheme_page(metadata: dict[str, object]) -> bool:
    source_url = metadata.get("url")
    if not isinstance(source_url, str):
        return False
    try:
        parsed = urlsplit(source_url)
    except ValueError:
        return False
    return (
        metadata.get("content_type") == "text/html"
        and metadata.get("document_type") == "scheme_page"
        and metadata.get("organization") == "hdfc_amc"
        and parsed.hostname == "www.hdfcfund.com"
        and parsed.path.startswith("/explore/mutual-funds/")
        and parsed.path.rstrip("/").endswith("/direct")
    )


def _required_string(metadata: dict[str, object], key: str, manifest: Path) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DocumentLoadError(f"{manifest}: missing or invalid {key!r}.")
    return value


def _read_manifest(manifest: Path, input_directory: Path) -> tuple[dict[str, object], Path]:
    try:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DocumentLoadError(
            f"Document loading failed: cannot read source metadata {manifest}: {exc}"
        ) from exc

    if not isinstance(metadata, dict):
        raise DocumentLoadError(f"{manifest}: metadata must be a JSON object.")

    source_id = _required_string(metadata, "source_id", manifest)
    scheme_id = _required_string(metadata, "scheme_id", manifest)
    organization = _required_string(metadata, "organization", manifest)
    url = _required_string(metadata, "url", manifest)
    content_type = _required_string(metadata, "content_type", manifest)
    expected_hash = _required_string(metadata, "sha256", manifest)
    filename = _required_string(metadata, "downloaded_file", manifest)

    try:
        validate_source(ApprovedSource(source_id, scheme_id, organization, url))
    except ValueError as exc:
        raise DocumentLoadError(f"{manifest}: invalid source provenance: {exc}") from exc

    extension = SUPPORTED_CONTENT_TYPES.get(content_type)
    if extension is None:
        raise DocumentLoadError(
            f"{manifest}: unsupported content type {content_type!r}."
        )
    if Path(filename).name != filename or not filename.endswith(extension):
        raise DocumentLoadError(
            f"{manifest}: downloaded_file must be a local {extension} filename."
        )
    if manifest.stem != Path(filename).stem:
        raise DocumentLoadError(
            f"{manifest}: metadata filename does not match the manifest name."
        )
    if len(expected_hash) != 64 or any(
        character not in "0123456789abcdef" for character in expected_hash
    ):
        raise DocumentLoadError(f"{manifest}: invalid SHA-256 content hash.")

    content_path = input_directory / filename
    try:
        resolved_directory = input_directory.resolve(strict=True)
        resolved_content = content_path.resolve(strict=True)
    except OSError as exc:
        raise DocumentLoadError(f"{manifest}: downloaded file is missing: {exc}") from exc
    if resolved_content.parent != resolved_directory or not resolved_content.is_file():
        raise DocumentLoadError(f"{manifest}: downloaded file must be inside the input directory.")

    return metadata, resolved_content


def _extract_html(raw: bytes) -> tuple[str, tuple[str, ...]]:
    soup = BeautifulSoup(raw, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for element in soup(["script", "style", "noscript", "template"]):
        element.decompose()

    body = soup.body or soup
    text = body.get_text(" ", strip=True)
    if not text:
        raise DocumentLoadError("HTML document contains no extractable text.")
    headings = tuple(
        heading.get_text(" ", strip=True)
        for heading in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        if heading.get_text(" ", strip=True)
    )
    return text, headings or ((title,) if title else ())


def _load_html(raw: bytes, metadata: dict[str, object]) -> tuple[str, tuple[ExtractedText, ...]]:
    text, headings = _extract_html(raw)

    title = str(metadata.get("document_title") or "")
    if not title:
        try:
            title = BeautifulSoup(raw, "html.parser").title.get_text(
                " ", strip=True
            )
        except AttributeError:
            title = ""
    if not title:
        title = urlsplit(str(metadata["url"])).path.rstrip("/").split("/")[-1]
    title = title or str(metadata["source_id"])

    section = " | ".join(headings) if headings else None
    return title, (ExtractedText(text=text, section=section),)


def _load_pdf(raw: bytes) -> tuple[str, tuple[ExtractedText, ...]]:
    if not raw.startswith(b"%PDF-"):
        raise DocumentLoadError("PDF signature is missing.")
    try:
        with pymupdf.open(stream=raw, filetype="pdf") as document:
            if document.is_closed or document.page_count == 0:
                raise DocumentLoadError("PDF has no pages.")
            title = str(document.metadata.get("title") or "").strip()
            pages_list: list[ExtractedText] = []
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages_list.append(
                        ExtractedText(text=text, page_number=page_number)
                    )
            pages = tuple(pages_list)
    except DocumentLoadError:
        raise
    except (
        pymupdf.FileDataError,
        pymupdf.EmptyFileError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise DocumentLoadError(f"Could not open or extract PDF: {exc}") from exc

    if not pages:
        raise DocumentLoadError("PDF contains no extractable text.")
    return title, pages


def load_document(manifest: Path, input_directory: Path) -> LoadedDocument:
    """Load one scraper manifest and extract HTML text or PDF page text."""
    metadata, content_path = _read_manifest(manifest, input_directory)
    try:
        raw = content_path.read_bytes()
    except OSError as exc:
        raise DocumentLoadError(f"Cannot read downloaded file {content_path}: {exc}") from exc
    if not raw:
        raise DocumentLoadError(f"Downloaded document is empty: {content_path}")

    actual_hash = hashlib.sha256(raw).hexdigest()
    expected_hash = str(metadata["sha256"])
    if actual_hash != expected_hash:
        raise DocumentLoadError(
            f"Content hash mismatch for {content_path}; refusing to load modified content."
        )

    content_type = str(metadata["content_type"])
    if content_type == "text/html":
        title, extracted = _load_html(raw, metadata)
    else:
        title, extracted = _load_pdf(raw)
        title = str(metadata.get("document_title") or title or metadata["source_id"])

    source_url = str(metadata["url"])
    combined_text = "\n".join(item.text for item in extracted)
    publication_date = metadata.get("publication_date")
    if not isinstance(publication_date, str) or not publication_date.strip():
        date_patterns = (
            r"\bdated\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})",
            r"\b(?:published|issued|updated|last updated)(?:\s+on)?\s+"
            r"([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})",
            r"\b(?:as of|as on|effective from)\s+"
            r"([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})",
        )
        publication_date = None
        for pattern in date_patterns:
            date_match = re.search(pattern, combined_text, re.IGNORECASE)
            if date_match:
                publication_date = date_match.group(1).strip()
                break
    last_checked_at = metadata.get("last_checked_at")
    if isinstance(last_checked_at, str):
        try:
            datetime.fromisoformat(last_checked_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise DocumentLoadError(
                f"{manifest}: invalid last_checked_at timestamp."
            ) from exc
    else:
        last_checked_at = None
    document_type = metadata.get("document_type")
    identifying_text = unquote(f"{title} {source_url}")
    if content_type == "application/pdf" and re.search(
        r"\b(?:key information memorandum|key information document|KIM)\b",
        identifying_text,
        re.IGNORECASE,
    ):
        document_type = "KIM"
    elif not isinstance(document_type, str) or not document_type:
        document_type = (
            "SID"
            if re.search(r"\b(?:scheme information document|SID)\b", identifying_text, re.I)
            else "scheme_page" if content_type == "text/html" else "PDF"
        )
    return LoadedDocument(
        source_id=str(metadata["source_id"]),
        scheme_id=str(metadata["scheme_id"]),
        scheme_name=SCHEME_BY_ID[str(metadata["scheme_id"])].display_name,
        organization=str(metadata["organization"]),
        url=source_url,
        source_domain=urlsplit(source_url).hostname or "",
        content_type=content_type,
        document_title=title,
        content_hash=actual_hash,
        local_path=content_path,
        content=extracted,
        document_type=document_type,
        publication_date=publication_date,
        last_checked_at=last_checked_at,
    )


def _latest_manifests(input_directory: Path) -> list[Path]:
    try:
        manifests = sorted(input_directory.glob("*.json"))
    except OSError as exc:
        raise DocumentLoadError(f"Cannot inspect input directory {input_directory}: {exc}") from exc
    if not manifests:
        raise DocumentLoadError(f"No scraper metadata files found in {input_directory}.")

    manifests_by_source: dict[str, list[Path]] = {}
    for manifest in manifests:
        try:
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DocumentLoadError(
                f"Document loading failed: cannot read source metadata {manifest}: {exc}"
            ) from exc
        if not isinstance(metadata, dict):
            raise DocumentLoadError(f"{manifest}: metadata must be a JSON object.")
        source_id = metadata.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise DocumentLoadError(f"{manifest}: missing or invalid 'source_id'.")
        manifests_by_source.setdefault(source_id, []).append(manifest)

    latest_manifests: list[Path] = []
    for source_manifests in manifests_by_source.values():
        def sort_key(path: Path) -> tuple[str, float]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            checked = data.get("last_checked_at", "") if isinstance(data, dict) else ""
            return (checked if isinstance(checked, str) else "", path.stat().st_mtime)

        ordered_manifests = sorted(source_manifests, key=sort_key, reverse=True)
        selected_manifest: Path | None = None
        is_hdfc_scheme_page = False
        for manifest in ordered_manifests:
            metadata, _ = _read_manifest(manifest, input_directory)
            if _is_hdfc_scheme_page(metadata):
                document = load_document(manifest, input_directory)
                if not has_scheme_page_key_facts(
                    "\n".join(item.text for item in document.content)
                ):
                    is_hdfc_scheme_page = True
                    continue
            selected_manifest = manifest
            break
        if selected_manifest is None:
            if is_hdfc_scheme_page:
                raise DocumentLoadError(
                    "No complete captured HDFC scheme webpage contains both TER "
                    "and riskometer data."
                )
            raise DocumentLoadError("No usable source manifest was found.")
        latest_manifests.append(selected_manifest)
    return latest_manifests


def load_documents(input_directory: Path = DEFAULT_INPUT_DIRECTORY) -> list[LoadedDocument]:
    """Load the newest manifest for every source; fail explicitly on invalid documents."""
    latest_manifests = _latest_manifests(input_directory)
    loaded: list[LoadedDocument] = []
    for manifest in latest_manifests:
        try:
            loaded.append(load_document(manifest, input_directory))
        except DocumentLoadError as exc:
            raise DocumentLoadError(f"Document loading failed: {exc}") from exc
    return loaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Load raw documents downloaded from explicitly approved URLs."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIRECTORY)
    args = parser.parse_args(argv)

    try:
        documents = load_documents(args.input)
    except (OSError, DocumentLoadError) as exc:
        print(f"Document loading failed: {exc}", file=sys.stderr)
        return 1

    for document in documents:
        page_count = sum(item.page_number is not None for item in document.content)
        print(
            f"Loaded {document.source_id} ({document.scheme_id}): "
            f"{len(document.content)} text unit(s), {page_count} PDF page(s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
