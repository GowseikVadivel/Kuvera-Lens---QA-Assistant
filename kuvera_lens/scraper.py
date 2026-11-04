"""Playwright-based HDFC scheme-page scraping and official PDF discovery."""

import argparse
import hashlib
import html
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pymupdf
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from kuvera_lens.loaders import has_scheme_page_key_facts
from kuvera_lens.schemes import SCHEME_BY_ID
from kuvera_lens.sources import ApprovedSource, OFFICIAL_HOSTS, load_sources, validate_source

DEFAULT_SOURCE_REGISTRY = Path("config/source_links.json")
DEFAULT_OUTPUT_DIRECTORY = Path("data/raw")
MAX_DOWNLOAD_BYTES = 30 * 1024 * 1024
MAX_REDIRECTS = 5
PDF_SIGNATURE = b"%PDF-"

_DATE_PATTERNS = (
    re.compile(
        r"\bdated\s+((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})",
        re.I,
    ),
    re.compile(
        r"\b(?:published|issued|updated|last updated)(?:\s+on)?\s+"
        r"((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})",
        re.I,
    ),
    re.compile(
        r"\b(?:as of|as on|effective from)\s+"
        r"((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})",
        re.I,
    ),
)


@dataclass(frozen=True)
class DiscoveredLink:
    url: str
    label: str


@dataclass(frozen=True)
class ScrapeStats:
    webpages_scraped: int
    relevant_pdfs_discovered: int
    new_documents: int
    changed_documents: int
    unchanged_documents: int
    failed_urls: tuple[str, ...]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hostname(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().rstrip(".")


def _is_allowed_url(url: str, organization: str) -> bool:
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "https" or parsed.username or parsed.password:
        return False
    return _hostname(url) in OFFICIAL_HOSTS[organization]


def _scheme_text(scheme_id: str) -> str:
    return SCHEME_BY_ID[scheme_id].display_name


def _scheme_aliases(scheme_id: str) -> tuple[str, ...]:
    return SCHEME_BY_ID[scheme_id].aliases


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", unquote(value)).strip().casefold()


def classify_document_type(*values: str) -> str | None:
    identifying_text = _normalize_text(" ".join(values[:3]))
    for text in (identifying_text, _normalize_text(" ".join(values))):
        if re.search(r"\b(key information memorandum|key information document|kim)\b", text):
            return "KIM"
        if re.search(r"\b(scheme information document|sid)\b", text):
            return "SID"
        if re.search(r"\bfactsheets?\b", text):
            return "factsheet"
    return None


def _matches_scheme(scheme_id: str, *values: str) -> bool:
    text = _normalize_text(" ".join(values))
    return any(_normalize_text(alias) in text for alias in _scheme_aliases(scheme_id))


def _publication_date(text: str) -> str | None:
    for pattern in _DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


def _content_type(response, url: str, expected: str) -> bool:
    raw = response.headers.get("content-type", response.headers.get("Content-Type", ""))
    media_type = raw.split(";", 1)[0].strip().lower()
    if expected == "text/html":
        return media_type in {"text/html", "application/xhtml+xml"}
    if expected == "application/pdf":
        return media_type == "application/pdf" or (
            media_type == "application/octet-stream"
            and urlsplit(url).path.lower().endswith(".pdf")
        )
    return False


def _check_download_response(response, requested_url: str, organization: str) -> bytes:
    if not 200 <= response.status < 300:
        raise ValueError(f"HTTP {response.status} for {requested_url}")
    if not _is_allowed_url(response.url, organization):
        raise ValueError(f"Unapproved final URL after redirect: {response.url}")
    if not _content_type(response, response.url, "application/pdf"):
        raise ValueError(
            f"Unexpected content type for PDF URL {requested_url}: "
            f"{response.headers.get('content-type', 'missing')}"
        )
    body = response.body()
    if not body:
        raise ValueError(f"Empty PDF response: {requested_url}")
    if len(body) > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"PDF exceeds the {MAX_DOWNLOAD_BYTES}-byte limit: {requested_url}")
    if not body.startswith(PDF_SIGNATURE):
        raise ValueError(f"Invalid PDF signature: {requested_url}")
    return body


def _pdf_details(body: bytes) -> tuple[str, str, str]:
    try:
        with pymupdf.open(stream=body, filetype="pdf") as document:
            if document.page_count == 0:
                raise ValueError("PDF has no pages.")
            title = str(document.metadata.get("title") or "").strip()
            opening_text = "\n".join(
                document[index].get_text("text")
                for index in range(min(document.page_count, 5))
            )
            full_text = "\n".join(page.get_text("text") for page in document)
    except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError) as exc:
        raise ValueError(f"PDF could not be parsed: {exc}") from exc
    return title, opening_text, full_text


def _stable_document_id(scheme_id: str, document_type: str, url: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return f"{scheme_id}-{document_type.lower()}-{digest}"


def _previous_hash(output_directory: Path, source_id: str) -> str | None:
    manifests = sorted(
        output_directory.glob(f"{source_id}-*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for manifest in manifests:
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and isinstance(data.get("sha256"), str):
            return data["sha256"]
    return None


def _cached_document_metadata(
    output_directory: Path,
    source_id: str,
    content_hash: str,
    scheme_id: str,
) -> dict[str, object] | None:
    manifests = sorted(
        output_directory.glob(f"{source_id}-*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for manifest in manifests:
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(data, dict)
            and data.get("sha256") == content_hash
            and data.get("scheme_id") == scheme_id
            and data.get("document_type") in {"SID", "KIM", "factsheet"}
        ):
            return data
    return None


def _cached_document_by_url(
    output_directory: Path,
    url: str,
    content_hash: str,
    scheme_id: str,
) -> dict[str, object] | None:
    for manifest in output_directory.glob("*.json"):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(data, dict)
            and data.get("sha256") == content_hash
            and data.get("scheme_id") == scheme_id
            and data.get("url") == url
            and data.get("document_type") in {"SID", "KIM", "factsheet"}
        ):
            return data
    return None


def _save_document(
    *,
    source_id: str,
    scheme_id: str,
    organization: str,
    url: str,
    content_type: str,
    body: bytes,
    output_directory: Path,
    document_title: str,
    document_type: str,
    publication_date: str | None,
    source_page_url: str | None = None,
) -> str:
    digest = hashlib.sha256(body).hexdigest()
    previous_digest = _previous_hash(output_directory, source_id)
    if previous_digest == digest:
        status = "unchanged"
    elif previous_digest is None:
        status = "new"
    else:
        status = "changed"

    extension = ".pdf" if content_type == "application/pdf" else ".html"
    output_directory.mkdir(parents=True, exist_ok=True)
    filename = f"{source_id}-{digest}{extension}"
    content_path = output_directory / filename
    manifest_path = output_directory / f"{source_id}-{digest}.json"
    if not content_path.exists():
        content_path.write_bytes(body)
    metadata = {
        "source_id": source_id,
        "scheme_id": scheme_id,
        "scheme_name": _scheme_text(scheme_id),
        "organization": organization,
        "url": url,
        "source_url": url,
        "source_page_url": source_page_url,
        "content_type": content_type,
        "document_type": document_type,
        "document_title": document_title,
        "publication_date": publication_date,
        "last_checked_at": _now(),
        "sha256": digest,
        "downloaded_file": filename,
    }
    manifest_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return status


def _page_links(page) -> list[DiscoveredLink]:
    items = page.locator("a[href]").evaluate_all(
        """elements => elements.map(a => ({
            url: a.href,
            label: (a.innerText || a.getAttribute("aria-label") || a.title || "").trim()
        }))"""
    )
    return [
        DiscoveredLink(item["url"], item["label"])
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get("url"), str)
        and isinstance(item.get("label"), str)
    ]


def _rendered_page_content(page, title: str) -> tuple[str, str]:
    main = page.locator("main")
    if main.count():
        text = main.inner_text(timeout=15_000).strip()
    else:
        text = page.locator("body").inner_text(timeout=15_000).strip()
    if len(text) < 100:
        raise ValueError(f"Page has no meaningful text: {page.url}")
    safe_title = html.escape(title)
    safe_text = html.escape(text)
    headings = page.locator("h1, h2, h3, h4, h5, h6").all_inner_texts()
    heading_markup = "".join(
        f"<h2>{html.escape(heading.strip())}</h2>"
        for heading in headings
        if heading.strip() and heading.strip().casefold() != title.casefold()
    )
    normalized_html = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{safe_title}</title></head><body><main><h1>{safe_title}</h1>"
        f"{heading_markup}<pre>{safe_text}</pre>"
        "</main></body></html>"
    )
    return text, normalized_html


def _discover_pdfs(links: list[DiscoveredLink], organization: str) -> list[DiscoveredLink]:
    unique: dict[str, DiscoveredLink] = {}
    for link in links:
        parsed = urlsplit(link.url)
        if parsed.path.lower().endswith(".pdf") and _is_allowed_url(link.url, organization):
            unique.setdefault(link.url, link)
    return list(unique.values())


def _download_pdf(
    context,
    link: DiscoveredLink,
    source: ApprovedSource,
    page_url: str,
    output_directory: Path,
) -> tuple[bytes, str, str, str | None, str] | None:
    response = context.request.get(
        link.url,
        timeout=60_000,
        max_redirects=MAX_REDIRECTS,
    )
    body = _check_download_response(response, link.url, source.organization)
    digest = hashlib.sha256(body).hexdigest()
    cached = _cached_document_by_url(
        output_directory, link.url, digest, source.scheme_id
    )
    if cached:
        cached_title = str(cached.get("document_title") or link.label or link.url)
        cached_type = str(cached["document_type"])
        filename = Path(unquote(urlsplit(link.url).path)).name
        inferred_type = classify_document_type(
            link.label, filename, cached_title, link.url
        )
        if inferred_type is None or inferred_type == cached_type:
            return (
                body,
                cached_type,
                cached_title,
                cached.get("publication_date")
                if isinstance(cached.get("publication_date"), str)
                else None,
                str(cached["source_id"]),
            )
        return (
            body,
            inferred_type,
            cached_title,
            cached.get("publication_date")
            if isinstance(cached.get("publication_date"), str)
            else None,
            _stable_document_id(source.scheme_id, inferred_type, link.url),
        )
    title, opening_text, full_text = _pdf_details(body)
    filename = Path(unquote(urlsplit(link.url).path)).name
    document_type = classify_document_type(link.label, filename, title, opening_text)
    if document_type is None:
        return None
    if not _matches_scheme(source.scheme_id, link.label, filename, title, opening_text):
        return None
    return (
        body,
        document_type,
        title or link.label or filename,
        _publication_date(full_text),
        _stable_document_id(source.scheme_id, document_type, link.url),
    )


def scrape_source_pages(
    sources: list[ApprovedSource],
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> ScrapeStats:
    """Render configured scheme pages, extract text, and download relevant linked PDFs."""
    page_sources = [
        source
        for source in sources
        if not urlsplit(source.url).path.lower().endswith(".pdf")
    ]
    direct_pdf_sources = [
        source
        for source in sources
        if urlsplit(source.url).path.lower().endswith(".pdf")
    ]
    if not page_sources and not direct_pdf_sources:
        raise ValueError("No explicitly configured pages or PDFs were provided.")

    webpages_scraped = 0
    relevant_pdfs_discovered = 0
    counts = {"new": 0, "changed": 0, "unchanged": 0}
    failures: list[str] = []
    output_directory.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        try:
            for source in page_sources:
                validate_source(source)
                page = context.new_page()
                try:
                    def guard_navigation(route) -> None:
                        request = route.request
                        if request.is_navigation_request() and not _is_allowed_url(
                            request.url, source.organization
                        ):
                            route.abort()
                            return
                        route.continue_()

                    page.route("**/*", guard_navigation)
                    response = page.goto(
                        source.url,
                        wait_until="networkidle",
                        timeout=60_000,
                    )
                    if response is None or response.status != 200:
                        status = response.status if response else "no response"
                        raise ValueError(f"HTTP {status} for {source.url}")
                    if not _is_allowed_url(page.url, source.organization):
                        raise ValueError(f"Unapproved final page URL: {page.url}")
                    if not _content_type(response, page.url, "text/html"):
                        raise ValueError(f"Unexpected page content type for {source.url}")

                    title = page.title().strip() or _scheme_text(source.scheme_id)
                    body_text, normalized_html = _rendered_page_content(page, title)
                    source_path = urlsplit(source.url).path
                    if (
                        source.organization == "hdfc_amc"
                        and urlsplit(source.url).hostname == "www.hdfcfund.com"
                        and source_path.startswith("/explore/mutual-funds/")
                        and source_path.rstrip("/").endswith("/direct")
                        and not has_scheme_page_key_facts(body_text)
                    ):
                        raise ValueError(
                            "Rendered scheme page is missing a TER or riskometer value."
                        )
                    html_body = normalized_html.encode("utf-8")
                    status = _save_document(
                        source_id=source.source_id,
                        scheme_id=source.scheme_id,
                        organization=source.organization,
                        url=page.url,
                        content_type="text/html",
                        body=html_body,
                        output_directory=output_directory,
                        document_title=title,
                        document_type="scheme_page",
                        publication_date=_publication_date(body_text),
                    )
                    counts[status] += 1
                    webpages_scraped += 1

                    pdf_links = _discover_pdfs(
                        _page_links(page), source.organization
                    )
                    for link in pdf_links:
                        try:
                            result = _download_pdf(
                                context, link, source, page.url, output_directory
                            )
                            if result is None:
                                continue
                            (
                                pdf_body,
                                document_type,
                                pdf_title,
                                publication_date,
                                pdf_source_id,
                            ) = result
                            relevant_pdfs_discovered += 1
                            status = _save_document(
                                source_id=pdf_source_id,
                                scheme_id=source.scheme_id,
                                organization=source.organization,
                                url=link.url,
                                content_type="application/pdf",
                                body=pdf_body,
                                output_directory=output_directory,
                                document_title=pdf_title,
                                document_type=document_type,
                                publication_date=publication_date,
                                source_page_url=page.url,
                            )
                            counts[status] += 1
                        except (PlaywrightError, ValueError, OSError) as exc:
                            failures.append(f"{link.url}: {exc}")
                except (PlaywrightError, ValueError, OSError) as exc:
                    failures.append(f"{source.url}: {exc}")
                finally:
                    page.close()

            for source in direct_pdf_sources:
                validate_source(source)
                try:
                    response = context.request.get(
                        source.url,
                        timeout=60_000,
                        max_redirects=MAX_REDIRECTS,
                    )
                    body = _check_download_response(
                        response, source.url, source.organization
                    )
                    digest = hashlib.sha256(body).hexdigest()
                    cached = _cached_document_metadata(
                        output_directory,
                        source.source_id,
                        digest,
                        source.scheme_id,
                    )
                    if cached:
                        title = str(cached.get("document_title") or source.source_id)
                        document_type = str(cached["document_type"])
                        publication_date = (
                            cached.get("publication_date")
                            if isinstance(cached.get("publication_date"), str)
                            else None
                        )
                    else:
                        title, opening_text, full_text = _pdf_details(body)
                        document_type = classify_document_type(
                            source.url, title, opening_text
                        )
                        if document_type is None or not _matches_scheme(
                            source.scheme_id, source.url, title, opening_text
                        ):
                            failures.append(
                                f"{source.url}: PDF is not clearly scheme-specific SID, KIM, or factsheet content."
                            )
                            continue
                        publication_date = _publication_date(full_text)
                    status = _save_document(
                        source_id=source.source_id,
                        scheme_id=source.scheme_id,
                        organization=source.organization,
                        url=source.url,
                        content_type="application/pdf",
                        body=body,
                        output_directory=output_directory,
                        document_title=title or source.source_id,
                        document_type=document_type,
                        publication_date=publication_date,
                    )
                    counts[status] += 1
                    relevant_pdfs_discovered += 1
                except (PlaywrightError, ValueError, OSError) as exc:
                    failures.append(f"{source.url}: {exc}")
        finally:
            context.close()
            browser.close()

    return ScrapeStats(
        webpages_scraped=webpages_scraped,
        relevant_pdfs_discovered=relevant_pdfs_discovered,
        new_documents=counts["new"],
        changed_documents=counts["changed"],
        unchanged_documents=counts["unchanged"],
        failed_urls=tuple(failures),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render configured scheme pages and discover official scheme PDFs."
    )
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCE_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args(argv)
    try:
        sources = load_sources(args.sources)
        if not sources:
            raise ValueError(f"No sources configured in {args.sources}.")
        stats = scrape_source_pages(sources, args.output)
    except (OSError, PlaywrightError, ValueError) as exc:
        print(f"Scraping failed: {exc}", file=sys.stderr)
        return 1

    print(f"Webpages successfully scraped: {stats.webpages_scraped}")
    print(f"Relevant PDFs discovered: {stats.relevant_pdfs_discovered}")
    print(
        "Documents: "
        f"{stats.new_documents} new, "
        f"{stats.changed_documents} changed, "
        f"{stats.unchanged_documents} unchanged"
    )
    for failure in stats.failed_urls:
        print(f"FAILED: {failure}", file=sys.stderr)
    return 1 if stats.failed_urls else 0


if __name__ == "__main__":
    raise SystemExit(main())
