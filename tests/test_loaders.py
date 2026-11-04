import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pymupdf

from kuvera_lens.loaders import DocumentLoadError, load_document, load_documents


def write_source(
    directory: Path,
    body: bytes,
    content_type: str,
    *,
    source_id: str = "large-cap-scheme",
    url: str = "https://www.hdfcfund.com/scheme",
    scheme_id: str = "hdfc_large_cap_direct_growth",
    organization: str = "hdfc_amc",
    extension: str | None = None,
    document_title: str | None = None,
    document_type: str | None = None,
    last_checked_at: str | None = None,
) -> Path:
    digest = hashlib.sha256(body).hexdigest()
    extension = extension or (".pdf" if content_type == "application/pdf" else ".html")
    filename = f"{source_id}-{digest}{extension}"
    (directory / filename).write_bytes(body)
    manifest = directory / f"{source_id}-{digest}.json"
    metadata = {
        "source_id": source_id,
        "scheme_id": scheme_id,
        "organization": organization,
        "url": url,
        "content_type": content_type,
        "sha256": digest,
        "downloaded_file": filename,
    }
    if document_title:
        metadata["document_title"] = document_title
    if document_type:
        metadata["document_type"] = document_type
    if last_checked_at:
        metadata["last_checked_at"] = last_checked_at
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    return manifest


def make_pdf(pages: list[str], title: str = "Fund document") -> bytes:
    document = pymupdf.open()
    document.set_metadata({"title": title})
    for text in pages:
        page = document.new_page()
        page.insert_text((72, 72), text)
    body = document.tobytes()
    document.close()
    return body


class DocumentLoaderTests(unittest.TestCase):
    def test_extracts_html_text_and_source_provenance(self) -> None:
        html = (
            b"<html><head><title>Scheme Details</title>"
            b"<script>secret ignored</script></head><body>"
            b"<h1>Investment details</h1><p>Minimum SIP is stated here.</p>"
            b'<a href="/internal-page">Internal link label</a>'
            b"</body></html>"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            manifest = write_source(directory, html, "text/html")
            document = load_document(manifest, directory)

        self.assertEqual(document.document_title, "Scheme Details")
        self.assertEqual(document.scheme_id, "hdfc_large_cap_direct_growth")
        self.assertEqual(document.source_domain, "www.hdfcfund.com")
        self.assertIn("Minimum SIP is stated here.", document.content[0].text)
        self.assertNotIn("secret ignored", document.content[0].text)
        self.assertNotIn("/internal-page", document.content[0].text)
        self.assertIn("Investment details", document.content[0].section or "")
        self.assertIsNone(document.content[0].page_number)

    def test_extracts_pdf_text_by_page_and_document_title(self) -> None:
        pdf = make_pdf(["First page fact", "Second page fact"])
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            manifest = write_source(
                directory,
                pdf,
                "application/pdf",
                url="https://www.hdfcfund.com/factsheet.pdf",
            )
            document = load_document(manifest, directory)

        self.assertEqual(document.document_title, "Fund document")
        self.assertEqual([item.page_number for item in document.content], [1, 2])
        self.assertIn("First page fact", document.content[0].text)
        self.assertIn("Second page fact", document.content[1].text)

    def test_corrects_kim_metadata_that_was_mislabeled_as_sid(self) -> None:
        pdf = make_pdf(["HDFC Large Cap Fund Key Information Memorandum"], "KIM - HDFC Large Cap Fund")
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            manifest = write_source(
                directory,
                pdf,
                "application/pdf",
                source_id="hdfc_large_cap_direct_growth-sid-old",
                url="https://files.hdfcfund.com/s3fs-public/KIM/large-cap.pdf",
                document_title="KIM - HDFC Large Cap Fund",
                document_type="SID",
            )
            document = load_document(manifest, directory)

        self.assertEqual(document.document_type, "KIM")

    def test_selects_latest_complete_hdfc_scheme_page_capture(self) -> None:
        source_id = "hdfc-large-cap-direct"
        url = (
            "https://www.hdfcfund.com/explore/mutual-funds/"
            "hdfc-large-cap-fund/direct"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            complete = write_source(
                directory,
                b"<html><body>TER 1.02 Riskometer Very High</body></html>",
                "text/html",
                source_id=source_id,
                url=url,
                document_type="scheme_page",
                last_checked_at="2026-11-03T10:00:00+00:00",
            )
            incomplete = write_source(
                directory,
                b"<html><body>TER 1.02 Riskometer Product Suitability FAQs</body></html>",
                "text/html",
                source_id=source_id,
                url=url,
                document_type="scheme_page",
                last_checked_at="2026-11-03T11:00:00+00:00",
            )
            complete_hash = json.loads(complete.read_text())["sha256"]
            incomplete_hash = json.loads(incomplete.read_text())["sha256"]

            documents = load_documents(directory)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].content_hash, complete_hash)
        self.assertNotEqual(documents[0].content_hash, incomplete_hash)

    def test_rejects_modified_file_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            manifest = write_source(directory, b"<p>Original</p>", "text/html")
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            (directory / metadata["downloaded_file"]).write_bytes(b"<p>Changed</p>")

            with self.assertRaisesRegex(DocumentLoadError, "hash mismatch"):
                load_document(manifest, directory)

    def test_rejects_unapproved_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            manifest = write_source(
                directory,
                b"<p>Not trusted</p>",
                "text/html",
                url="https://example.com/scheme",
            )

            with self.assertRaisesRegex(DocumentLoadError, "invalid source provenance"):
                load_document(manifest, directory)

    def test_rejects_empty_or_non_text_html(self) -> None:
        for body in (b"", b"<html><script>only script</script></html>"):
            with self.subTest(body=body):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    directory = Path(temporary_directory)
                    manifest = write_source(directory, body, "text/html")
                    with self.assertRaises(DocumentLoadError):
                        load_document(manifest, directory)

    def test_rejects_pdf_without_extractable_text(self) -> None:
        pdf = make_pdf([""])
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            manifest = write_source(
                directory,
                pdf,
                "application/pdf",
                url="https://www.hdfcfund.com/empty.pdf",
            )
            with self.assertRaisesRegex(DocumentLoadError, "no extractable text"):
                load_document(manifest, directory)

    def test_batch_fails_explicitly_when_manifest_is_missing_or_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            with self.assertRaisesRegex(DocumentLoadError, "No scraper metadata"):
                load_documents(directory)

            (directory / "broken.json").write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(DocumentLoadError, "Document loading failed"):
                load_documents(directory)


if __name__ == "__main__":
    unittest.main()
