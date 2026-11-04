import json
import tempfile
import unittest
from pathlib import Path

from kuvera_lens.chunking import (
    chunk_document,
    chunk_documents,
    split_text,
    write_chunks,
)
from kuvera_lens.loaders import ExtractedText, LoadedDocument


def make_document(
    *,
    source_id: str = "mid-cap-sid",
    url: str = "https://files.hdfcfund.com/mid-cap-sid.pdf",
    content_hash: str = "a" * 64,
    content: tuple[ExtractedText, ...] | None = None,
) -> LoadedDocument:
    return LoadedDocument(
        source_id=source_id,
        scheme_id="hdfc_mid_cap_direct_growth",
        scheme_name="HDFC Mid Cap Fund – Direct Growth",
        organization="hdfc_amc",
        url=url,
        source_domain="files.hdfcfund.com",
        content_type="application/pdf",
        document_title="HDFC Mid Cap Fund SID",
        content_hash=content_hash,
        local_path=Path("mid-cap-sid.pdf"),
        content=content
        or (
            ExtractedText(
                "Minimum SIP is Rs. 100. Additional purchases are subject to scheme terms.",
                page_number=12,
                section="Investment Details",
            ),
        ),
        document_type="SID",
        publication_date="November 21, 2025",
        last_checked_at="2026-11-03T00:00:00+00:00",
    )


class TextChunkingTests(unittest.TestCase):
    def test_splits_at_sentence_boundaries_with_configured_overlap(self) -> None:
        text = (
            "First sentence contains an exact value Rs. 100. "
            "Second sentence describes the SIP condition. "
            "Third sentence gives another factual condition."
        )
        chunks = split_text(text, max_chars=75, overlap_chars=25)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 75 for chunk in chunks))
        self.assertTrue(any("Rs. 100." in chunk for chunk in chunks))
        self.assertTrue(any(
            set(chunks[index].split()) & set(chunks[index + 1].split())
            for index in range(len(chunks) - 1)
        ))

    def test_long_sentence_splits_at_word_boundaries_without_losing_words(self) -> None:
        words = ["financial"] * 40 + ["Rs."] + ["100"] + ["monthly"] * 30
        text = " ".join(words)
        chunks = split_text(text, max_chars=50, overlap_chars=10)

        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= 50 for chunk in chunks))
        reconstructed_words = set(" ".join(chunks).split())
        self.assertTrue(set(words).issubset(reconstructed_words))
        self.assertTrue(
            any(
                set(chunks[index].split()) & set(chunks[index + 1].split())
                for index in range(len(chunks) - 1)
            )
        )

    def test_blank_text_returns_no_chunks_and_invalid_sizes_fail(self) -> None:
        self.assertEqual(split_text(" \n\t "), [])
        with self.assertRaisesRegex(ValueError, "max_chars"):
            split_text("some text", max_chars=0)
        with self.assertRaisesRegex(ValueError, "overlap_chars"):
            split_text("some text", max_chars=10, overlap_chars=10)


class DocumentChunkingTests(unittest.TestCase):
    def test_keeps_pdf_page_and_citation_metadata(self) -> None:
        document = make_document()
        chunks = chunk_document(document)

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.metadata["scheme_id"], document.scheme_id)
        self.assertEqual(chunk.metadata["scheme_name"], document.scheme_name)
        self.assertEqual(chunk.metadata["source_id"], document.source_id)
        self.assertEqual(chunk.metadata["source_url"], document.url)
        self.assertEqual(chunk.metadata["document_title"], document.document_title)
        self.assertEqual(chunk.metadata["document_type"], "SID")
        self.assertEqual(chunk.metadata["page_number"], 12)
        self.assertEqual(chunk.metadata["section"], "Investment Details")
        self.assertEqual(chunk.metadata["publication_date"], "November 21, 2025")
        self.assertEqual(chunk.metadata["content_hash"], document.content_hash)

    def test_pages_remain_separate_and_chunk_ids_are_deterministic(self) -> None:
        document = make_document(
            content=(
                ExtractedText("First page has scheme details.", page_number=1),
                ExtractedText("Second page has exit load details.", page_number=2),
            )
        )
        first = chunk_document(document)
        second = chunk_document(document)

        self.assertEqual([chunk.metadata["page_number"] for chunk in first], [1, 2])
        self.assertEqual(
            [chunk.chunk_id for chunk in first],
            [chunk.chunk_id for chunk in second],
        )
        self.assertNotEqual(first[0].chunk_id, first[1].chunk_id)

    def test_rejects_unapproved_or_ambiguous_source_documents(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid source provenance"):
            chunk_document(make_document(url="https://example.com/fund.pdf"))

        with self.assertRaisesRegex(ValueError, "invalid content hash"):
            chunk_document(make_document(content_hash="not-a-hash"))

    def test_deduplicates_same_content_and_prefers_direct_pdf_citation(self) -> None:
        page_source = make_document(
            source_id="mid-cap-page",
            url="https://www.hdfcfund.com/explore/mutual-funds/hdfc-mid-cap-fund/direct",
        )
        direct_pdf = make_document(source_id="mid-cap-direct-sid")

        chunks = chunk_documents([page_source, direct_pdf])

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["source_id"], "mid-cap-direct-sid")
        self.assertEqual(chunks[0].metadata["source_url"], direct_pdf.url)

    def test_rejects_conflicting_versions_for_same_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "Conflicting loaded document versions"):
            chunk_documents(
                [
                    make_document(content_hash="a" * 64),
                    make_document(content_hash="b" * 64),
                ]
            )

    def test_writes_metadata_jsonl_atomically_and_is_idempotent(self) -> None:
        document = make_document()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            first_counts = write_chunks([document], output)
            active_manifest = json.loads(
                (output / "active_chunks.json").read_text(encoding="utf-8")
            )
            output_file = output / active_manifest["files"][0]["path"]
            first_bytes = output_file.read_bytes()
            second_counts = write_chunks([document], output)
            second_manifest = json.loads(
                (output / "active_chunks.json").read_text(encoding="utf-8")
            )
            self.assertEqual(active_manifest, second_manifest)
            second_bytes = output_file.read_bytes()
            rows = [
                json.loads(line)
                for line in output_file.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(first_counts, second_counts)
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(len(rows), first_counts[document.source_id])
        self.assertTrue(all(row["chunk_id"] for row in rows))
        self.assertTrue(all(row["metadata"]["page_number"] == 12 for row in rows))
        self.assertEqual(active_manifest["chunk_size"], 1200)
        self.assertEqual(active_manifest["overlap_chars"], 160)

    def test_new_document_version_publishes_new_manifest_without_deleting_old_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            write_chunks([make_document()], output)
            previous_manifest = json.loads(
                (output / "active_chunks.json").read_text(encoding="utf-8")
            )
            previous_file = output / previous_manifest["files"][0]["path"]

            changed = make_document(
                content_hash="b" * 64,
                content=(
                    ExtractedText(
                        "Updated SID wording, minimum SIP remains Rs. 100.",
                        page_number=12,
                    ),
                ),
            )
            write_chunks([changed], output)
            active_manifest = json.loads(
                (output / "active_chunks.json").read_text(encoding="utf-8")
            )
            active_file = output / active_manifest["files"][0]["path"]
            self.assertNotEqual(previous_file, active_file)
            self.assertTrue(previous_file.exists())
            self.assertTrue(active_file.exists())

        self.assertEqual(active_manifest["files"][0]["content_hash"], "b" * 64)


if __name__ == "__main__":
    unittest.main()
