import tempfile
import unittest
from pathlib import Path

import pymupdf

from kuvera_lens.import_sids import (
    EXPECTED_DOCUMENT_DATE,
    SIDImportError,
    SIDS,
    import_sids,
)
from kuvera_lens.loaders import load_documents


def create_sid(path: Path, scheme_name: str, include_date: bool = True) -> None:
    document = pymupdf.open()
    page = document.new_page()
    date_line = EXPECTED_DOCUMENT_DATE if include_date else "Date not present"
    page.insert_text((72, 72), f"{scheme_name}\n{date_line}")
    document.save(path)
    document.close()


class SIDImportTests(unittest.TestCase):
    def test_imports_all_four_verified_sids_for_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_directory = root / "input"
            output_directory = root / "raw"
            input_directory.mkdir()
            for sid in SIDS:
                create_sid(input_directory / sid.filename, sid.expected_name)

            manifests = import_sids(input_directory, output_directory)
            documents = load_documents(output_directory)

        self.assertEqual(len(manifests), 4)
        self.assertEqual(len(documents), 4)
        self.assertEqual(
            {document.scheme_id for document in documents},
            {sid.scheme_id for sid in SIDS},
        )
        self.assertTrue(all(document.content for document in documents))

    def test_rejects_scheme_mismatch_before_writing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_directory = root / "input"
            output_directory = root / "raw"
            input_directory.mkdir()
            for sid in SIDS:
                create_sid(input_directory / sid.filename, sid.expected_name)
            create_sid(
                input_directory / SIDS[0].filename,
                SIDS[1].expected_name,
            )

            with self.assertRaisesRegex(SIDImportError, "expected scheme name"):
                import_sids(input_directory, output_directory)

            self.assertFalse(output_directory.exists())

    def test_rejects_document_with_wrong_date(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_directory = root / "input"
            output_directory = root / "raw"
            input_directory.mkdir()
            for sid in SIDS:
                create_sid(input_directory / sid.filename, sid.expected_name)
            create_sid(
                input_directory / SIDS[-1].filename,
                SIDS[-1].expected_name,
                include_date=False,
            )

            with self.assertRaisesRegex(SIDImportError, "expected SID date"):
                import_sids(input_directory, output_directory)
            self.assertFalse(output_directory.exists())


if __name__ == "__main__":
    unittest.main()
