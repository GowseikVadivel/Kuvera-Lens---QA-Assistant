"""Import user-provided HDFC 2025 SID PDFs into the raw document directory."""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from kuvera_lens.scraper import DEFAULT_OUTPUT_DIRECTORY

EXPECTED_DOCUMENT_DATE = "This Scheme Information Document is dated November 21, 2025."


@dataclass(frozen=True)
class SIDImport:
    source_id: str
    filename: str
    scheme_id: str
    document_title: str
    official_url: str
    expected_name: str


SIDS = (
    SIDImport(
        "hdfc-large-cap-sid-2025-11-21",
        "SID - HDFC Large Cap Fund.pdf",
        "hdfc_large_cap_direct_growth",
        "HDFC Large Cap Fund - SID",
        "https://files.hdfcfund.com/s3fs-public/SID/2025-11/"
        "SID%20-%20HDFC%20Large%20Cap%20Fund%20dated%20November%2021%2C%202025_0.pdf"
        "?VersionId=EaBqM_CLfPzNZ0sa96.9__cmqJdRI6jq",
        "HDFC Large Cap Fund",
    ),
    SIDImport(
        "hdfc-flexi-cap-sid-2025-11-21",
        "SID - HDFC Flexi Cap Fund.pdf",
        "hdfc_flexi_cap_direct_growth",
        "HDFC Flexi Cap Fund - SID",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/direct",
        "HDFC Flexi Cap Fund",
    ),
    SIDImport(
        "hdfc-elss-tax-saver-sid-2025-11-21",
        "SID - HDFC ELSS Tax Saver.pdf",
        "hdfc_elss_tax_saver_direct_growth",
        "HDFC ELSS Tax Saver - SID",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-elss-tax-saver-fund/direct",
        "HDFC ELSS Tax Saver",
    ),
    SIDImport(
        "hdfc-mid-cap-sid-2025-11-21",
        "SID - HDFC Mid Cap Fund.pdf",
        "hdfc_mid_cap_direct_growth",
        "HDFC Mid Cap Fund - SID",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-mid-cap-fund/direct",
        "HDFC Mid Cap Fund",
    ),
)


class SIDImportError(ValueError):
    """Raised when a provided SID cannot be verified or imported."""


def _verify_pdf(path: Path, expected_name: str) -> tuple[bytes, int]:
    try:
        body = path.read_bytes()
    except OSError as exc:
        raise SIDImportError(f"Cannot read {path}: {exc}") from exc
    if not body.startswith(b"%PDF-"):
        raise SIDImportError(f"{path.name} is not a PDF file.")

    try:
        with pymupdf.open(stream=body, filetype="pdf") as document:
            if document.page_count == 0:
                raise SIDImportError(f"{path.name} has no pages.")
            opening_text = "\n".join(
                document[index].get_text("text")
                for index in range(min(4, document.page_count))
            )
            full_text = "\n".join(page.get_text("text") for page in document)
            page_count = document.page_count
    except SIDImportError:
        raise
    except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError, ValueError) as exc:
        raise SIDImportError(f"Cannot parse {path.name}: {exc}") from exc

    if expected_name.casefold() not in opening_text.casefold():
        raise SIDImportError(
            f"{path.name} does not contain the expected scheme name "
            f"{expected_name!r} in its opening pages."
        )
    if EXPECTED_DOCUMENT_DATE.casefold() not in full_text.casefold():
        raise SIDImportError(
            f"{path.name} does not contain the expected SID date "
            f"{EXPECTED_DOCUMENT_DATE!r}."
        )
    return body, page_count


def import_sids(input_directory: Path, output_directory: Path) -> list[Path]:
    """Validate all four PDFs, then copy them with source metadata sidecars."""
    verified: list[tuple[SIDImport, bytes, int]] = []
    for sid in SIDS:
        path = input_directory / sid.filename
        body, page_count = _verify_pdf(path, sid.expected_name)
        verified.append((sid, body, page_count))

    output_directory.mkdir(parents=True, exist_ok=True)
    manifests: list[Path] = []
    for sid, body, page_count in verified:
        digest = hashlib.sha256(body).hexdigest()
        pdf_name = f"{sid.source_id}-{digest}.pdf"
        manifest_path = output_directory / f"{sid.source_id}-{digest}.json"
        (output_directory / pdf_name).write_bytes(body)
        metadata = {
            "source_id": sid.source_id,
            "scheme_id": sid.scheme_id,
            "organization": "hdfc_amc",
            "url": sid.official_url,
            "content_type": "application/pdf",
            "sha256": digest,
            "downloaded_file": pdf_name,
            "document_title": sid.document_title,
            "document_date": "2025-11-21",
            "page_count": page_count,
            "provided_by_user": True,
            "official_origin_confirmed_by_user": True,
        }
        manifest_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        manifests.append(manifest_path)
    return manifests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import and validate the four user-provided November 2025 HDFC SIDs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing the four SID PDFs with their expected filenames.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Raw document directory (default: data/raw).",
    )
    args = parser.parse_args(argv)
    try:
        manifests = import_sids(args.input, args.output)
    except (OSError, SIDImportError) as exc:
        print(f"SID import failed: {exc}", file=sys.stderr)
        return 1

    for manifest in manifests:
        print(f"Imported {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
