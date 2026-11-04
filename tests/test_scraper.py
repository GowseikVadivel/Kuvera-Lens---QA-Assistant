import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

from kuvera_lens.scraper import (
    DiscoveredLink,
    _check_download_response,
    _discover_pdfs,
    _download_pdf,
    _save_document,
    _matches_scheme,
    classify_document_type,
    scrape_source_pages,
)
from kuvera_lens.sources import ApprovedSource, load_sources, validate_source


class FakeLocator:
    def __init__(self, value: str = "", links: list[dict[str, str]] | None = None) -> None:
        self.value = value
        self.links = links or []

    def inner_text(self, timeout: int | None = None) -> str:
        return self.value

    def evaluate_all(self, script: str) -> list[dict[str, str]]:
        return self.links

    def count(self) -> int:
        return 0

    def all_inner_texts(self) -> list[str]:
        return ["Investment Details", "Minimum SIP"]


class FakeRequestNavigation:
    def __init__(self, url: str, navigation: bool = True) -> None:
        self.url = url
        self._navigation = navigation

    def is_navigation_request(self) -> bool:
        return self._navigation


class FakeRoute:
    def __init__(self, url: str, navigation: bool = True) -> None:
        self.request = FakeRequestNavigation(url, navigation)
        self.action: str | None = None

    def abort(self) -> None:
        self.action = "abort"

    def continue_(self) -> None:
        self.action = "continue"


class FakePage:
    def __init__(self, url: str, links: list[dict[str, str]]) -> None:
        self.url = url
        self.links = links
        self.closed = False
        self.route_handler = None

    def route(self, pattern: str, handler) -> None:
        self.route_handler = handler

    def goto(self, url: str, **kwargs):
        return FakeResponse(url, b"<html>page</html>", "text/html")

    def locator(self, selector: str) -> FakeLocator:
        if selector == "body":
            return FakeLocator(
                "HDFC Large Cap Fund scheme details and long-term investment information. "
                "Fund Managers Mr. Rahul Baijal. Total Expense Ratio TER 1.02 "
                "Riskometer Very High. Minimum SIP is Rs. 100."
            )
        return FakeLocator(links=self.links)

    def content(self) -> str:
        return "<html><head><title>HDFC Large Cap Fund</title></head><body>Scheme content</body></html>"

    def title(self) -> str:
        return "HDFC Large Cap Fund"

    def close(self) -> None:
        self.closed = True


class FakeResponse:
    def __init__(
        self,
        url: str,
        body: bytes,
        content_type: str,
        status: int = 200,
    ) -> None:
        self.url = url
        self._body = body
        self.status = status
        self.headers = {"content-type": content_type}

    def body(self) -> bytes:
        return self._body


class FakeRequest:
    def __init__(self, responses: dict[str, FakeResponse]) -> None:
        self.responses = responses
        self.urls: list[str] = []

    def get(self, url: str, **kwargs):
        self.urls.append(url)
        return self.responses[url]


class FakeContext:
    def __init__(self, links: list[dict[str, str]], responses: dict[str, FakeResponse]) -> None:
        self.page = FakePage(
            "https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct",
            links,
        )
        self.request = FakeRequest(responses)
        self.closed = False

    def new_page(self) -> FakePage:
        return self.page

    def close(self) -> None:
        self.closed = True


class FakeBrowser:
    def __init__(self, context: FakeContext) -> None:
        self.context = context
        self.closed = False

    def new_context(self) -> FakeContext:
        return self.context

    def close(self) -> None:
        self.closed = True


class FakeChromium:
    def __init__(self, browser: FakeBrowser) -> None:
        self.browser = browser

    def launch(self, headless: bool = True) -> FakeBrowser:
        return self.browser


class FakePlaywright:
    def __init__(self, browser: FakeBrowser) -> None:
        self.chromium = FakeChromium(browser)


class FakePlaywrightManager:
    def __init__(self, browser: FakeBrowser) -> None:
        self.playwright = FakePlaywright(browser)

    def __enter__(self) -> FakePlaywright:
        return self.playwright

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None


def make_pdf(text: str, title: str) -> bytes:
    document = pymupdf.open()
    document.set_metadata({"title": title})
    page = document.new_page()
    page.insert_text((72, 72), text)
    body = document.tobytes()
    document.close()
    return body


def large_cap_source() -> ApprovedSource:
    return ApprovedSource(
        "hdfc-large-cap-direct",
        "hdfc_large_cap_direct_growth",
        "hdfc_amc",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct",
    )


class SourceValidationTests(unittest.TestCase):
    def test_loads_explicit_sources_and_rejects_non_allowlisted_hosts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources.json"
            path.write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "source_id": "large-cap",
                                "scheme_id": "hdfc_large_cap_direct_growth",
                                "organization": "hdfc_amc",
                                "url": "https://www.hdfcfund.com/scheme",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            sources = load_sources(path)
        self.assertEqual(len(sources), 1)

        with self.assertRaisesRegex(ValueError, "not approved"):
            validate_source(
                ApprovedSource(
                    "invalid",
                    "hdfc_large_cap_direct_growth",
                    "hdfc_amc",
                    "https://not-hdfcfund.com/scheme",
                )
            )

    def test_classifies_only_in_scope_document_types(self) -> None:
        self.assertEqual(classify_document_type("Scheme Information Document"), "SID")
        self.assertEqual(classify_document_type("HDFC Factsheet"), "factsheet")
        self.assertEqual(classify_document_type("Key Information Memorandum"), "KIM")
        self.assertEqual(
            classify_document_type(
                "KIM - HDFC Mid Cap Fund dated November 21, 2025",
                "KIM - HDFC Mid Cap Fund",
                "HDFC Mid Cap Fund",
                "This document refers to the Scheme Information Document (SID).",
            ),
            "KIM",
        )
        self.assertIsNone(classify_document_type("Generic AMC booklet"))

    def test_requires_exact_scheme_identity_for_pdf_relevance(self) -> None:
        self.assertTrue(
            _matches_scheme(
                "hdfc_large_cap_direct_growth",
                "SID HDFC Large Cap Fund",
            )
        )
        self.assertFalse(
            _matches_scheme(
                "hdfc_large_cap_direct_growth",
                "HDFC Mutual Fund Investor Booklet",
            )
        )

    def test_discovery_ignores_non_pdf_and_unapproved_links(self) -> None:
        links = [
            DiscoveredLink("https://www.hdfcfund.com/docs/sid.pdf", "SID"),
            DiscoveredLink("https://evil.example/sid.pdf", "SID"),
            DiscoveredLink("https://www.hdfcfund.com/other-scheme", "Other"),
        ]
        found = _discover_pdfs(links, "hdfc_amc")
        self.assertEqual([link.url for link in found], ["https://www.hdfcfund.com/docs/sid.pdf"])

    def test_pdf_validation_checks_status_content_type_and_signature(self) -> None:
        valid = FakeResponse(
            "https://files.hdfcfund.com/file.pdf",
            b"%PDF-1.7 data",
            "application/pdf",
        )
        self.assertEqual(
            _check_download_response(valid, valid.url, "hdfc_amc"),
            b"%PDF-1.7 data",
        )
        for response in (
            FakeResponse(valid.url, b"%PDF-data", "application/pdf", status=403),
            FakeResponse(valid.url, b"%PDF-data", "text/html"),
            FakeResponse(valid.url, b"not a pdf", "application/pdf"),
        ):
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    _check_download_response(response, valid.url, "hdfc_amc")


class PlaywrightScraperTests(unittest.TestCase):
    def test_scrapes_page_downloads_relevant_pdf_and_skips_generic_pdf(self) -> None:
        scheme_pdf_url = "https://files.hdfcfund.com/docs/hdfc-large-cap-sid.pdf"
        generic_pdf_url = "https://files.hdfcfund.com/docs/amc-booklet.pdf"
        scheme_pdf = make_pdf(
            "HDFC Large Cap Fund Scheme Information Document",
            "HDFC Large Cap Fund SID",
        )
        generic_pdf = make_pdf(
            "HDFC Mutual Fund Investor Booklet",
            "HDFC Mutual Fund Booklet",
        )
        links = [
            {"url": scheme_pdf_url, "label": "HDFC Large Cap Fund SID"},
            {"url": generic_pdf_url, "label": "Investor booklet"},
        ]
        context = FakeContext(
            links,
            {
                scheme_pdf_url: FakeResponse(scheme_pdf_url, scheme_pdf, "application/pdf"),
                generic_pdf_url: FakeResponse(generic_pdf_url, generic_pdf, "application/pdf"),
            },
        )
        browser = FakeBrowser(context)
        source = large_cap_source()

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "kuvera_lens.scraper.sync_playwright",
                return_value=FakePlaywrightManager(browser),
            ):
                stats = scrape_source_pages([source], Path(directory))
            manifests = list(Path(directory).glob("*.json"))
            html_files = list(Path(directory).glob("*.html"))
            html_content = html_files[0].read_text(encoding="utf-8")
            loaded_metadata = [
                json.loads(manifest.read_text(encoding="utf-8"))
                for manifest in manifests
            ]

        self.assertEqual(stats.webpages_scraped, 1)
        self.assertEqual(stats.relevant_pdfs_discovered, 1)
        self.assertEqual(stats.new_documents, 2)
        self.assertEqual(stats.failed_urls, ())
        blocked_route = FakeRoute("https://evil.example/redirect", navigation=True)
        context.page.route_handler(blocked_route)
        self.assertEqual(blocked_route.action, "abort")
        allowed_resource = FakeRoute("https://cdn.example/resource.js", navigation=False)
        context.page.route_handler(allowed_resource)
        self.assertEqual(allowed_resource.action, "continue")
        self.assertEqual(len(manifests), 2)
        pdf_metadata = next(item for item in loaded_metadata if item["content_type"] == "application/pdf")
        self.assertEqual(pdf_metadata["scheme_name"], "HDFC Large Cap Fund – Direct Growth")
        self.assertEqual(pdf_metadata["document_type"], "SID")
        self.assertEqual(pdf_metadata["source_page_url"], source.url)
        self.assertTrue(pdf_metadata["last_checked_at"])
        self.assertIn(generic_pdf_url, context.request.urls)
        self.assertEqual(len(html_files), 1)
        self.assertIn("<h2>Investment Details</h2>", html_content)

    def test_scraper_does_not_save_scheme_page_without_required_metrics(self) -> None:
        source = large_cap_source()
        context = FakeContext([], {})
        context.page.body_text = (
            "HDFC Large Cap Fund. " * 8
            + "TER 1.02 Riskometer Product Suitability."
        )
        original_locator = context.page.locator

        def locator(selector: str) -> FakeLocator:
            if selector == "body":
                return FakeLocator(context.page.body_text)
            return original_locator(selector)

        context.page.locator = locator
        browser = FakeBrowser(context)
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "kuvera_lens.scraper.sync_playwright",
                return_value=FakePlaywrightManager(browser),
            ):
                stats = scrape_source_pages([source], Path(directory))

            self.assertEqual(list(Path(directory).glob("*.json")), [])
            self.assertEqual(len(stats.failed_urls), 1)
            self.assertTrue(stats.failed_urls[0].startswith(source.url))
            self.assertEqual(stats.webpages_scraped, 0)

    def test_cached_kim_document_is_reclassified_instead_of_trusted_as_sid(self) -> None:
        source = large_cap_source()
        url = "https://files.hdfcfund.com/s3fs-public/KIM/large-cap.pdf"
        body = b"%PDF-1.7 cached KIM"
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            _save_document(
                source_id="hdfc_large_cap_direct_growth-sid-old",
                scheme_id=source.scheme_id,
                organization=source.organization,
                url=url,
                content_type="application/pdf",
                body=body,
                output_directory=output,
                document_title="KIM - HDFC Large Cap Fund",
                document_type="SID",
                publication_date=None,
            )
            context = FakeContext([], {url: FakeResponse(url, body, "application/pdf")})

            result = _download_pdf(
                context,
                DiscoveredLink(url, "HDFC Large Cap Fund SID"),
                source,
                source.url,
                output,
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result[1], "KIM")
        self.assertTrue(result[4].startswith(f"{source.scheme_id}-kim-"))

    def test_repeated_scrape_reports_unchanged_documents(self) -> None:
        source = large_cap_source()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for attempt in range(2):
                browser = FakeBrowser(FakeContext([], {}))
                with patch(
                    "kuvera_lens.scraper.sync_playwright",
                    return_value=FakePlaywrightManager(browser),
                ):
                    stats = scrape_source_pages([source], output)
                if attempt == 0:
                    self.assertEqual(stats.new_documents, 1)
                else:
                    self.assertEqual(stats.unchanged_documents, 1)
                    self.assertEqual(stats.new_documents, 0)
if __name__ == "__main__":
    unittest.main()
