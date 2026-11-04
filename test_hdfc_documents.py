
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin, urlparse
from pathlib import Path

SCHEMES = {
    "large_cap": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct",
    "flexi_cap": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/direct",
    "elss_tax_saver": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-elss-tax-saver-fund/direct",
    "mid_cap": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-mid-cap-fund/direct",
}

ALLOWED_DOMAINS = {
    "hdfcfund.com",
    "www.hdfcfund.com",
    "files.hdfcfund.com",
}

OUTPUT_DIR = Path("data/discovered_pdfs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def is_official_url(url):
    hostname = urlparse(url).hostname
    return hostname in ALLOWED_DOMAINS if hostname else False


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context()
    
    for scheme, page_url in SCHEMES.items():
        print(f"\n{'=' * 50}")
        print(f"SCHEME: {scheme.upper()}")
        print(f"{'=' * 50}")

        page = context.new_page()

        try:
            response = page.goto(
                page_url,
                wait_until="networkidle",
                timeout=60000
            )

            if not response or response.status != 200:
                print("Scheme page unavailable")
                continue

            links = page.locator("a[href]").evaluate_all(
                "(elements) => elements.map(a => ({"
                "text: a.innerText.trim(), "
                "href: a.href"
                "}))"
            )

            pdf_links = {}

            for link in links:
                url = urljoin(page_url, link["href"])
                path = urlparse(url).path.lower()

                if path.endswith(".pdf") and is_official_url(url):
                    pdf_links[url] = link["text"]

            print("Official PDF links found:", len(pdf_links))

            for index, (pdf_url, link_text) in enumerate(
                pdf_links.items(), start=1
            ):
                print(f"\nPDF {index}: {link_text or 'Untitled'}")
                print("URL:", pdf_url)

                try:
                    pdf_response = context.request.get(
                        pdf_url,
                        timeout=60000
                    )

                    print("HTTP Status:", pdf_response.status)
                    content = pdf_response.body()

                    if (
                        pdf_response.ok
                        and content.startswith(b"%PDF-")
                    ):
                        filename = OUTPUT_DIR / f"{scheme}_{index}.pdf"
                        filename.write_bytes(content)

                        print("VALID PDF: Downloaded")
                        print("Saved:", filename)
                    else:
                        print("Not a valid downloadable PDF")

                except Exception as error:
                    print("Download failed:", error)

        except Exception as error:
            print("Page failed:", error)

        finally:
            page.close()

    browser.close()

print("\nPDF discovery test completed.")
