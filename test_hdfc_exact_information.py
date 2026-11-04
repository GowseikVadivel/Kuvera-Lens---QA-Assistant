
from playwright.sync_api import sync_playwright

urls = {
    "Large Cap": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct",
    "Flexi Cap": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/direct",
    "ELSS Tax Saver": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-elss-tax-saver-fund/direct",
    "Mid Cap": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-mid-cap-fund/direct",
}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)

    for name, url in urls.items():
        page = browser.new_page()

        try:
            response = page.goto(
                url,
                wait_until="networkidle",
                timeout=60000
            )

            page.wait_for_timeout(3000)

            content = page.locator("body").inner_text()

            with open(
                f"{name.lower().replace(' ', '_')}.txt",
                "w",
                encoding="utf-8"
            ) as file:
                file.write(content)

            print(f"\n{name}: {response.status}")
            print("Extracted characters:", len(content))
            print("Preview:", content[:500])

        except Exception as error:
            print(f"{name} failed: {error}")

        finally:
            page.close()

    browser.close()
