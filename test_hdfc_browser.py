from playwright.sync_api import sync_playwright

urls = {
    "Large Cap": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct",
    "Flexi Cap": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/direct",
    "ELSS Tax Saver": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-elss-tax-saver-fund/direct",
    "Mid Cap": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-mid-cap-fund/direct",
}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    for name, url in urls.items():
        try:
            response = page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=60000
            )

            print(f"\n{name}")
            print("HTTP Status:", response.status if response else "No response")
            print("Page Title:", page.title())
            print("Final URL:", page.url)
            print("Content Length:", len(page.locator("body").inner_text()))

        except Exception as e:
            print(f"\n{name} failed:", str(e))

    browser.close()