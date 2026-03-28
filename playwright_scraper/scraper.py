import os
import time
from urllib.parse import quote
from playwright.sync_api import sync_playwright

def scrape_trendyol_keywords(keywords_file="keywords.txt", targets_dir="../targets"):
    # Ensure targets directory exists
    if not os.path.exists(targets_dir):
        os.makedirs(targets_dir)

    # Read keywords
    if not os.path.exists(keywords_file):
        print(f"Error: {keywords_file} not found.")
        return

    with open(keywords_file, "r", encoding="utf-8") as f:
        keywords = [line.strip() for line in f if line.strip()]

    if not keywords:
        print("No keywords found.")
        return

    print(f"Starting Playwright scraper for {len(keywords)} keywords...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Using a realistic user agent to help prevent blocking
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        for keyword in keywords:
            print(f"\nProcessing keyword: '{keyword}'")
            try:
                # URL encode the keyword
                query = quote(keyword)
                url = f"https://www.trendyol.com/sr?q={query}"
                print(f"Navigating to {url}")

                # Navigate and wait for the page to load
                page.goto(url, wait_until="domcontentloaded", timeout=60000)

                # Wait for initial products to appear
                # Trendyol typically uses '.prdct-cntnr-wrppr' or '.p-card-wrppr' for products
                try:
                    page.wait_for_selector(".p-card-wrppr", timeout=15000)
                except Exception as e:
                    print(f"Warning: Could not find product container initially for {keyword}. It might be empty or blocked.")

                # Scroll down multiple times to load more products (target: ~60)
                # Trendyol infinite scrolling loads items as you scroll
                for i in range(5):
                    print(f"  Scrolling... {i+1}/5")
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    time.sleep(2)  # Wait for API to load items

                # Wait for a brief moment for any final render
                time.sleep(2)

                # Get the full HTML
                html_content = page.content()

                # Format keyword for filename (replace spaces with hyphens)
                safe_keyword = keyword.replace(" ", "-")
                output_file = os.path.join(targets_dir, f"{safe_keyword}-target.html")

                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(html_content)

                print(f"Successfully saved {len(html_content)} bytes to {output_file}")

            except Exception as e:
                print(f"Failed to scrape '{keyword}': {e}")

        browser.close()
        print("\nPlaywright scraping completed.")

if __name__ == "__main__":
    scrape_trendyol_keywords()
