import os
import csv
import time
import random
import traceback
from datetime import datetime
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup


class CoindeskScraper:
    def __init__(self, use_proxy=False, delay_min=2, delay_max=5):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.use_proxy = use_proxy
        self.proxy_url = "grifterProxy"
        self.proxy_username = "us"
        self.proxy_password = "pw"
        self.output_csv = f"coindesk_articles_{self.timestamp}.csv"
        self.base_url = "https://www.coindesk.com"
        self.latest_news_url = f"{self.base_url}/latest-crypto-news"
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.processed_urls = set()  # Track processed URLs to avoid duplicates

        # Initialize CSV file with headers
        with open(self.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Title',
                'URL',
                'Author',
                'Date',
                'Category',
                'Summary',
                'Content',
                'Tags'
            ])

    def scrape_homepage(self):
        """Scrape the homepage to get article links"""
        try:
            with sync_playwright() as pw:
                browser_options = {}
                context_options = {
                    "user_agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    "bypass_csp": True,
                    "ignore_https_errors": True
                }

                # Add proxy settings if enabled
                if self.use_proxy:
                    context_options["proxy"] = {
                        "server": f"http://{self.proxy_url}",
                        "username": self.proxy_username,
                        "password": self.proxy_password
                    }

                browser = pw.chromium.launch(headless=False, **browser_options)
                context = browser.new_context(**context_options)

                # Add extra headers to appear more like a real browser
                context.set_extra_http_headers({
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate, br",
                    "DNT": "1",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                    "Sec-Fetch-User": "?1",
                    "Cache-Control": "max-age=0",
                })

                page = context.new_page()
                print(f"Accessing Coindesk homepage: {self.latest_news_url}")

                # Use a more lenient wait condition and longer timeout
                try:
                    # First try with domcontentloaded which is faster
                    response = page.goto(
                        self.latest_news_url, wait_until='domcontentloaded', timeout=30000)

                    # Wait a bit for dynamic content to load
                    page.wait_for_timeout(5000)

                except Exception as e:
                    print(f"Initial page load failed: {e}")
                    print("Trying alternative approach...")

                    # Try a different approach - go to main site first
                    response = page.goto(
                        self.base_url, wait_until='domcontentloaded', timeout=30000)
                    page.wait_for_timeout(3000)

                    # Then navigate to latest news
                    response = page.goto(
                        self.latest_news_url, wait_until='domcontentloaded', timeout=30000)
                    page.wait_for_timeout(5000)

                if response is None:
                    print("No response received. Attempting to continue anyway...")
                elif response.status != 200:
                    print(f"Warning: Received status code {response.status}")

                # Extract article links
                article_links = []

                # Try to wait for content to load
                try:
                    # Wait for any link to appear
                    page.wait_for_selector('a[href]', timeout=10000)
                except Exception as e:
                    print(f"Warning: Timeout waiting for links: {e}")

                # Take a screenshot for debugging
                page.screenshot(
                    path=f"coindesk_screenshot_{self.timestamp}.png")
                print(
                    f"Saved screenshot to coindesk_screenshot_{self.timestamp}.png")

                # Get all article links - try multiple approaches
                try:
                    # First try to find article cards or containers
                    articles = page.query_selector_all(
                        'article a[href], .article-card a[href], .story-card a[href], .article a[href]')

                    if not articles or len(articles) < 5:
                        print(
                            "Few article-specific links found, trying generic links...")
                        # Fall back to all links if specific article selectors don't work
                        articles = page.query_selector_all('a[href]')

                    print(f"Found {len(articles)} potential article links")

                    for article in articles:
                        href = article.get_attribute('href')
                        if not href:
                            continue

                        # Make relative URLs absolute
                        if href.startswith('/'):
                            href = f"{self.base_url}{href}"
                        elif not href.startswith('http'):
                            continue

                        # Filter for article links (typically they contain year/month/day in URL)
                        # Also accept other article patterns
                        if (('/20' in href or '/article/' in href or '/news/' in href) and
                            'coindesk.com' in href and
                                href not in article_links):
                            article_links.append(href)

                except Exception as e:
                    print(f"Error extracting links: {e}")
                    traceback.print_exc()

                # If we still don't have links, try extracting from HTML directly
                if not article_links:
                    print("Trying to extract links from HTML directly...")
                    html_content = page.content()
                    soup = BeautifulSoup(html_content, 'html.parser')

                    for a_tag in soup.find_all('a', href=True):
                        href = a_tag['href']

                        # Make relative URLs absolute
                        if href.startswith('/'):
                            href = f"{self.base_url}{href}"
                        elif not href.startswith('http'):
                            continue

                        # Filter for article links
                        if (('/20' in href or '/article/' in href or '/news/' in href) and
                                'coindesk.com' in href):
                            article_links.append(href)

                # Remove duplicates while preserving order
                unique_links = []
                seen = set()
                for link in article_links:
                    if link not in seen:
                        seen.add(link)
                        unique_links.append(link)

                print(f"Found {len(unique_links)} unique article links")

                if len(unique_links) == 0:
                    print(
                        "Warning: No article links found. The site structure may have changed.")

                browser.close()
                return unique_links

        except Exception as e:
            print(f"An error occurred while scraping the homepage")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            traceback.print_exc()
            return []

    def scrape_article(self, article_url):
        """Scrape a single article page"""
        # Skip if already processed
        if article_url in self.processed_urls:
            print(f"Skipping already processed article: {article_url}")
            return None

        # Add random delay to avoid detection
        delay = random.uniform(self.delay_min, self.delay_max)
        print(f"Waiting {delay:.2f} seconds before accessing next article...")
        time.sleep(delay)

        try:
            with sync_playwright() as pw:
                browser_options = {}
                context_options = {
                    "user_agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    "bypass_csp": True,
                    "ignore_https_errors": True
                }

                # Add proxy settings if enabled
                if self.use_proxy:
                    context_options["proxy"] = {
                        "server": f"http://{self.proxy_url}",
                        "username": self.proxy_username,
                        "password": self.proxy_password
                    }

                browser = pw.chromium.launch(headless=True, **browser_options)
                context = browser.new_context(**context_options)

                # Add extra headers to appear more like a real browser
                context.set_extra_http_headers({
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate, br",
                    "DNT": "1",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                    "Sec-Fetch-User": "?1",
                    "Cache-Control": "max-age=0",
                })

                page = context.new_page()
                print(f"Accessing article: {article_url}")

                # Use a more lenient wait condition
                response = page.goto(
                    article_url, wait_until='domcontentloaded', timeout=30000)

                # Wait a bit for dynamic content to load
                page.wait_for_timeout(3000)

                if response is None:
                    print("No response received. Attempting to continue anyway...")
                elif response.status != 200:
                    print(f"Warning: Received status code {response.status}")
                    if response.status >= 400:
                        browser.close()
                        return None

                # Mark as processed
                self.processed_urls.add(article_url)

                # Extract article data
                article_data = {}

                # Use BeautifulSoup for easier parsing
                soup = BeautifulSoup(page.content(), 'html.parser')

                # Title
                title_element = soup.select_one('h1')
                article_data['title'] = title_element.text.strip(
                ) if title_element else "No title found"

                # URL
                article_data['url'] = article_url

                # Author
                author_element = soup.select_one('a[href*="/author/"]')
                if not author_element:
                    # Try alternative selectors
                    author_element = soup.select_one(
                        '.at-author__name') or soup.select_one('[data-test-id="byline"]')
                article_data['author'] = author_element.text.strip(
                ) if author_element else "Unknown author"

                # Date
                date_element = soup.select_one('time')
                if date_element and date_element.get('datetime'):
                    article_data['date'] = date_element.get('datetime')
                else:
                    # Try alternative date selectors
                    date_text = None
                    date_candidates = [
                        soup.select_one('.at-created'),
                        soup.select_one('.article-date'),
                        soup.select_one('[data-test-id="published-date"]')
                    ]
                    for candidate in date_candidates:
                        if candidate:
                            date_text = candidate.text.strip()
                            break
                    article_data['date'] = date_text if date_text else "Unknown date"

                # Category
                category_element = soup.select_one('a[href*="/tag/"]') or soup.select_one(
                    'a[href*="/markets/"]') or soup.select_one('a[href*="/business/"]')
                article_data['category'] = category_element.text.strip(
                ) if category_element else "Uncategorized"

                # Summary
                summary_element = soup.select_one('h2') or soup.select_one(
                    '.article-summary') or soup.select_one('.at-subhead')
                article_data['summary'] = summary_element.text.strip(
                ) if summary_element else "No summary available"

                # Content
                # Try multiple selectors for content
                content_elements = soup.select(
                    '.article-body p') or soup.select('.at-content-wrapper p') or soup.select('article p')

                if content_elements:
                    # Filter out empty paragraphs and join with spaces
                    content_text = ' '.join(
                        [p.text.strip() for p in content_elements if p.text.strip()])
                    article_data['content'] = content_text if content_text else "No content available"
                else:
                    article_data['content'] = "No content available"

                # Tags
                tag_elements = soup.select('a[href*="/tag/"]')
                article_data['tags'] = ', '.join(
                    [tag.text.strip() for tag in tag_elements]) if tag_elements else "No tags"

                browser.close()

                # Save to CSV
                self.save_to_csv(article_data)

                return article_data

        except Exception as e:
            print(f"An error occurred while scraping article {article_url}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            traceback.print_exc()
            return None

    def save_to_csv(self, article_data):
        """Save article data to CSV file"""
        try:
            with open(self.output_csv, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    article_data.get('title', ''),
                    article_data.get('url', ''),
                    article_data.get('author', ''),
                    article_data.get('date', ''),
                    article_data.get('category', ''),
                    article_data.get('summary', ''),
                    article_data.get('content', ''),
                    article_data.get('tags', '')
                ])
            print(
                f"Saved article: {article_data.get('title', 'Unknown title')}")
            return True
        except Exception as e:
            print(f"Error saving to CSV: {e}")
            return False


def main():
    # Set use_proxy to False to disable proxy
    # Set delay_min and delay_max to control the random delay between requests
    scraper = CoindeskScraper(use_proxy=False, delay_min=2, delay_max=5)

    # Get article links from homepage
    article_links = scraper.scrape_homepage()

    # Limit to first 20 articles to avoid overloading
    article_links = article_links[:20]

    # Scrape each article
    successful = 0
    for i, link in enumerate(article_links):
        print(f"\nScraping article {i+1} of {len(article_links)}...")
        result = scraper.scrape_article(link)
        if result:
            successful += 1

    print(
        f"\nScraped {successful} out of {len(article_links)} articles successfully")
    print(f"Results saved to {scraper.output_csv}")


if __name__ == "__main__":
    main()
