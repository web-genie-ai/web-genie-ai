import bittensor as bt
import nltk
import random

from bs4 import BeautifulSoup, Tag, NavigableString
from collections import Counter
from duckduckgo_search import DDGS
from nltk.corpus import brown
from playwright.async_api import async_playwright
from urllib.parse import urljoin
from typing import Optional

from webgenie.datasets.dataset import Dataset, DatasetEntry
from webgenie.constants import GROUND_TRUTH_HTML_LOAD_TIME

class RandomWebsiteDataset(Dataset):
    def __init__(self , **kwargs):
        nltk.download("brown", quiet=True)
        words = brown.words()
        word_freq = Counter(word.lower() for word in words)
        most_common = word_freq.most_common(25000)
        common_words = [word for word, _ in most_common]
        self.english_words = common_words

    async def get_random_website_url(self, retries: int = 3) -> Optional[str]:
        try:
            ddg = DDGS()
            for _ in range(retries):
                random_words = " ".join(random.sample(self.english_words, 5))
                results = list(ddg.text(random_words))
                if results:
                    website_url = random.choice(results)["href"]
                    return website_url
                    
        except Exception as ex:
            print(f"Failed to get search results from DuckDuckGo: {ex}")
        return None

    async def get_rendered_html(self, url):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url)
            # Wait for 10 seconds to ensure content loads
            await page.wait_for_timeout(GROUND_TRUTH_HTML_LOAD_TIME)
            rendered_html = await page.content()  # Get the rendered HTML
            await browser.close()

            # Parse the HTML with BeautifulSoup
            soup = BeautifulSoup(rendered_html, 'html.parser')

            # Attributes that need to be absolute
            attributes = ['href', 'src', 'srcset']

            # Find all elements with 'href', 'src', or 'srcset' attributes
            for attr in attributes:
                for element in soup.find_all(attrs={attr: True}):
                    original_attr = element[attr]
                    # Handle 'srcset' differently because it can contain multiple URLs
                    if attr == 'srcset':
                        new_urls = []
                        parts = original_attr.split(',')
                        for part in parts:
                            # Split on whitespace and check if there is a descriptor
                            pieces = part.strip().split(maxsplit=1)
                            if len(pieces) == 2:
                                url_part, descriptor = pieces
                            else:
                                url_part = pieces[0]
                                descriptor = ''

                            new_url = urljoin(url, url_part.strip())
                            if descriptor:
                                new_urls.append(f"{new_url} {descriptor}")
                            else:
                                new_urls.append(new_url)

                        element[attr] = ', '.join(new_urls)
                    else:
                        element[attr] = urljoin(url, original_attr)

            # Return the modified HTML as a string
            return str(soup)
        
    async def shorten_html(self, html: str, max_children: int = 20, max_text_length: int = 400)->str:
        soup = BeautifulSoup(html, "html.parser")
        def traverse(node):
            # If it’s a tag, we might need to limit its children
            if isinstance(node, Tag):
                # If node has too many children, remove the extras
                if len(node.contents) > max_children:
                    # Keep only the first max_children
                    node.contents = node.contents[:max_children]
                
                # Recurse on each child
                for child in node.contents:
                    traverse(child)

            # If it’s a text node, shorten if needed
            elif isinstance(node, NavigableString):
                text_str = str(node)
                if len(text_str) > max_text_length:
                    shortened = text_str[:max_text_length] + "..."
                    node.replace_with(shortened)

        # Start traversal from the root soup element (html, body, etc.)
        traverse(soup)

        return str(soup)



    async def generate_context(self)->DatasetEntry:
        try:
            bt.logging.info("Generating Random Website context")
            website_url = await self.get_random_website_url()
            if website_url is None:
                raise Exception("Failed to get a valid website URL")
            bt.logging.debug(f"Generated website URL: {website_url}")
            html = await self.get_rendered_html(website_url)
            html = await self.shorten_html(html)
            return DatasetEntry(
                src="random_website",
                topic="random_website",
                ground_truth_html=html,
                prompt="",
                base64_image="",
            )
        except Exception as e:
            bt.logging.error(f"Error in generate_context: {e}")
            raise e
            
