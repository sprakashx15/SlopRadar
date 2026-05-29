"""
scrape_wikibooks.py
Data Collection Module.
Scrapes human-written educational text from Wikibooks.
"""

import requests
from bs4 import BeautifulSoup
import re
import random
import time

def scrape_wikibooks(target_count, min_length=200, max_length=600):
    seed_urls = [
        "https://en.wikibooks.org/wiki/Subject:Computing",
        "https://en.wikibooks.org/wiki/Subject:Science",
        "https://en.wikibooks.org/wiki/Subject:Humanities",
        "https://en.wikibooks.org/wiki/Subject:Mathematics",
        "https://en.wikibooks.org/wiki/Subject:Engineering"
    ]
    
    queue = seed_urls.copy()
    visited = set(seed_urls)
    results = []
    
    print(f"Starting Wikibooks scraping (target: {target_count} paragraphs) via link traversal...")
    
    while len(results) < target_count and queue:
        # Pick a random URL from the front of the queue to diversify paths
        idx = random.randint(0, min(len(queue)-1, 20))
        url = queue.pop(idx)
        
        try:
            headers = {'User-Agent': 'AISlopDetectorBot/1.0 (contact@example.com)'}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"Wikibooks HTTP {response.status_code} for {url}")
                continue
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract paragraphs
            paragraphs = soup.find_all('p')
            valid_paragraphs = []
            for p in paragraphs:
                text = p.get_text().strip()
                # Remove citations like [1], [a]
                text = re.sub(r'\[\w+\]', '', text)
                if min_length <= len(text) <= max_length:
                    valid_paragraphs.append(text)
            
            if valid_paragraphs:
                # Randomly select up to 3 paragraphs per page
                sample_size = min(3, len(valid_paragraphs))
                selected = random.sample(valid_paragraphs, sample_size)
                
                for text in selected:
                    if len(results) < target_count:
                        results.append({
                            "text": text,
                            "source": "wikibooks",
                            "genre": "educational",
                            "url": url
                        })
    
            
            
            # Find new internal Wikibooks links to add to the queue
            content_div = soup.find(id="bodyContent")
            if content_div:
                links = content_div.find_all('a', href=re.compile(r"^/wiki/[^:]+$"))
                if links:
                    # Sample a few links to branch out
                    new_links = random.sample(links, min(10, len(links)))
                    for link in new_links:
                        full_url = "https://en.wikibooks.org" + link['href']
                        if full_url not in visited:
                            visited.add(full_url)
                            queue.append(full_url)
            
            # Be polite
            time.sleep(0.1)
            
        except Exception as e:
            time.sleep(1)
            
    return results

if __name__ == "__main__":
    res = scrape_wikibooks(10)
    for r in res:
        print(r['text'][:50], "...")
