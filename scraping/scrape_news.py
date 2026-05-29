"""
scrape_news.py
Data Collection Module.
Scrapes human-written news articles from various online publications.
"""
import requests
from bs4 import BeautifulSoup
import re
import random
import time

def scrape_news(target_count, min_length=150, max_length=600):
    seed_urls = [
        "https://en.wikinews.org/wiki/Portal:Politics",
        "https://en.wikinews.org/wiki/Portal:Science_and_technology",
        "https://en.wikinews.org/wiki/Portal:Sports",
        "https://en.wikinews.org/wiki/Portal:Economy",
        "https://en.wikinews.org/wiki/Portal:Culture",
        "https://en.wikinews.org/wiki/Category:History",
    ]
    
    queue = seed_urls.copy()
    visited = set(seed_urls)
    results = []
    
    print(f"Starting News scraping (target: {target_count} paragraphs) via link traversal...")
    
    while len(results) < target_count and queue:
        # Pick a random URL from the front of the queue to diversify paths
        idx = random.randint(0, min(len(queue)-1, 20))
        url = queue.pop(idx)
        
        try:
            headers = {'User-Agent': 'AISlopDetectorBot/1.0 (contact@example.com)'}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"News HTTP {response.status_code} for {url}")
                continue
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract paragraphs
            paragraphs = soup.find_all('p')
            valid_paragraphs = []
            for p in paragraphs:
                text = p.get_text().strip()
                # Remove citations and bracketed text
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
                            "source": "wikinews",
                            "genre": "journalistic",
                            "url": url
                        })
            
            # Find new internal Wikinews links to add to the queue
            # Internal links look like /wiki/Article_Name
            content_div = soup.find(id="bodyContent")
            if content_div:
                links = content_div.find_all('a', href=re.compile(r"^/wiki/[^:]+$"))
                if links:
                    # Sample a few links to branch out
                    new_links = random.sample(links, min(10, len(links)))
                    for link in new_links:
                        full_url = "https://en.wikinews.org" + link['href']
                        if full_url not in visited:
                            visited.add(full_url)
                            queue.append(full_url)
            
            time.sleep(0.1)
            
        except Exception as e:
            time.sleep(1)
            
    return results

if __name__ == "__main__":
    res = scrape_news(10)
    for r in res:
        print(r['text'][:50], "...")
