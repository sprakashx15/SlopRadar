"""
scrape_wikipedia.py
Data Collection Module.
Crawls and scrapes human-written paragraphs from random Wikipedia articles.
This serves as the primary dataset for the Human class.
"""
import requests
from bs4 import BeautifulSoup
import re
import random
import time

def scrape_wikipedia(target_count, min_length=200, max_length=600):
    seed_urls = [
        "https://en.wikipedia.org/wiki/World_War_II",
        "https://en.wikipedia.org/wiki/Quantum_mechanics",
        "https://en.wikipedia.org/wiki/Renaissance",
        "https://en.wikipedia.org/wiki/Solar_System"
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Earth",
        "https://en.wikipedia.org/wiki/Psychology",
        "https://en.wikipedia.org/wiki/Philosophy",
        "https://en.wikipedia.org/wiki/Lionel_Messi",
        "https://en.wikipedia.org/wiki/Buddhism",
        "https://en.wikipedia.org/wiki/Sun",
        "https://en.wikipedia.org/wiki/Elon_Musk",
        "https://en.wikipedia.org/wiki/Nuclear_weapon"
        "https://en.wikipedia.org/wiki/Special:Random"

    ]
    
    queue = seed_urls.copy()
    visited = set(seed_urls)
    results = []
    
    print(f"Starting Wikipedia scraping (target: {target_count} paragraphs) via link traversal...")
    
    while len(results) < target_count and queue:
        # Pick a random URL from the front of the queue to diversify paths
        idx = random.randint(0, min(len(queue)-1, 20))
        url = queue.pop(idx)
        
        try:
            headers = {'User-Agent': 'AISlopDetectorBot/1.0 (contact@example.com)'}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"Wikipedia HTTP {response.status_code} for {url}")
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
                            "source": "wikipedia",
                            "genre": "encyclopedic",
                            "url": url
                        })
            
            # Find new internal article links to add to the queue
            # Internal links look like /wiki/Article_Name and don't contain a colon (like Special:, File:, Category:)
            content_div = soup.find(id="bodyContent")
            if content_div:
                links = content_div.find_all('a', href=re.compile(r"^/wiki/[^:]+$"))
                if links:
                    # Sample a few links to branch out without exploding the queue
                    new_links = random.sample(links, min(10, len(links)))
                    for link in new_links:
                        full_url = "https://en.wikipedia.org" + link['href']
                        if full_url not in visited:
                            visited.add(full_url)
                            queue.append(full_url)
            
            # Be polite to the server
            time.sleep(0.1)
            
        except Exception as e:
            # Silently handle transient network errors
            time.sleep(1)
            
    return results

if __name__ == "__main__":
    res = scrape_wikipedia(10)
    for r in res:
        print(r['text'][:50], "...")
