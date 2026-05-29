"""
scrape_books.py
Data Collection Module.
Scrapes human-written text excerpts from Project Gutenberg books.
"""
import requests
from bs4 import BeautifulSoup
import re
import random
import time

def scrape_books(target_count, min_length=150, max_length=600):
    # Seed books: Pride and Prejudice, Frankenstein, Sherlock Holmes, Moby Dick, Alice in Wonderland
    seed_ids = [1342, 84, 1661, 2701, 11]
    
    queue = seed_ids.copy()
    visited = set(seed_ids)
    results = []
    
    print(f"Starting Books scraping (target: {target_count} paragraphs) via catalog traversal...")
    
    while len(results) < target_count and queue:
        # Pick a random ID from the front of the queue
        idx = random.randint(0, min(len(queue)-1, 20))
        book_id = queue.pop(idx)
        
        metadata_url = f"https://www.gutenberg.org/ebooks/{book_id}"
        text_url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
        
        try:
            # 1. Fetch metadata to find related books (spidering)
            headers = {'User-Agent': 'AISlopDetectorBot/1.0 (contact@example.com)'}
            meta_response = requests.get(metadata_url, headers=headers, timeout=10)
            if meta_response.status_code == 200:
                soup = BeautifulSoup(meta_response.content, 'html.parser')
                links = soup.find_all('a', href=re.compile(r"^/ebooks/\d+$"))
                if links:
                    new_links = random.sample(links, min(10, len(links)))
                    for link in new_links:
                        match = re.search(r"/ebooks/(\d+)", link['href'])
                        if match:
                            new_id = int(match.group(1))
                            if new_id not in visited:
                                visited.add(new_id)
                                queue.append(new_id)
            
            # 2. Fetch the actual book text
            text_response = requests.get(text_url, headers=headers, timeout=10)
            if text_response.status_code != 200:
                print(f"Books HTTP {text_response.status_code} for {text_url}")
                continue
                
            text = text_response.text
            # Robust split for paragraphs (handles \r\n\r\n and \n\n)
            paragraphs = re.split(r'\r?\n\r?\n', text)
            
            valid_paragraphs = []
            for p in paragraphs:
                # Clean up newlines within the paragraph
                p_text = re.sub(r'\s+', ' ', p).strip()
                
                # Ignore Project Gutenberg boilerplate and license text
                if "Project Gutenberg" in p_text or "PROJECT GUTENBERG" in p_text or "www.gutenberg.org" in p_text:
                    continue
                    
                if min_length <= len(p_text) <= max_length:
                    valid_paragraphs.append(p_text)
            
            if valid_paragraphs:
                # Take up to 5 paragraphs per book to ensure diversity
                sample_size = min(5, len(valid_paragraphs))
                selected = random.sample(valid_paragraphs, sample_size)
                
                for p_text in selected:
                    if len(results) < target_count:
                        results.append({
                            "text": p_text,
                            "source": f"gutenberg_pg{book_id}",
                            "genre": "creative"
                        })
            
            time.sleep(0.5)
            
        except Exception as e:
            time.sleep(1)
            
    return results

if __name__ == "__main__":
    res = scrape_books(10)
    for r in res:
        print(r['text'][:50], "...")
