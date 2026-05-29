"""
run_gathering.py
Data Collection Coordinator.
Orchestrates the execution of all individual scraping scripts to collect 
human-written text from multiple sources.
"""
import json
import os
import argparse
from scrape_wikipedia import scrape_wikipedia
from scrape_news import scrape_news
from scrape_books import scrape_books
from scrape_wikibooks import scrape_wikibooks

def main():
    parser = argparse.ArgumentParser(description="Gather human text for AI Slop Detector.")
    parser.add_argument('--test', action='store_true', help="Run a small test batch (10 items per genre)")
    args = parser.parse_args()

    # Targets for the final dataset
    if args.test:
        TARGET_WIKI = 10
        TARGET_NEWS = 10
        TARGET_BOOKS = 10
        TARGET_WIKIBOOKS = 10
    else:
        TARGET_WIKI = 5000
        TARGET_NEWS = 2500
        TARGET_BOOKS = 2000
        TARGET_WIKIBOOKS = 500
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scrapes')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'human_text.jsonl')
    
    all_data = []
    
    print(f"--- Phase 1: Data Gathering (Test mode: {args.test}) ---")
    
    wiki_data = scrape_wikipedia(TARGET_WIKI)
    all_data.extend(wiki_data)
    
    news_data = scrape_news(TARGET_NEWS)
    all_data.extend(news_data)
    
    books_data = scrape_books(TARGET_BOOKS)
    all_data.extend(books_data)
    
    wikibooks_data = scrape_wikibooks(TARGET_WIKIBOOKS)
    all_data.extend(wikibooks_data)
    
    print(f"Writing {len(all_data)} paragraphs to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item) + '\n')
            
    print("Data gathering complete!")
    print(f"Dataset stored at: {output_file}")

if __name__ == "__main__":
    main()
