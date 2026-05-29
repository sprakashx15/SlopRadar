"""
generate_ai_slop_groq.py
Data Generation Pipeline.
Reads human-written text (scraped from Wikipedia) and uses the Groq API (LLaMA 3) 
to rewrite it into 'AI Slop'. It processes data in chunks and manages rate limits 
automatically, rotating through multiple API keys in the .env file.
"""
import os
import json
import time
import random
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from pydantic import BaseModel
from typing import List

class ResponseSchema(BaseModel):
    items: List[str]

def load_env():
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
    except Exception:
        pass

# Initialize our clients for round-robin
def get_clients():
    load_env()
    # Read GROQ_API_KEYS from .env (comma separated)
    keys_str = os.environ.get("GROQ_API_KEYS")
    if not keys_str:
        print("[ERROR] GROQ_API_KEYS not found in .env file.")
        return []
    
    keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    clients = [OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1") for key in keys]
    return clients

clients = get_clients()
current_client_idx = 0

def get_next_client():
    global current_client_idx
    client = clients[current_client_idx]
    current_client_idx = (current_client_idx + 1) % len(clients)
    return client

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=20))
def call_groq(prompt):
    client = get_next_client()
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    
    # We ask for a JSON object with a single key 'items'
    content = response.choices[0].message.content
    try:
        return json.loads(content)["items"]
    except Exception as e:
        raise Exception(f"Failed to parse json: {content}")

def main():
    if not clients:
        return
        
    print(f"Loaded {len(clients)} Groq API keys.")

    input_file = os.path.join(os.path.dirname(__file__), "..", "scrapes", "human_text.jsonl")
    output_file = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.jsonl")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    processed_human_texts = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    datum = json.loads(line)
                    if datum.get("label") == 0:
                        processed_human_texts.add(datum["text"])
                except:
                    pass
    
    all_data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                datum = json.loads(line)
                if datum["text"] not in processed_human_texts:
                    all_data.append(datum)
            except:
                pass
                
    # Process the entire dataset now that it's manually reduced
    # all_data = all_data[:10000]
    
    print(f"Total rows left to process: {len(all_data)}")
    if len(all_data) == 0:
        print("Everything is processed!")
        return
        
    CHUNK_SIZE = 10
    
    for i in tqdm(range(0, len(all_data), CHUNK_SIZE)):
        chunk = all_data[i:i+CHUNK_SIZE]
        
        # PASS 1: Summarize
        summary_prompt = "I will provide a list of paragraphs. For EACH paragraph, write a summary containing bullet points outlining its core facts. You MUST return exactly a JSON object containing a SINGLE key called 'items'. The value of 'items' must be a JSON array of strings with exactly the same number of items as the input. The string at index 0 should be the combined bullet points for Paragraph 1, index 1 for Paragraph 2, etc. Do NOT split a single paragraph's bullet points into multiple array items.\n\n"
        for idx, item in enumerate(chunk):
            summary_prompt += f"Paragraph {idx+1}:\n{item['text']}\n\n"
            
        try:
            summaries = call_groq(summary_prompt)
        except Exception as e:
            print(f"\nSkipping chunk due to API error: {e}")
            time.sleep(10)
            continue
            
        if len(summaries) != len(chunk):
            print(f"\nWarning: Model returned {len(summaries)} summaries instead of {len(chunk)}. Skipping chunk to avoid data corruption.")
            continue
            
        # PASS 2: Rewrite
        rewrite_prompt = "I will provide a list of summaries and requested styles. Rewrite each summary into a single cohesive paragraph. You MUST return exactly a JSON object containing a SINGLE key called 'items'. The value of 'items' must be a JSON array of strings with exactly the same number of items as the input. The string at index 0 should be the rewritten paragraph for Item 1, etc.\n\n"
        for idx, (item, summary) in enumerate(zip(chunk, summaries)):
            rewrite_prompt += f"Item {idx+1}:\nStyle: {item.get('genre', 'general')} (Source: {item.get('source', 'unknown')})\nKey points: {summary}\n\n"
            
        try:
            rewrites = call_groq(rewrite_prompt)
        except Exception as e:
            print(f"\nSkipping chunk due to API error: {e}")
            time.sleep(10)
            continue
            
        if len(rewrites) != len(chunk):
            print(f"\nWarning: Model returned {len(rewrites)} rewrites instead of {len(chunk)}. Skipping chunk.")
            continue
            
        # SAVE
        with open(output_file, "a", encoding="utf-8") as f:
            for original, rewrite in zip(chunk, rewrites):
                # Write Human (Label 0)
                human_row = {
                    "text": original["text"],
                    "label": 0,
                    "source": original.get("source", "unknown"),
                    "genre": original.get("genre", "general")
                }
                f.write(json.dumps(human_row, ensure_ascii=False) + "\n")
                
                # Write AI (Label 1)
                ai_row = {
                    "text": rewrite,
                    "label": 1,
                    "source": original.get("source", "unknown"),
                    "genre": original.get("genre", "general")
                }
                f.write(json.dumps(ai_row, ensure_ascii=False) + "\n")
                
        # Groq is fast, but we must respect the Total Tokens Per Minute limit (approx 14k)
        time.sleep(8)

if __name__ == "__main__":
    main()
