import pandas as pd
import requests
import time
import os

# Replace with your OpenRouter API key
OPENROUTER_API_KEY = "******************"/////////enter ur api key

# Choose "meta-llama/llama-3.3-70b-instruct:free" for free tier
# Or "meta-llama/llama-3.3-70b-instruct" for fast execution (<$0.40 total cost)
MODEL_ID = "meta-llama/llama-3.3-70b-instruct:free"

INPUT_FILE = r"D:\Research\AI detection module\master_clean_originals.csv"
OUTPUT_FILE = r"D:\Research\AI detection module\master_rewritten_samples.csv"
CHECKPOINT_FILE = r"D:\Research\AI detection module\rewritten_checkpoint.csv"

LIGHT_PROMPT = "Rewrite the following text in a natural academic tone. Change sentence structures and word choices meaningfully, but preserve all factual content and meaning exactly. Do not add new information."
HEAVY_PROMPT = "Completely rephrase the following text. Substantially alter sentence structures, vocabulary, and phrasing throughout the entire passage while preserving the core meaning and maintaining academic coherence. The result should read as if written by a different author."

def call_openrouter(prompt, text, retries=5):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "Forensic Stylometry Research"
    }
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ],
        "temperature": 0.7
    }
    
    for attempt in range(retries):
        try:
            res = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=60)
            if res.status_code == 200:
                return res.json()['choices'][0]['message']['content'].strip()
            elif res.status_code == 429:
                wait_time = (attempt + 1) * 10
                print(f"Rate limit hit (429). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"API Error {res.status_code}: {res.text}")
                return None
        except Exception as e:
            print(f"Connection Exception: {e}")
            time.sleep(5)
            
    return None

processed_ids = set()
results_list = []

# Resume from existing non-empty checkpoint
if os.path.exists(CHECKPOINT_FILE) and os.path.getsize(CHECKPOINT_FILE) > 0:
    try:
        df_ckpt = pd.read_csv(CHECKPOINT_FILE)
        if not df_ckpt.empty and 'original_text' in df_ckpt.columns:
            results_list = df_ckpt.to_dict('records')
            processed_ids = set(df_ckpt['original_text'])
            print(f"Resuming from checkpoint: {len(results_list)} samples completed.")
    except Exception as e:
        print(f"Starting fresh checkpoint: {e}")

df_orig = pd.read_csv(INPUT_FILE)
print(f"Total target samples: {len(df_orig)}")

for idx, row in df_orig.iterrows():
    orig_text = row['original_text']
    if orig_text in processed_ids:
        continue

    # 1. Generate Light Rewrite
    light = call_openrouter(LIGHT_PROMPT, orig_text)
    time.sleep(2.0) # Pace requests to stay within OpenRouter rate limits

    # 2. Generate Heavy Rewrite
    heavy = call_openrouter(HEAVY_PROMPT, orig_text)
    time.sleep(2.0)

    if light and heavy:
        results_list.append({
            'model': row.get('model'),
            'domain': row.get('domain'),
            'original_text': orig_text,
            'light_rewrite': light,
            'heavy_rewrite': heavy
        })
        processed_ids.add(orig_text)
        print(f"[{len(results_list)}/{len(df_orig)}] Sample rewritten successfully.")

    # Save checkpoint every 5 completed samples
    if len(results_list) % 5 == 0 and len(results_list) > 0:
        pd.DataFrame(results_list).to_csv(CHECKPOINT_FILE, index=False)

pd.DataFrame(results_list).to_csv(OUTPUT_FILE, index=False)
print(f"\nRewriting complete! All samples saved to: {OUTPUT_FILE}")