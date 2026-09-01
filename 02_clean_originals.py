import pandas as pd
import os

# Explicit list of your 5 CSV files
model_files = [
    "chatgpt_300samples.csv",
    "claude_300samples.csv",
    "gemini_300samples.csv",
    "gpt4_300samples.csv",
    "grok_300samples.csv"
]

search_dirs = [
    r"D:\Research\RAID_dataset",
    r"D:\Research\AI detection module"
]

# Common names for text output columns in NLP benchmarks
candidate_text_cols = [
    'original_text', 'text', 'generation', 'generated_text', 
    'content', 'output', 'response', 'sample'
]

cleaned_dfs = []

for filename in model_files:
    found_path = None
    for d in search_dirs:
        possible = os.path.join(d, filename)
        if os.path.exists(possible):
            found_path = possible
            break

    if not found_path:
        print(f"Warning: Could not find {filename}")
        continue

    try:
        df = pd.read_csv(found_path)
        # Normalize column header names
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Determine the text column
        text_col = None
        for col in candidate_text_cols:
            if col in df.columns:
                text_col = col
                break
        
        # Fallback: find the column containing the longest string text
        if not text_col:
            string_cols = df.select_dtypes(include=['object']).columns
            if len(string_cols) > 0:
                text_col = max(string_cols, key=lambda c: df[c].astype(str).str.len().mean())

        if text_col:
            df.rename(columns={text_col: 'original_text'}, inplace=True)
            
            # Ensure model column exists
            if 'model' not in df.columns:
                df['model'] = filename.split('_')[0]

            # Clean blank or short generations (< 50 words)
            df.dropna(subset=['original_text'], inplace=True)
            df['original_text'] = df['original_text'].astype(str).str.strip()
            df = df[df['original_text'] != ""]
            df['word_count'] = df['original_text'].apply(lambda x: len(x.split()))
            valid_df = df[df['word_count'] >= 50].copy()

            cleaned_dfs.append(valid_df)
            print(f"Processed {filename}: Found text in column '{text_col}' -> {len(valid_df)} valid samples.")
        else:
            print(f"Error: Could not identify text column in {filename}. Columns found: {list(df.columns)}")

    except Exception as e:
        print(f"Failed to process {filename}: {e}")

if cleaned_dfs:
    master_df = pd.concat(cleaned_dfs, ignore_index=True)
    master_df.drop_duplicates(subset=['original_text'], inplace=True)

    print("\n=== Master Dataset Summary ===")
    print(master_df.groupby('model').size())
    print(f"\nTotal clean original samples: {len(master_df)}")

    output_path = r"D:\Research\AI detection module\master_clean_originals.csv"
    master_df.to_csv(output_path, index=False)
    print(f"\nSaved master file to: {output_path}")
else:
    print("No data processed.")