import pandas as pd
import json
import os
import sys

def extract_top_100_json():
    # Setup paths
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.paths import DEFAULT_CANDIDATES_PATH, DEFAULT_OUTPUT_PATH
    
    submission_path = DEFAULT_OUTPUT_PATH
    candidates_path = DEFAULT_CANDIDATES_PATH
    output_path = os.path.join(os.path.dirname(DEFAULT_OUTPUT_PATH), 'top_100_candidates.jsonl')
    
    if not os.path.exists(submission_path):
        print(f"Error: Could not find {submission_path}")
        return

    # 1. Load the 100 winning candidate IDs
    print("Loading Top-100 IDs from submission.csv...")
    df = pd.read_csv(submission_path)
    top_ids = set(df['candidate_id'])
    
    print(f"Found {len(top_ids)} unique IDs to extract.")
    
    # 2. Stream through the massive 100k JSONL file and pull only the winners
    print(f"Scanning main dataset (this may take a few seconds)...")
    extracted_count = 0
    
    with open(candidates_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
         
        for line in infile:
            if not line.strip(): continue
            try:
                cand = json.loads(line)
                if cand.get('candidate_id') in top_ids:
                    # Write the exact original JSON line into the new file
                    outfile.write(line)
                    extracted_count += 1
                    
                    # Optimization: If we found all 100, we can stop reading!
                    if extracted_count == len(top_ids):
                        break
            except Exception as e:
                pass

    print(f"============================================================")
    print(f" SUCCESS! Extracted {extracted_count} full profiles.")
    print(f" Saved to: {output_path}")
    print(f"============================================================")

if __name__ == '__main__':
    extract_top_100_json()
