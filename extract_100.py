import random

def extract_candidates():
    input_file = r'F:\autorecruit-\Raking_engine\top_100_candidates.jsonl'
    output_file = r'f:\autorecruit-\sample_100.jsonl'
    
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"Found {len(lines)} candidates. Sampling 100...")
    random.seed(42)  # For reproducibility
    sampled_lines = random.sample(lines, 100)
    
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(sampled_lines)
        
    print("Done!")

if __name__ == "__main__":
    extract_candidates()
