import pandas as pd
import json

def validate_submission(df: pd.DataFrame, candidates_path: str):
    expected_cols = ['candidate_id', 'rank', 'score', 'reasoning']
    assert list(df.columns) == expected_cols, f"Missing or wrong columns: {list(df.columns)}"
    
    assert len(df) == 100, f"Expected 100 rows, got {len(df)}"
    
    expected_ranks = set(range(1, 101))
    actual_ranks = set(df['rank'].tolist())
    assert actual_ranks == expected_ranks, "Ranks not 1-100 unique"
    
    assert df['score'].is_monotonic_decreasing, "Scores not non-increasing"
    
    valid_ids = set()
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                cand = json.loads(line.strip())
                valid_ids.add(cand['candidate_id'])
            except:
                pass
                
    invalid_ids = set(df['candidate_id']) - valid_ids
    assert len(invalid_ids) == 0, f"Invalid candidate_ids: {invalid_ids}"
    
    print("✓ Submission validation passed!")
