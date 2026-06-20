import pandas as pd
import json

def validate_submission(df: pd.DataFrame, candidates_path: str):
    expected_cols = ['candidate_id', 'rank', 'score', 'reasoning']
    assert list(df.columns) == expected_cols, f"Missing or wrong columns: {list(df.columns)}"
    
    valid_ids = set()
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                cand = json.loads(line.strip())
                valid_ids.add(cand['candidate_id'])
            except:
                pass

    expected_len = min(100, len(valid_ids))
    assert len(df) == expected_len, f"Expected {expected_len} rows, got {len(df)}"
    
    expected_ranks = set(range(1, expected_len + 1))
    actual_ranks = set(df['rank'].tolist())
    assert actual_ranks == expected_ranks, f"Ranks not 1-{expected_len} unique"
    
    assert df['score'].is_monotonic_decreasing, "Scores not non-increasing"
    
    invalid_ids = set(df['candidate_id']) - valid_ids
    assert len(invalid_ids) == 0, f"Invalid candidate_ids: {invalid_ids}"
    
    print("✓ Submission validation passed!")
