import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from filters.hard_filters import HardFilter

def test_hard_filter_rejects_suspicious():
    behavior_df = pd.DataFrame({
        'candidate_id': ['CAND_001', 'CAND_002'],
        'honey_pot_labels': ['suspicious', 'trusted'],
        'honeypot_score': [3, 0],
    }).set_index('candidate_id')
    
    hf = HardFilter(behavior_df)
    
    assert hf.is_rejected({'candidate_id': 'CAND_001'}) == True
    assert hf.is_rejected({'candidate_id': 'CAND_002'}) == False
