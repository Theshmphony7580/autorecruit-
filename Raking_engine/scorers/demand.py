import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.safe_extract import safe_float

def get_demand_score(candidate: dict) -> float:
    """
    Measures how much the market WANTS this candidate.
    Uses 4 signals: profile views, search appearances, recruiter saves, and active applications.
    """
    signals = candidate.get('redrob_signals', {})
    if not signals:
        return 0.0

    views    = safe_float(signals.get('profile_views_received_30d', 0))
    searches = safe_float(signals.get('search_appearance_30d', 0))
    saves    = safe_float(signals.get('saved_by_recruiters_30d', 0))
    apps     = safe_float(signals.get('applications_submitted_30d', 0))

    # Normalize against typical max values in a 100k dataset
    v_norm    = min(1.0, views    / 200.0)
    s_norm    = min(1.0, searches / 100.0)
    save_norm = min(1.0, saves    / 20.0)
    apps_norm = min(1.0, apps     / 10.0)

    return float((v_norm + s_norm + save_norm + apps_norm) / 4.0)
