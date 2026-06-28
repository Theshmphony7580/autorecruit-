import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.safe_extract import safe_float

def get_demand_score(candidate: dict) -> float:
    """
    Measures how much the market WANTS this candidate.
    Prioritizes high-intent inbound pull (saves, searches) over outbound push (applications),
    and incorporates proven market desirability (historical offer track record).
    """
    signals = candidate.get('redrob_signals', {})
    if not signals:
        return 0.0

    views      = safe_float(signals.get('profile_views_received_30d', 0))
    searches   = safe_float(signals.get('search_appearance_30d', 0))
    saves      = safe_float(signals.get('saved_by_recruiters_30d', 0))
    apps       = safe_float(signals.get('applications_submitted_30d', 0))
    offer_rate = safe_float(signals.get('offer_acceptance_rate', -1.0), -1.0)
    connections = safe_float(signals.get('connection_count', 0))

    # Normalize against typical ceiling values in a 100k dataset
    v_norm      = min(1.0, views    / 200.0)
    s_norm      = min(1.0, searches / 100.0)
    save_norm   = min(1.0, saves    / 20.0)
    apps_norm   = min(1.0, apps     / 10.0)
    conn_norm   = min(1.0, connections / 500.0)
    
    # Offer track record: if offer_acceptance_rate != -1, candidate has proven offer history
    offer_norm  = 1.0 if offer_rate >= 0.0 else 0.0

    # Weighted demand model prioritizing market pull & external desirability:
    # 35% Recruiter Saves (highest intent pull)
    # 25% Search Appearances (top-of-funnel pull)
    # 15% Profile Views (mid-funnel attraction)
    # 10% Historical Offer Track Record (proven market validation)
    # 10% Active Applications (outbound proactivity)
    # 5% Network Reach (connections)
    score = (
        save_norm  * 0.35 +
        s_norm     * 0.25 +
        v_norm     * 0.15 +
        offer_norm * 0.10 +
        apps_norm  * 0.10 +
        conn_norm  * 0.05
    )

    return float(score)
