def get_demand_score(candidate: dict) -> float:
    signals = candidate.get('redrob_signals', {})
    if not signals:
        return 0.0
        
    views = signals.get('profile_views_received_30d', 0)
    searches = signals.get('search_appearance_30d', 0)
    saves = signals.get('saved_by_recruiters_30d', 0)
    
    # Heuristic normalization (from 0 to 1) based on typical maximums in the dataset
    v_norm = min(1.0, views / 200.0)
    s_norm = min(1.0, searches / 100.0)
    save_norm = min(1.0, saves / 20.0)
    
    return float((v_norm + s_norm + save_norm) / 3.0)
