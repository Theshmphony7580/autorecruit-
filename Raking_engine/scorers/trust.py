def get_trust_score(candidate: dict) -> float:
    signals = candidate.get('redrob_signals', {})
    if not signals:
        return 0.0
        
    response_rate = signals.get('recruiter_response_rate', 0.0)
    completion = signals.get('profile_completeness_score', 0) / 100.0
    
    # Trust is a mix of how complete their profile is and how well they respond
    return float((response_rate + completion) / 2.0)
