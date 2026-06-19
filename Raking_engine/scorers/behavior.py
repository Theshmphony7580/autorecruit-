def get_behavior_score(candidate: dict) -> float:
    signals = candidate.get('redrob_signals', {})
    if not signals:
        return 0.0
        
    open_to_work = 1.0 if signals.get('open_to_work_flag', False) else 0.0
    apps = min(1.0, signals.get('applications_submitted_30d', 0) / 10.0)
    
    return float((open_to_work * 0.7) + (apps * 0.3))
