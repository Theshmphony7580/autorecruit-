def normalize_experience(years: float, min_years: int = 5, max_years: int = 9, decay: int = 5) -> float:
    """Normalize years of experience to [0, 1]."""
    if not isinstance(years, (int, float)):
        return 0.0
        
    if min_years <= years <= max_years:
        return 1.0
    elif years < min_years:
        return max(0.0, years / min_years)
    else:
        return max(0.0, 1.0 - (years - max_years) / decay)
