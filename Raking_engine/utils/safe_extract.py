def safe_float(val, default=0.0) -> float:
    """Safely cast to float, handling None, empty strings, and invalid types."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def safe_int(val, default=0) -> int:
    """Safely cast to int, handling None, floats (truncating), and invalid types."""
    if val is None:
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default

def safe_str(val, default="") -> str:
    """Safely cast to string, ensuring None returns the default empty string."""
    if val is None:
        return default
    return str(val)

def safe_bool(val, default=False) -> bool:
    """Safely cast to bool, handling None, strings ('true'/'1'), and numbers."""
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ('true', '1', 'yes', 'y')
    if isinstance(val, (int, float)):
        return val == 1
    return default
