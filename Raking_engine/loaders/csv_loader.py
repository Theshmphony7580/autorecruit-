import pandas as pd

def load_behavior_scores(path: str) -> pd.DataFrame:
    """
    Load pre-computed behavior/trust/honeypot scores.
    Returns: DataFrame with candidate_id as index
    """
    df = pd.read_csv(
        path,
        usecols=[
            'candidate_id', 'demand_score', 'behavior_score',
            'skill_desc_similarity', 'honeypot_score', 'honey_pot_labels',
            'date_anomaly', 'reliability_score'
        ],
        index_col='candidate_id',
        low_memory=False
    )
    return df

def load_jd_similarity(path: str) -> pd.DataFrame:
    """
    Load pre-computed JD similarity scores.
    Returns: DataFrame with candidate_id as index
    """
    df = pd.read_csv(path, index_col='candidate_id')
    return df
