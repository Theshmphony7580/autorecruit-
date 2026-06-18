def get_demand_score(candidate_id: str, behavior_df) -> float:
    if candidate_id not in behavior_df.index:
        return 0.0
    score = behavior_df.loc[candidate_id, 'demand_score']
    if not isinstance(score, (int, float)) or score != score:
        return 0.0
    return float(max(0.0, min(1.0, score)))
