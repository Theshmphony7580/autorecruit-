def get_trust_score(candidate_id: str, behavior_df) -> float:
    if candidate_id not in behavior_df.index:
        return 0.0
    score = behavior_df.loc[candidate_id, 'skill_desc_similarity']
    if not isinstance(score, (int, float)) or score != score:
        return 0.0
    return float(max(0.0, min(1.0, score)))
