import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.constants import WEIGHTS
from utils.normalize import normalize_experience

class CompositeScorer:
    def __init__(self, behavior_df, jd_sim_df, jd_scorer, bundle_scorer):
        self.behavior_df = behavior_df
        self.jd_sim_df = jd_sim_df
        self.jd_scorer = jd_scorer
        self.bundle_scorer = bundle_scorer
        
    def compute_score(self, candidate: dict, candidate_id: str, stuffer_penalty: float = 0.0) -> float:
        jd_fit = self.jd_scorer.compute(candidate, candidate_id, self.jd_sim_df)
        bundle = self.bundle_scorer.compute(candidate)
        
        from scorers.demand import get_demand_score
        from scorers.behavior import get_behavior_score
        from scorers.trust import get_trust_score
        
        demand = get_demand_score(candidate)
        behavior = get_behavior_score(candidate)
        trust = get_trust_score(candidate)
        
        years = candidate.get('profile', {}).get('years_of_experience', 0)
        experience = normalize_experience(years)
        
        base = (
            jd_fit * WEIGHTS['jd_fit'] +
            bundle * WEIGHTS['bundle_quality'] +
            demand * WEIGHTS['demand'] +
            behavior * WEIGHTS['behavior'] +
            trust * WEIGHTS['trust'] +
            experience * WEIGHTS['experience']
        )
        
        final = base - (stuffer_penalty * 0.15)
        return max(0.0, final)
