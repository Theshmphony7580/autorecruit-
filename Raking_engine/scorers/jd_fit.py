import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.constants import JD_CORE_SKILLS, ML_CAREER_TERMS, JD_FIT_SUB_WEIGHTS, JD_REALISTIC_SKILL_MAXIMUM
from utils.safe_extract import safe_str

class JDFitScorer:
    def compute(self, candidate: dict, candidate_id: str, jd_sim_df) -> float:
        if candidate_id in jd_sim_df.index:
            semantic_sim = float(jd_sim_df.loc[candidate_id, 'jd_similarity'])
            semantic_sim = max(0.0, semantic_sim)
        else:
            semantic_sim = 0.0

        raw_skills = candidate.get('skills', [])
        if not isinstance(raw_skills, list):
            raw_skills = []
        candidate_skills_lower = {safe_str(s.get('name', '')).lower().strip() for s in raw_skills if isinstance(s, dict) and s.get('name')}

        overlap_count = len(candidate_skills_lower & JD_CORE_SKILLS)
        skill_overlap = overlap_count / JD_REALISTIC_SKILL_MAXIMUM
        skill_overlap = min(1.0, skill_overlap)

        profile = candidate.get('profile', {})
        title = safe_str(profile.get('current_title', '')).lower()
        headline = safe_str(profile.get('headline', '')).lower()
        text = f"{title} {headline}"

        has_ml_context = any(term in text for term in ML_CAREER_TERMS)
        title_score = 1.0 if has_ml_context else 0.0

        jd_fit = (
            semantic_sim * JD_FIT_SUB_WEIGHTS['semantic_similarity'] +
            skill_overlap * JD_FIT_SUB_WEIGHTS['keyword_overlap'] +
            title_score * JD_FIT_SUB_WEIGHTS['title_context']
        )

        return min(1.0, max(0.0, jd_fit))
