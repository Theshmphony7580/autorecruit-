import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.constants import NON_ML_TITLES, JD_CORE_SKILLS, ML_CAREER_TERMS

class CareerContextFilter:
    def passes(self, candidate: dict) -> bool:
        # We don't hard drop stuffers; we just penalize them later.
        return True

def compute_stuffer_penalty(candidate: dict) -> float:
    profile = candidate.get('profile', {})
    title = profile.get('current_title', '').lower()
    
    has_non_ml_title = any(kw in title for kw in NON_ML_TITLES)
    if not has_non_ml_title:
        return 0.0
        
    raw_skills = candidate.get('skills', [])
    if not isinstance(raw_skills, list):
        raw_skills = []
    candidate_skills = {s.lower().strip() for s in raw_skills}
    ml_skill_count = len(candidate_skills & JD_CORE_SKILLS)
    
    if ml_skill_count == 0:
        return 0.0
        
    career_text = ''
    for role in candidate.get('career_history', []):
        desc = role.get('description', '')
        if desc: career_text += ' ' + desc.lower()
        
    ml_mentions = sum(1 for term in ML_CAREER_TERMS if term in career_text)
    
    if ml_mentions >= 3:
        return 0.1
    elif ml_mentions >= 1:
        return 0.3
    else:
        return 0.7
