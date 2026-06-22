import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.constants import JD_CORE_SKILLS, ML_CAREER_TERMS
from utils.safe_extract import safe_str

class JDSkillFilter:
    def passes(self, candidate: dict) -> bool:
        raw_skills = candidate.get('skills', [])
        if not isinstance(raw_skills, list):
            raw_skills = []
            
        candidate_skills = {safe_str(s.get('name', '')).lower().strip() for s in raw_skills if isinstance(s, dict) and s.get('name')}
        
        if candidate_skills & JD_CORE_SKILLS:
            return True
            
        career_text = ''
        for role in candidate.get('career_history', []):
            desc = safe_str(role.get('description', ''))
            if desc: career_text += ' ' + desc.lower()
            
        profile = candidate.get('profile', {})
        title = safe_str(profile.get('current_title', '')).lower()
        headline = safe_str(profile.get('headline', '')).lower()
        summary = safe_str(profile.get('summary', '')).lower()
        
        full_text = f"{title} {headline} {summary} {career_text}"
        
        for term in ML_CAREER_TERMS:
            if term in full_text:
                return True
                
        return False
