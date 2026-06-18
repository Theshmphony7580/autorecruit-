import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.constants import JD_CORE_SKILLS, ML_CAREER_TERMS

class JDSkillFilter:
    def passes(self, candidate: dict) -> bool:
        raw_skills = candidate.get('skills', [])
        if not isinstance(raw_skills, list):
            raw_skills = []
            
        candidate_skills = {s.lower().strip() for s in raw_skills}
        
        if candidate_skills & JD_CORE_SKILLS:
            return True
            
        career_text = ''
        for role in candidate.get('career_history', []):
            desc = role.get('description', '')
            if desc: career_text += ' ' + desc.lower()
            
        profile = candidate.get('profile', {})
        title = profile.get('current_title', '').lower()
        headline = profile.get('headline', '').lower()
        summary = profile.get('summary', '').lower()
        
        full_text = f"{title} {headline} {summary} {career_text}"
        
        for term in ML_CAREER_TERMS:
            if term in full_text:
                return True
                
        return False
