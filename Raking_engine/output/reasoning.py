import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.constants import JD_CORE_SKILLS

class ReasoningGenerator:
    def __init__(self, behavior_df):
        self.behavior_df = behavior_df
        self._jd_skills_lower = {s.lower() for s in JD_CORE_SKILLS}
        
    def generate(self, candidate: dict, score: float) -> str:
        profile = candidate.get('profile', {})
        skills = candidate.get('skills', [])
        career = candidate.get('career_history', [])
        cid = candidate['candidate_id']
        
        current_company = profile.get('current_company', '')
        current_title = profile.get('current_title', 'Engineer')
        years = profile.get('years_of_experience', 0)
        
        if not isinstance(skills, list): skills = []
        relevant_skills = [s['name'] for s in skills if isinstance(s, dict) and 'name' in s and s['name'].lower() in self._jd_skills_lower][:3]
        
        recent_desc = ""
        if career:
            for h in career:
                desc = h.get('description', '')
                if desc and any(t in desc.lower() for t in ['recommendation', 'search', 'retrieval', 'ranking']):
                    recent_desc = desc[:150]
                    break
            if not recent_desc and career[0].get('description'):
                recent_desc = career[0]['description'][:150]
                
        behavior_row = self.behavior_df.loc[cid] if cid in self.behavior_df.index else None
        if behavior_row is not None:
            response_rate = behavior_row.get('response_rate', 0)
            saved = behavior_row.get('saved', 0)
        else:
            response_rate = 0
            saved = 0
            
        parts = []
        if current_company and years > 0:
            parts.append(f"{current_title} with {years} yrs")
        elif years > 0:
            parts.append(f"{current_title} with {years} yrs experience")
            
        if relevant_skills:
            parts.append(f"strong in {', '.join(relevant_skills[:3])}")
            
        if recent_desc:
            clean_desc = recent_desc.split('.')[0]
            parts.append(f"built {clean_desc.lower()}")
            
        if saved >= 5:
            parts.append(f"saved by {int(saved)} recruiters")
        elif response_rate > 0.8:
            parts.append(f"{response_rate:.0%} recruiter response rate")
            
        if not parts:
            return "Ranked based on profile signals and JD alignment"
            
        reasoning = ". ".join(parts) + "."
        return reasoning[:300]
