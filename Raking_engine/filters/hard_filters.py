import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.constants import (
    JD_CORE_SKILLS, 
    JD_DISQUALIFIED_TITLES, 
    JD_DISQUALIFIED_DOMAINS, 
    JD_ACCEPTED_LOCATIONS, 
    JD_CONSULTING_FIRMS
)

class HardFilter:
    def __init__(self, behavior_df):
        self.behavior_df = behavior_df
        self._jd_skills_lower = {s.lower() for s in JD_CORE_SKILLS}

    def is_rejected(self, candidate: dict) -> bool:
        cid = candidate.get('candidate_id')
        if not cid or cid not in self.behavior_df.index:
            return True # Reject unknown
            
        row = self.behavior_df.loc[cid]
        
        label = row.get('honey_pot_labels', 'trusted')
        if label in ('suspicious', 'high_risk'):
            return True
            
        hp_score = row.get('honeypot_score', 0)
        if isinstance(hp_score, (int, float)) and hp_score >= 3:
            return True
            
        date_anomaly = row.get('date_anomaly', False)
        is_anomaly = False
        
        if isinstance(date_anomaly, bool):
            is_anomaly = date_anomaly
        elif isinstance(date_anomaly, str):
            is_anomaly = date_anomaly.lower() in ('true', '1', 'yes')
        elif isinstance(date_anomaly, (int, float)):
            is_anomaly = date_anomaly == 1
            
        if is_anomaly:
            return True
            
        # --- NEW: Hard Quality Floor ---
        profile = candidate.get('profile', {})
        years = profile.get('years_of_experience', 0)
        if years < 3.0:
            return True  # Reject juniors/entry-level for a Senior role
            
        # REMOVED: Strict skill-count filter. 
        # The JD explicitly warns against filtering by AI keywords to find "hidden talent" 
        # (Tier-5 candidates who built recommendation systems but didn't tag "RAG" or "Pinecone").
            
        # --- NEW: Specific JD Disqualifiers ---
        
        # 1. Location & Relocation Filter
        location = profile.get('location', '').lower()
        country = profile.get('country', '').lower()
        signals = candidate.get('redrob_signals', {})
        willing_to_relocate = signals.get('willing_to_relocate', False)
        
        base_cities = JD_ACCEPTED_LOCATIONS
        if 'india' in country or country == 'in':
            in_base_city = any(city in location for city in base_cities)
            if not in_base_city and not willing_to_relocate:
                return True # Reject: not in preferred city and won't relocate
                
        # 2. Career History Filters
        history = candidate.get('career_history', [])
        if history:
            # Check for Consulting-only career
            consulting_firms = JD_CONSULTING_FIRMS
            all_consulting = True
            total_months = 0
            
            for job in history:
                company = job.get('company', '').lower()
                if not any(c in company for c in consulting_firms):
                    all_consulting = False
                total_months += job.get('duration_months', 0)
                
            if all_consulting:
                return True # Reject: entire career in consulting/services
                
            # Check for Job Hoppers (avg tenure < 1.5 years across >= 3 jobs)
            if len(history) >= 3:
                avg_months = total_months / len(history)
                if avg_months < 18:
                    return True # Reject: job hopper
                    
        # 3. Title & Domain Traps
        title = profile.get('current_title', '').lower()
        
        # JD trap: "whose title is 'Marketing Manager' is not a fit"
        bad_titles = JD_DISQUALIFIED_TITLES
        if any(bad in title for bad in bad_titles):
            return True # Reject: non-engineering title
            
        # JD trap: "primary expertise is computer vision, speech, or robotics without significant NLP/IR"
        wrong_domains = JD_DISQUALIFIED_DOMAINS
        if any(wrong in title for wrong in wrong_domains):
            # Unless they explicitly mention NLP/IR in their summary
            summary = profile.get('summary', '').lower()
            if 'nlp' not in summary and 'information retrieval' not in summary and 'llm' not in summary:
                return True # Reject: wrong AI domain
                
        return False
