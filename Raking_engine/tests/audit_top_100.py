import pandas as pd
import json
import os
import sys

# Add the project root to sys.path so we can import constants
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.constants import JD_CORE_SKILLS

def audit_top_100():
    # Setup paths
    from config.paths import DEFAULT_CANDIDATES_PATH, DEFAULT_BEHAVIOR_CSV, DEFAULT_OUTPUT_PATH
    
    submission_path = DEFAULT_OUTPUT_PATH
    candidates_path = DEFAULT_CANDIDATES_PATH
    behavior_path = DEFAULT_BEHAVIOR_CSV
    
    # Load submission
    sub_df = pd.read_csv(submission_path)
    top_ids = set(sub_df['candidate_id'])
    
    # Load behavior for honeypots
    behavior_df = pd.read_csv(behavior_path).set_index('candidate_id')
    
    # Load profiles
    profiles = {}
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                cand = json.loads(line)
                cid = cand['candidate_id']
                if cid in top_ids:
                    profiles[cid] = cand
            except:
                pass

    print("============================================================")
    print(" TOP 100 CANDIDATES FORENSIC AUDIT ")
    print("============================================================\n")
    
    red_flags = 0
    honeypots = 0
    
    # Core JD Skills
    core = {s.lower() for s in JD_CORE_SKILLS}
            
    # Valid tech titles
    valid_titles = ['engineer', 'scientist', 'developer', 'ml', 'ai', 'data', 'architect', 'research']
    
    for idx, row in sub_df.iterrows():
        cid = row['candidate_id']
        rank = row['rank']
        score = row['score']
        
        cand = profiles.get(cid, {})
        prof = cand.get('profile', {})
        
        title = prof.get('current_title', 'Unknown')
        yoe = prof.get('years_of_experience', 0)
        
        # Check honeypot
        is_honeypot = False
        if cid in behavior_df.index:
            label = behavior_df.loc[cid, 'honey_pot_labels']
            if label in ['suspicious', 'high_risk']:
                is_honeypot = True
                honeypots += 1
                
        # Check skills
        raw_skills = cand.get('skills', [])
        if not isinstance(raw_skills, list): raw_skills = []
        skill_names = [s['name'].lower().strip() for s in raw_skills if isinstance(s, dict) and 'name' in s]
        overlap = len(set(skill_names) & core)
        
        # Determine Red Flags
        flags = []
        if is_honeypot:
            flags.append("🚨 HONEYPOT")
        if yoe < 3:
            flags.append(f"⚠️ LOW YOE ({yoe} yrs)")
        if not any(t in title.lower() for t in valid_titles):
            flags.append(f"⚠️ UNRELATED TITLE ('{title}')")
        if overlap < 2:
            flags.append(f"⚠️ LOW RELEVANT SKILLS ({overlap} found)")
            
        if flags:
            red_flags += 1
            print(f"Rank {rank} | Score {score:.4f} | {cid} | {title} | {yoe} YOE")
            print(f"  Flags: {', '.join(flags)}\n")
            
    print("============================================================")
    print(f" Audit Complete: Checked {len(sub_df)} candidates.")
    print(f" Total Red Flagged Candidates: {red_flags} / {len(sub_df)}")
    print(f" Total Honeypots Found: {honeypots} / {len(sub_df)}")
    if red_flags == 0:
        print(" 🎉 NO RED FLAGS FOUND! Your Top 100 is incredibly clean.")
    print("============================================================")

if __name__ == '__main__':
    audit_top_100()
