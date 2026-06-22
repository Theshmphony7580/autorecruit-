import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.constants import JD_RELEVANT_BUNDLES
from utils.safe_extract import safe_str

class BundleScorer:
    def compute(self, candidate: dict) -> float:
        raw_skills = candidate.get('skills', [])
        if not isinstance(raw_skills, list):
            return 0.0

        candidate_skills_lower = {safe_str(s.get('name', '')).lower().strip() for s in raw_skills if isinstance(s, dict) and s.get('name')}

        bundle_matches = 0
        for bundle in JD_RELEVANT_BUNDLES:
            bundle_skills = {s.strip().lower() for s in bundle.split(',')}
            overlap = len(candidate_skills_lower & bundle_skills)
            if overlap >= 2:
                bundle_matches += 1

        return min(1.0, bundle_matches / 5.0)
