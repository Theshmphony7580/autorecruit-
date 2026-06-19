import os

# Relative path computation based on the fact that this file is in Raking_engine/config/paths.py
# PROJECT_ROOT is f:\autorecruit-
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data_forensic _files')
ASSETS_DIR = os.path.join(PROJECT_ROOT, 'Raking_engine', 'assets')

DEFAULT_CANDIDATES_PATH = os.path.join(DATA_DIR, 'candidates.jsonl')
DEFAULT_BEHAVIOR_CSV    = os.path.join(ASSETS_DIR, 'candidate_behavior_scores_full.csv')
DEFAULT_BUNDLES_CSV     = os.path.join(ASSETS_DIR, 'frequent_skill_bundles.csv')
DEFAULT_EMBEDDINGS_NPY  = os.path.join(ASSETS_DIR, 'candidate_embeddings.npy')
DEFAULT_EMBEDDINGS_IDS  = os.path.join(ASSETS_DIR, 'candidate_embeddings_ids.json')
DEFAULT_JD_SIMILARITY   = os.path.join(ASSETS_DIR, 'jd_similarity_scores.csv')
DEFAULT_JD_TEXT_PATH    = os.path.join(PROJECT_ROOT, 'jd_extracted.txt')
DEFAULT_OUTPUT_PATH     = os.path.join(PROJECT_ROOT, 'Raking_engine', 'submission.csv')
