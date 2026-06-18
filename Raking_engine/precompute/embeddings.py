import json
import numpy as np
from sentence_transformers import SentenceTransformer

def build_candidate_text(candidate: dict) -> str:
    """Build embedding text from candidate profile fields."""
    profile = candidate.get('profile', {})
    parts = []

    title = profile.get('current_title', '')
    if title: parts.append(title)

    headline = profile.get('headline', '')
    if headline: parts.append(headline)

    summary = profile.get('summary', '')
    if summary: parts.append(summary[:500])

    skills = candidate.get('skills', [])
    if skills:
        skill_names = [s['name'] for s in skills if isinstance(s, dict) and 'name' in s]
        if skill_names:
            parts.append(', '.join(skill_names))

    career = candidate.get('career_history', [])
    for role in career[:3]:
        desc = role.get('description', '')
        if desc: parts.append(desc[:300])

    return ' | '.join(parts) if parts else ''

from tqdm import tqdm

def generate_embeddings(candidates_path: str, output_path: str):
    """Generate and save embeddings for all candidates."""
    print("Loading AI Model into memory (this may take 10-30 seconds, please don't press Ctrl+C)...")
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    
    texts = []
    ids = []
    
    print("Loading candidates and building text from JSONL...")
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=100000, desc="Parsing Profiles"):
            line = line.strip()
            if not line: continue
            try:
                cand = json.loads(line)
                ids.append(cand['candidate_id'])
                texts.append(build_candidate_text(cand))
            except Exception:
                pass
                
    print(f"Generating embeddings for {len(texts)} candidates...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    
    print(f"Saving embeddings to {output_path}...")
    np.save(output_path, embeddings)
    
    ids_path = output_path.replace('.npy', '_ids.json')
    with open(ids_path, 'w') as f:
        json.dump(ids, f)
        
    print("Candidate embeddings generation complete.")
    return embeddings, ids

def generate_jd_embedding(jd_text: str):
    """Generate normalized embedding for JD."""
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    embedding = model.encode([jd_text], normalize_embeddings=True)
    return embedding[0]

if __name__ == '__main__':
    import sys
    import os
    # Add the Raking_engine directory to the path so we can import config
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from config.paths import (
        DEFAULT_CANDIDATES_PATH, 
        DEFAULT_EMBEDDINGS_NPY, 
        DEFAULT_JD_TEXT_PATH, 
        DEFAULT_JD_SIMILARITY
    )
    from precompute.jd_similarity import compute_jd_similarity, save_jd_similarity
    
    print("=== STEP 1: Generating Candidate Embeddings ===")
    candidate_embeddings, candidate_ids = generate_embeddings(
        candidates_path=DEFAULT_CANDIDATES_PATH, 
        output_path=DEFAULT_EMBEDDINGS_NPY
    )
    
    print("\n=== STEP 2: Computing JD Similarity ===")
    print(f"Reading JD text from {DEFAULT_JD_TEXT_PATH}...")
    try:
        with open(DEFAULT_JD_TEXT_PATH, 'r', encoding='utf-8') as f:
            jd_text = f.read()
    except FileNotFoundError:
        print(f"WARNING: Could not find {DEFAULT_JD_TEXT_PATH}. Using fallback text.")
        jd_text = "Senior AI Engineer with experience in LLMs, Vector Search, Langchain, Pinecone, and Machine Learning."

    print("Generating JD embedding...")
    jd_embedding = generate_jd_embedding(jd_text)
    
    print("Computing cosine similarities...")
    similarity_dict = compute_jd_similarity(candidate_embeddings, jd_embedding, candidate_ids)
    
    print(f"Saving similarity scores to {DEFAULT_JD_SIMILARITY}...")
    save_jd_similarity(similarity_dict, DEFAULT_JD_SIMILARITY)
    
    print("\n=== PRE-COMPUTATION FULLY COMPLETED ===")
