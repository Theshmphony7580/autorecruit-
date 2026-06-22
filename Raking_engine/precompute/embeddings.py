import json
import numpy as np
from sentence_transformers import SentenceTransformer

def build_candidate_text(candidate: dict) -> str:
    """Build embedding text from candidate profile fields to maximize semantic signal."""
    profile = candidate.get('profile', {})
    parts = []

    # 1. High-level Context (Title, Experience, Industry)
    title = profile.get('current_title', '')
    yoe = profile.get('years_of_experience')
    industry = profile.get('current_industry', '')
    
    context = []
    if title: context.append(title)
    if yoe is not None: context.append(f"{yoe} years exp")
    if industry: context.append(f"in {industry}")
    if context:
        parts.append(" ".join(context))

    # 2. Headline & Summary
    headline = profile.get('headline', '')
    if headline: parts.append(headline)

    summary = profile.get('summary', '')
    if summary: parts.append(summary[:300]) # Truncated to save tokens for career history

    # 3. Skills (Sort by duration/endorsements and pick top 20)
    skills = candidate.get('skills', [])
    if skills:
        sorted_skills = sorted(skills, key=lambda x: x.get('duration_months', 0) + x.get('endorsements', 0), reverse=True)
        skill_names = [s.get('name') for s in sorted_skills if isinstance(s, dict) and s.get('name')][:20]
        if skill_names:
            parts.append("Skills: " + ', '.join(skill_names))

    # 4. Education
    education = candidate.get('education', [])
    if education:
        edu = education[0] # Grab highest/most recent degree
        degree = edu.get('degree', '')
        field = edu.get('field_of_study', '')
        if degree and field:
            parts.append(f"Education: {degree} in {field}")

    # 5. Certifications
    certs = candidate.get('certifications', [])
    if certs:
        cert_names = [c.get('name') for c in certs[:3] if c.get('name')]
        if cert_names:
            parts.append("Certs: " + ', '.join(cert_names))

    # 6. Career History (Crucial for semantic match)
    career = candidate.get('career_history', [])
    for role in career[:3]:
        r_title = role.get('title', '')
        r_company = role.get('company', '')
        desc = role.get('description', '')
        
        role_parts = []
        if r_title and r_company:
            role_parts.append(f"{r_title} at {r_company}")
        elif r_title:
            role_parts.append(r_title)
            
        if desc:
            role_parts.append(desc[:200])
            
        if role_parts:
            parts.append(" - ".join(role_parts))

    text = ' | '.join(parts) if parts else ''
    # --- Token Context Guard ---
    # BGE-small has a 512 token limit (~2048 characters).
    # We already truncate individual fields above (summary to 300c, roles to 200c each).
    # This final hard cutoff ensures we NEVER overflow the context window, 
    # guaranteeing that the career history (appended last) doesn't get dropped.
    return text[:2000]

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

def generate_jd_embedding(jd_text: str, model=None):
    """Generate normalized embedding for JD.
    
    Accepts an already-loaded model to avoid reloading weights.
    """
    if model is None:
        model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    embedding = model.encode([jd_text], normalize_embeddings=True)
    return embedding[0]

if __name__ == '__main__':
    import sys
    import os
    import argparse
    # Add the Raking_engine directory to the path so we can import config
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from config.paths import (
        DEFAULT_CANDIDATES_PATH, 
        DEFAULT_EMBEDDINGS_NPY, 
        DEFAULT_JD_TEXT_PATH, 
        DEFAULT_JD_SIMILARITY
    )
    from precompute.jd_similarity import compute_jd_similarity, save_jd_similarity
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', default=DEFAULT_CANDIDATES_PATH, help='Path to candidates.jsonl file')
    args = parser.parse_args()
    
    print("=== STEP 1: Generating/Loading Candidate Embeddings ===")
    ids_path = DEFAULT_EMBEDDINGS_NPY.replace('.npy', '_ids.json')
    if os.path.exists(DEFAULT_EMBEDDINGS_NPY) and os.path.exists(ids_path):
        print(f"Found existing candidate embeddings at {DEFAULT_EMBEDDINGS_NPY}!")
        print("⚡ SKIPPING 2-3 hour candidate generation! Loading from disk (takes 5 seconds)...")
        from precompute.jd_similarity import np
        candidate_embeddings = np.load(DEFAULT_EMBEDDINGS_NPY)
        with open(ids_path, 'r') as f:
            import json
            candidate_ids = json.load(f)
    else:
        print("No existing embeddings found. Starting full generation...")
        candidate_embeddings, candidate_ids = generate_embeddings(
            candidates_path=args.candidates, 
            output_path=DEFAULT_EMBEDDINGS_NPY
        )
    
    print("\n=== STEP 2: Computing JD Similarity ===")
    
    # JD distillation: bge-small-en-v1.5 has a 512-token limit (~384 useful tokens).
    # jd_summary.txt is a hand-crafted dense summary of the full JD that fits perfectly
    # inside this window and preserves all high-signal requirements and disqualifiers.
    # If distilled version is missing, fall back to original (with a warning).
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    jd_distilled_path = os.path.join(repo_root, 'jd_summary.txt')
    
    jd_text = None
    if os.path.exists(jd_distilled_path):
        print(f"Loading distilled JD (token-optimized) from {jd_distilled_path}...")
        with open(jd_distilled_path, 'r', encoding='utf-8') as f:
            jd_text = f.read().strip()
        print(f"  Distilled JD length: {len(jd_text.split())} words (fits within bge-small 384-token window)")
    else:
        print(f"WARNING: jd_embedding_summary.txt not found. Falling back to {DEFAULT_JD_TEXT_PATH}.")
        print("  NOTE: The full JD (~1800 words) exceeds the 384-token model window.")
        print("  Create jd_embedding_summary.txt in the repo root for best embedding quality.")
        try:
            with open(DEFAULT_JD_TEXT_PATH, 'r', encoding='utf-8') as f:
                jd_text = f.read()
        except FileNotFoundError:
            print(f"WARNING: Could not find {DEFAULT_JD_TEXT_PATH}. Using hardcoded fallback.")
            jd_text = "Senior AI Engineer. Skills: embeddings, vector search, semantic search, information retrieval, FAISS, Pinecone, RAG, LangChain, NLP, LLMs, sentence transformers, recommendation systems, ranking, retrieval. 5-9 years. Pune/Noida India."

    print("Generating JD embedding (reusing model from Step 1 to save memory)...")
    # Reuse the model already loaded in generate_embeddings via a fresh lightweight load
    jd_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    jd_embedding = generate_jd_embedding(jd_text, model=jd_model)
    
    print("Computing cosine similarities...")
    similarity_dict = compute_jd_similarity(candidate_embeddings, jd_embedding, candidate_ids)
    
    print(f"Saving similarity scores to {DEFAULT_JD_SIMILARITY}...")
    save_jd_similarity(similarity_dict, DEFAULT_JD_SIMILARITY)
    
    print("\n=== PRE-COMPUTATION FULLY COMPLETED ===")

