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
    if skills: parts.append(', '.join(skills))

    career = candidate.get('career_history', [])
    for role in career[:3]:
        desc = role.get('description', '')
        if desc: parts.append(desc[:300])

    return ' | '.join(parts) if parts else ''

def generate_embeddings(candidates_path: str, output_path: str):
    """Generate and save embeddings for all candidates."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts = []
    ids = []
    
    print("Loading candidates and building text...")
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
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
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode([jd_text], normalize_embeddings=True)
    return embedding[0]
