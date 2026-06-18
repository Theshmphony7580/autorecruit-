import numpy as np
import pandas as pd

def compute_jd_similarity(
    candidate_embeddings: np.ndarray,
    jd_embedding: np.ndarray,
    candidate_ids: list
) -> dict:
    """Compute cosine similarity between JD and candidates (already normalized)."""
    similarities = np.dot(candidate_embeddings, jd_embedding)
    return {cid: sim for cid, sim in zip(candidate_ids, similarities)}

def save_jd_similarity(similarity_dict: dict, output_path: str):
    """Save the similarity dictionary to a CSV file."""
    df = pd.DataFrame({
        'candidate_id': list(similarity_dict.keys()),
        'jd_similarity': list(similarity_dict.values())
    })
    df.to_csv(output_path, index=False)
