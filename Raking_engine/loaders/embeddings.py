import numpy as np
import json
from typing import Tuple, List

def load_candidate_embeddings(npy_path: str, ids_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load embeddings and candidate IDs mapping from disk.
    """
    embeddings = np.load(npy_path)
    with open(ids_path, 'r') as f:
        ids = json.load(f)
    return embeddings, ids
