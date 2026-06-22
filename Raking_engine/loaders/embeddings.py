import numpy as np
import json
from typing import Tuple, List

# BGE-small-en-v1.5 (and most sentence-transformers) hard-truncate at 512 tokens.
# Keep the JD comfortably under this to guarantee zero truncation.
_MAX_TOKENS = 512
_TOKEN_CHAR_RATIO = 4  # conservative: 1 token ≈ 4 chars for BERT WordPiece


def load_candidate_embeddings(npy_path: str, ids_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load embeddings and candidate IDs mapping from disk.
    """
    embeddings = np.load(npy_path)
    with open(ids_path, 'r') as f:
        ids = json.load(f)
    return embeddings, ids


def get_jd_embedding(model, jd_summary_path: str = "jd_summary.txt") -> np.ndarray:
    """
    Embed the JD using a pre-distilled dense summary that fits within the
    model's 512-token context window.

    WHY: The raw jd_extracted.txt (~9 KB) far exceeds 512 tokens. Hard
    truncation silently drops the entire technical-skills section, biasing
    the query vector toward narrative intro text instead of actual requirements.

    FIX: jd_summary.txt is a compressed, high-density distillation (~350 tokens)
    retaining 100% technical signal: required skills, preferred skills, ideal
    profile, and disqualifiers — with all narrative/cultural fluff removed.

    Args:
        model: Sentence-transformers (or compatible) model with `.encode()`.
        jd_summary_path: Path to the dense JD summary text file.

    Returns:
        np.ndarray: L2-normalised 1-D float32 embedding vector.

    Raises:
        ValueError: If the summary exceeds the safe token budget, alerting
                    you to trim jd_summary.txt before the embedding is wrong.
    """
    with open(jd_summary_path, 'r', encoding='utf-8') as f:
        jd_text = f.read().strip()

    # --- Token budget guard -------------------------------------------
    # Estimate token count via char ratio (conservative; real tokenizer is
    # slightly more efficient, so this is a safe upper-bound check).
    estimated_tokens = len(jd_text) / _TOKEN_CHAR_RATIO
    if estimated_tokens > _MAX_TOKENS:
        raise ValueError(
            f"jd_summary.txt is too long (~{int(estimated_tokens)} estimated tokens). "
            f"Must be under {_MAX_TOKENS} tokens for zero-truncation embedding. "
            f"Trim {jd_summary_path} before re-running."
        )
    # -----------------------------------------------------------------

    # normalize_embeddings=True → L2 norm → cosine sim == dot product downstream
    embedding = model.encode(jd_text, normalize_embeddings=True)
    return np.array(embedding, dtype=np.float32)
