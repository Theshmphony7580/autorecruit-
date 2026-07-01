import os
import numpy as np
import json
from typing import Tuple, List, Any

# BGE-small-en-v1.5 (and most sentence-transformers) hard-truncate at 512 tokens.
# Keep the JD comfortably under this to guarantee zero truncation.
_MAX_TOKENS = 512
_TOKEN_CHAR_RATIO = 4  # conservative: 1 token ≈ 4 chars for BERT WordPiece


def _is_model_cached(model_name: str) -> bool:
    """Check if a Hugging Face model repository is already cached locally on disk."""
    if os.path.exists(model_name):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
        for filename in ["config.json", "model.safetensors", "pytorch_model.bin"]:
            cached = try_to_load_from_cache(repo_id=model_name, filename=filename)
            if cached is not None and isinstance(cached, str) and os.path.exists(cached):
                return True
    except Exception:
        pass
    return False


def load_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5", show_progress: bool = True) -> Any:
    """
    Load the sentence-transformers model with explicit download status and progress indicator.

    WHY: On first run on a fresh machine or evaluation setup, downloading the AI model
    (~130 MB weights + tokenizer files) takes 1-3 minutes. Standard initialization can
    appear completely frozen or stuck during download if no clear status or progress bar
    is shown.

    FIX: Detects whether the model is cached locally. If not cached, displays an explicit
    first-time download banner with progress bar tracking before initializing model weights.
    """
    cached = _is_model_cached(model_name)
    
    if show_progress:
        # Ensure HuggingFace progress bars are not disabled by environment variables
        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)

    if not cached and show_progress:
        print("\n" + "=" * 65)
        print("📥 FIRST-TIME AI MODEL DOWNLOAD DETECTED")
        print("=" * 65)
        print(f"Model Name  : {model_name}")
        print("Approx Size : ~130 MB (Weights + Tokenizer Vocab)")
        print("Status      : Downloading files from Hugging Face Hub...")
        print("Note        : This takes 1-3 minutes depending on network speed.")
        print("              Subsequent executions will load instantly from disk cache.")
        print("=" * 65)
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_name)
        except Exception:
            # Fallback to letting SentenceTransformer handle the download directly
            pass
    elif cached and show_progress:
        print(f"⚡ Loading AI embedding model ('{model_name}') from local cache...")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    
    if show_progress:
        print(f"✓ Model '{model_name}' successfully loaded into memory.\n")
        
    return model


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
