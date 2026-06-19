# Redrob AI Ranking Engine - Reproduction Guide

Our ranking engine is completely decoupled from any local machine paths and dynamically maps to the repository root. Follow these steps to perfectly reproduce `submission.csv` on your environment.

## 1. Environment Setup
Ensure you are running Python 3.9+ and install the strict dependencies:
```bash
pip install -r Raking_engine/requirements.txt
```

## 2. Data Placement
By default, the engine expects the 100k candidate dataset and the forensic CSV to be located in:
`data_forensic _files/`

If your data is located elsewhere, simply update the variables at the top of `Raking_engine/config/paths.py`.

## 3. Pre-Computation (Optional)
If you wish to re-generate the semantic embeddings and JD similarity scores from scratch, run the precompute pipeline. *(Note: This requires ~2GB of RAM to load the `BAAI/bge-small-en-v1.5` model).*

The generated outputs will be automatically saved into `Raking_engine/assets/`.

```bash
python Raking_engine/precompute/embeddings.py
python Raking_engine/precompute/jd_similarity.py
```

## 4. Execute the Ranking Engine
The main engine runs an O(N log K) Min-Heap over the streaming JSONL file. It processes all 100,000 candidates while maintaining a strict 16GB memory limit. It reads the live `redrob_signals` directly from the JSON schema and applies the hard filters from the JD.

```bash
python Raking_engine/rank.py
```

**Expected Runtime:** ~20-30 seconds (CPU only).
**Output:** `Raking_engine/submission.csv`

## 5. Audit (Optional)
We provided an audit script to prove our Top 100 output contains 0 honeypots and correctly filters out non-relevant titles.
```bash
python Raking_engine/tests/audit_top_100.py
```
