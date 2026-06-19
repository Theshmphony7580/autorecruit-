# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge**. The task: rank the top 100 candidates out of 100,000 for a "Senior AI Engineer — Founding Team" JD at Redrob AI and submit them as a CSV with reasoning.

### Key Constraints

- **Ranking step**: ≤5 min, 16 GB RAM, CPU-only, no network/hosted LLMs
- **Pre-computation**: Can exceed 5 min (embeddings, indexes)
- **Submission**: 100 rows, columns: `candidate_id,rank,score,reasoning`
- **Honeypot rate**: ≤10% in top 100 (disqualification filter)
- **Evaluation**: `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`

## Directory Structure

```
F:\autorecruit-\
├── Raking_engine/              # Main ranking engine implementation
│   ├── rank.py                 # CLI entry point
│   ├── config/                 # Constants and paths
│   ├── loaders/                # JSONL streamer, CSV loader, embeddings
│   ├── filters/                # Hard filters, JD skill filter, career context
│   ├── scorers/                # JD fit, skill bundle, behavior, demand, trust
│   ├── composite/              # Composite scoring formula
│   ├── output/                 # Top-K tracker, reasoning generator, CSV writer
│   ├── precompute/             # Embeddings generation, JD similarity
│   ├── utils/                  # Normalization, timing, validators
│   └── tests/                  # Unit tests
├── data_forensic _files/       # Pre-computed assets (not in git)
│   ├── candidates.jsonl       # 100K candidate pool
│   ├── candidate_behavior_scores_full.csv
│   └── frequent_skill_bundles.csv
└── resources provided by the hackthon/  # Hackathon reference files
```

## Commands

```bash
# Run the ranking engine
python Raking_engine/rank.py --candidates "data_forensic _files/candidates.jsonl" --output "submission.csv"

# Validate submission format
python -c "
import pandas as pd
df = pd.read_csv('Raking_engine/submission.csv')
assert list(df.columns) == ['candidate_id','rank','score','reasoning']
assert len(df) == 100
assert df['score'].is_monotonic_decreasing
print('valid: ok')
"

# Run tests
python -m pytest Raking_engine/tests/

# Count candidates
python -c "import json; print(len([json.loads(l) for l in open('data_forensic _files/candidates.jsonl')]))"
```

## Architecture

The ranking pipeline has 4 stages:

1. **Load** — Stream JSONL candidates, load pre-computed CSVs
2. **Filter** — Hard filters → JD skill filter → Career context (keyword stuffer detection)
3. **Score** — JD fit + skill bundles + demand + behavior + trust → Composite
4. **Output** — Heap-based Top-100 tracker, reasoning generation, CSV write

### Scoring Formula

```
final_score = (
    JD_FIT×0.30 + 
    BUNDLE×0.15 + 
    DEMAND×0.20 + 
    BEHAVIOR×0.20 + 
    TRUST×0.10
) × TRUST_MULTIPLIER - HONEYPOT_PENALTY
```

### Key Insight: Keyword Stuffers vs Plain-Language ML Engineers

The dataset contains:
- **Keyword stuffers**: "Marketing Manager" with AI skill keywords
- **Plain-language Tier 5s**: Recommendation system builder who doesn't say "RAG"

The system uses career context checks to catch these.

## Important Files

| File | Purpose |
|------|---------|
| `Raking_engine/rank.py` | Main CLI entry point |
| `Raking_engine/config/constants.py` | JD skills, weights, thresholds |
| `Raking_engine/filters/hard_filters.py` | Reject honeypots/suspicious |
| `Raking_engine/filters/career_context.py` | Keyword stuffer detection |
| `Raking_engine/composite/formula.py` | Final scoring formula |
| `Raking_engine/output/reasoning.py` | Candidate-specific reasoning |

## Data Sources

- `candidate_behavior_scores_full.csv` — Pre-computed behavior + trust + honeypot scores
- `frequent_skill_bundles.csv` — Apriori-mined skill co-occurrences
- `candidate_embeddings.npy` — Pre-computed embeddings (if generated)

## Notes

- Directory `data_forensic _files` has a trailing space — use Python/tools rather than bash `cd`
- Reasoning must be candidate-specific, not templated (penalized in manual review)
- NDCG@10 dominates at 50% — Top 10 positions matter most
- Honeypot detection: `trusted=1.0, minor_issue=0.9, suspicious=0.7, high_risk=0.4`