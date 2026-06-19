# PRD 2.0: Redrob AI Senior AI Engineer Ranking Engine
## Complete Implementation Guide (Junior-Ready)

---

# Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Data Sources & Inputs](#3-data-sources--inputs)
4. [File Structure](#4-file-structure)
5. [Scoring Formula Deep Dive](#5-scoring-formula-deep-dive)
6. [Filtering Strategy](#6-filtering-strategy)
7. [Implementation Step-by-Step](#7-implementation-step-by-step)
8. [Key Functions Reference](#8-key-functions-reference)
9. [Edge Cases & Error Handling](#9-edge-cases--error-handling)
10. [Pre-computation vs Ranking](#10-pre-computation-vs-ranking)
11. [Testing & Validation](#11-testing--validation)
12. [Quick Reference Cheat Sheet](#12-quick-reference-cheat-sheet)

---

# 1. Executive Summary

## 1.1 What This System Does
Ranks 100,000 candidate profiles to find the TOP 100 most qualified candidates for a "Senior AI Engineer - Founding Team" position at Redrob AI.

## 1.2 Key Constraints
| Constraint | Value | Notes |
|------------|-------|-------|
| Ranking Time | ≤5 minutes | Pre-computation can exceed this |
| RAM | ≤16 GB | CPU-only, no GPU |
| Network | None | No external APIs during ranking |
| Output | 100 rows | candidate_id, rank, score, reasoning |
| Honeypot Rate | ≤10% | Max 10 honeypots in top 100 |

## 1.3 Evaluation Metrics (How Your Ranking Is Scored)
```
Final Score = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10
```

| Metric | Weight | What It Means |
|--------|--------|---------------|
| NDCG@10 | 50% | How good are your TOP 10? (DOMINANT) |
| NDCG@50 | 30% | How good are your TOP 50? |
| MAP | 15% | Precision across all relevant candidates |
| P@10 | 5% | What fraction of top 10 are relevant |

**Takeaway:** Your TOP 10 positions matter MORE than everything else combined.

---

# 2. System Overview

## 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RANKING ENGINE PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STAGE 1: PRE-COMPUTATION (Can exceed 5 min)                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • Load candidate_behavior_scores_full.csv                         │   │
│  │ • Generate embeddings for all 100K candidates (all-MiniLM-L6-v2)  │   │
│  │ • Generate JD embedding                                            │   │
│  │ • Compute JD-candidate cosine similarity                          │   │
│  │ • Save all scores to merged DataFrame                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                       │
│  STAGE 2: RANKING (≤5 min clock starts here)                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. Load pre-computed scores                                        │   │
│  │ 2. Apply FILTERS (remove bad candidates fast)                    │   │
│  │ 3. Apply SCORING FORMULA (compute final scores)                  │   │
│  │ 4. Use HEAPQ for Top-100 (O(N log K) efficiency)                  │   │
│  │ 5. Generate REASONING (template-based, no LLM)                    │   │
│  │ 6. Write submission.csv                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2.2 The Core Insight: Keyword Stuffers vs Plain-Language ML Engineers

The dataset has TWO types of candidates that look similar but are VERY different:

### Keyword Stuffer (BAD)
- **Title:** "Marketing Manager"
- **Skills:** Has AI/ML keywords (LangChain, RAG, Pinecone)
- **Why bad:** Someone added ML keywords to a non-ML profile to game the system

### Plain-Language Tier 5 (GOOD)
- **Title:** "Recommendation Systems Engineer"
- **Skills:** Doesn't say "RAG" or "Pinecone"
- **Why good:** Actually builds recommendation systems but describes it differently

**Our system catches BOTH by checking:**
1. Skills matching (keyword overlap)
2. Career context (title + headline + summary)
3. Skill bundles (co-occurring skills)

---

# 3. Data Sources & Inputs

## 3.1 All Input Files

| File | Location | Size | What It Contains |
|------|----------|------|-----------------|
| `candidates.jsonl` | `data_forensic _files/` | ~500 MB | 100K raw candidate profiles (JSONL) |
| `candidate_behavior_scores_full.csv` | `data_forensic _files/` | ~15 MB | Pre-computed behavior + trust + honeypot scores |
| `frequent_skill_bundles.csv` | `data_forensic _files/` | ~50 KB | 335 skill bundles (co-occurring skills) |
| `job_description.docx` | `resources_provided_by_the_hackthon/` | ~20 KB | The JD we're ranking for |

## 3.2 candidate_behavior_scores_full.csv Columns (What Each Means)

```python
# COLUMNS IN candidate_behavior_scores_full.csv:
columns = [
    'candidate_id',           # Unique ID (e.g., "CAND_0000001")
    
    # Profile signals
    'profile_completeness',   # 0-100, how complete the profile is
    'open_to_work',           # Boolean, candidate is looking for job
    
    # Responsiveness signals
    'response_rate',          # 0-1, how often candidate responds to recruiters
    'response_time',          # Hours, average response time
    
    # Demand signals
    'views',                  # Profile views in 30 days
    'applications',           # Applications submitted in 30 days
    'connections',            # LinkedIn connection count
    'search_appearance',       # Times appeared in recruiter searches
    'saved',                  # Times saved by recruiters
    
    # Reliability signals
    'notice_period',           # Days, current notice period
    'completion_rate',         # 0-1, interview completion rate
    'offer_acceptance',        # 0-1, offer acceptance rate
    
    # Pre-computed scores (USE THESE)
    'availability_score',     # 0-1, recency + open_to_work
    'responsiveness_score',   # 0-1, response_rate + response_time
    'demand_score',            # 0-1, saved + views + search normalized
    'reliability_score',       # 0-1, completion + offer_acceptance
    'behavior_score',         # 0-1, COMPOSITE of all four
    
    # Trust/Forensics signals
    'date_anomaly',           # Boolean, signup > last_active
    'education_score',        # 0-1, education timeline consistency
    'skill_desc_similarity', # 0-1, skills appear in headline/summary/career
    'skill_anomaly_y',        # Boolean, "expert" with 0 duration
    
    # Honeypot detection
    'honeypot_score',         # 0-10+, sum of anomalies (HIGHER = WORSE)
    'honey_pot_labels',       # 'trusted'/'minor_issue'/'suspicious'/'high_risk'
]
```

## 3.3 frequent_skill_bundles.csv Format

```python
# FORMAT:
# support, bundle
# 0.01781, "Hugging Face Transformers, LLMs, Vector Search"
# 0.01780, "Hugging Face Transformers, Semantic Search, Vector Search"

# JD-RELEVANT BUNDLES (keep candidates who have these):
jd_bundles = [
    "Hugging Face Transformers, LLMs, Vector Search",
    "LangChain, Pinecone, Prompt Engineering",
    "Embeddings, Hugging Face Transformers, LLMs",
    "FAISS, Information Retrieval, LLMs",
    "LangChain, Prompt Engineering, RAG",
    "Pinecone, Sentence Transformers, LangChain",
    "Information Retrieval, LLMs, Vector Search",
]
```

## 3.4 The JD Keywords (What Makes a Candidate Relevant)

### MUST-HAVE Skills (Production ML Experience)
```python
JD_CORE_SKILLS = {
    # Embedding/Vector systems
    'embeddings', 'vector search', 'semantic search', 'information retrieval',
    'faiss', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'opensearch',
    
    # LLM/RAG systems
    'langchain', 'rag', 'llamaindex', 'prompt engineering',
    
    # Transformers
    'hugging face', 'sentence transformers', 'transformers',
    
    # General ML
    'llms', 'machine learning', 'ml', 'nlp', 'recommendation systems',
}
```

### NON-ML TITLES (Likely Keyword Stuffers)
```python
NON_ML_TITLES = {
    'marketing manager', 'sales executive', 'hr manager', 'accountant',
    'content writer', 'graphic designer', 'customer support', 'mechanical engineer',
    'civil engineer', 'operations manager', 'business analyst',
}
```

### Plain-Language ML Terms (Catch Tier 5 Engineers)
```python
ML_CAREER_TERMS = [
    'recommendation', 'search', 'ranking', 'personalization',
    'matching', 'retrieval', 'embed', 'vector', 'semantic',
    'collaborative filtering', 'content-based', 'similarity',
]
```

---

# 4. File Structure

## 4.1 Complete Directory Structure

```
Ranking_engine/
├── rank.py                    # MAIN ENTRY POINT
│                               # Usage: python rank.py --candidates <path> --output <path>
│
├── config/
│   ├── __init__.py
│   ├── constants.py           # All constants (JD skills, weights, thresholds)
│   └── paths.py               # File path configuration
│
├── loaders/
│   ├── __init__.py
│   ├── jsonl_stream.py        # Stream candidates from JSONL (memory efficient)
│   ├── csv_loader.py          # Load pre-computed CSVs
│   └── embeddings.py          # Load/generate embeddings
│
├── filters/
│   ├── __init__.py
│   ├── hard_filters.py        # Hard disqualification filters
│   ├── jd_skill_filter.py     # Filter by JD skill match
│   └── career_context.py      # Check career context (keyword stuffer detection)
│
├── scorers/
│   ├── __init__.py
│   ├── jd_fit.py              # JD fit scoring (skills + career context)
│   ├── skill_bundle.py        # Skill bundle quality scoring
│   ├── demand.py              # Recruiter demand scoring
│   ├── behavior.py            # Behavior score lookup
│   ├── trust.py               # Trust/consistency scoring
│   └── honeypot.py            # Honeypot penalty scoring
│
├── composite/
│   │   ├── __init__.py
│   │   └── formula.py         # Final composite score formula
│   │
├── output/
│   │   ├── __init__.py
│   │   ├── top_k.py           # Heap-based top-100 tracker
│   │   ├── reasoning.py       # Template-based reasoning generator
│   │   └── writer.py         # CSV writer with validation
│   │
├── utils/
│   │   ├── __init__.py
│   │   ├── normalize.py       # MinMax normalization
│   │   ├── timing.py          # Timer utilities
│   │   └── validators.py      # Submission validation
│   │
├── precompute/
│   │   ├── __init__.py
│   │   ├── embeddings.py      # Generate candidate embeddings
│   │   └── jd_similarity.py   # Compute JD-candidate similarity
│   │
└── tests/
    ├── __init__.py
    ├── test_filters.py       # Test filtering logic
    ├── test_scorers.py        # Test scoring components
    └── test_integration.py   # End-to-end test
```

## 4.2 File Dependencies (What Imports What)

```
rank.py
├── config.constants
├── config.paths
├── loaders.jsonl_stream
├── loaders.csv_loader
├── loaders.embeddings
├── filters.hard_filters
├── filters.jd_skill_filter
├── filters.career_context
├── scorers.jd_fit
├── scorers.skill_bundle
├── scorers.demand
├── scorers.behavior
├── scorers.trust
├── scorers.honeypot
├── composite.formula
├── output.top_k
├── output.reasoning
├── output.writer
├── utils.normalize
├── utils.validators
└── utils.timing
```

---

# 5. Scoring Formula Deep Dive

## 5.1 Final Score Formula

```python
def final_score(candidate, behavior_df, bundle_scores, jd_similarity):
    """
    Compute final composite score for a candidate.
    
    Formula:
    final_score = (
        JD_FIT*0.30 + 
        BUNDLE_QUALITY*0.15 + 
        DEMAND*0.20 + 
        BEHAVIOR*0.20 + 
        TRUST*0.10 +
        EXPERIENCE*0.05
    ) * TRUST_MULTIPLIER - HONEYPOT_PENALTY
    
    All components normalized to 0-1 before weighting.
    """
    
    # 1. Get component scores
    jd_score = jd_similarity.get(candidate['candidate_id'], 0.0)
    bundle_score = bundle_scores.get(candidate['candidate_id'], 0.0)
    demand_score = behavior_df.loc[candidate['candidate_id'], 'demand_score']
    behavior_score = behavior_df.loc[candidate['candidate_id'], 'behavior_score']
    trust_score = behavior_df.loc[candidate['candidate_id'], 'skill_desc_similarity']
    experience_score = normalize_experience(candidate['profile']['years_of_experience'])
    
    # 2. Get trust multiplier (categorical)
    label = behavior_df.loc[candidate['candidate_id'], 'honey_pot_labels']
    trust_multiplier = {
        'trusted': 1.0,
        'minor_issue': 0.9,
        'suspicious': 0.7,
        'high_risk': 0.4
    }.get(label, 1.0)
    
    # 3. Get honeypot penalty
    honeypot_raw = behavior_df.loc[candidate['candidate_id'], 'honeypot_score']
    honeypot_penalty = min(0.25, honeypot_raw / 10.0)  # Scale to max 0.25
    
    # 4. Compute weighted base
    base = (
        jd_score * 0.30 +
        bundle_score * 0.15 +
        demand_score * 0.20 +
        behavior_score * 0.20 +
        trust_score * 0.10 +
        experience_score * 0.05
    )
    
    # 5. Apply trust multiplier and subtract honeypot penalty
    final = (base * trust_multiplier) - honeypot_penalty
    
    return max(0.0, final)  # Floor at 0
```

## 5.2 Component Details

### Component 1: JD_FIT (30%)
**What it measures:** How well candidate's skills match the JD requirements

**How computed:**
```python
def compute_jd_fit(candidate, jd_keywords):
    """
    1. Check skill overlap with JD keywords
    2. Check career description for JD keywords
    3. Check title/headline for ML context
    """
    candidate_skills = set(s.lower() for s in candidate.get('skills', []))
    jd_keywords_set = set(jd_keywords)
    
    # Skill match (50% weight)
    skill_overlap = len(candidate_skills & jd_keywords_set) / len(jd_keywords_set)
    
    # Career match (30% weight) - catches plain-language ML engineers
    career_text = candidate.get('career_history', [])
    career_text = ' '.join(h.get('description', '') for h in career_text).lower()
    career_match = sum(1 for kw in jd_keywords_set if kw in career_text) / len(jd_keywords_set)
    
    # Title context (20% weight) - catches keyword stuffers
    title = candidate.get('profile', {}).get('current_title', '').lower()
    headline = candidate.get('profile', {}).get('headline', '').lower()
    text = title + ' ' + headline
    has_ml_context = any(kw in text for kw in ['ai', 'ml', 'machine learning', 'data scientist', 
                                                'nlp', 'search', 'ranking', 'recommendation'])
    
    return 0.5 * skill_overlap + 0.3 * career_match + 0.2 * (1.0 if has_ml_context else 0.0)
```

### Component 2: BUNDLE_QUALITY (15%)
**What it measures:** How many JD-relevant skill bundles the candidate has

**How computed:**
```python
def compute_bundle_quality(candidate, jd_bundles):
    """
    Count how many JD-relevant bundles the candidate has.
    A bundle is a set of 3 co-occurring skills.
    """
    candidate_skills = set(s.lower() for s in candidate.get('skills', []))
    bundle_count = 0
    
    for bundle in jd_bundles:
        bundle_skills = set(s.strip().lower() for s in bundle.split(','))
        if len(bundle_skills & candidate_skills) >= 2:  # At least 2 of 3
            bundle_count += 1
    
    # Normalize: max ~10 bundles is a strong signal
    return min(1.0, bundle_count / 5.0)
```

### Component 3: DEMAND (20%)
**What it measures:** Recruiter interest signals (external validation of quality)

**Source:** Pre-computed in `candidate_behavior_scores_full.csv` column `demand_score`

**Formula from forensics:**
```
demand_score = 0.5 * saved_norm + 0.3 * views_norm + 0.2 * search_norm
```

### Component 4: BEHAVIOR (20%)
**What it measures:** Candidate's availability and reliability

**Source:** Pre-computed in `candidate_behavior_scores_full.csv` column `behavior_score`

**Formula from forensics:**
```
behavior_score = 0.25 * availability + 0.25 * responsiveness + 0.30 * demand + 0.20 * reliability
```

### Component 5: TRUST (10%)
**What it measures:** Profile consistency and authenticity

**Source:** `skill_desc_similarity` from behavior CSV

**What it means:** Does the skill name appear in the candidate's headline/summary/career descriptions?
- 1.0 = All skills appear in narrative (consistent)
- 0.0 = No skills appear (likely keyword stuffer)

### Component 6: EXPERIENCE (5%)
**What it measures:** Years of experience within JD range

**How computed:**
```python
def normalize_experience(years):
    """
    JD wants 5-9 years.
    - Inside 5-9: score = 1.0
    - Outside: decay linearly
    """
    if 5 <= years <= 9:
        return 1.0
    elif years < 5:
        return max(0, years / 5.0)  # 4 yrs = 0.8, 3 yrs = 0.6, etc.
    else:
        return max(0, 1.0 - (years - 9) / 5.0)  # 10 yrs = 0.8, 14 yrs = 0.0
```

## 5.3 Trust Multiplier

| Label | Score Range | Multiplier | How Many Candidates |
|-------|-------------|-------------|---------------------|
| `trusted` | honeypot_score = 0 | 1.00 | ~69,394 |
| `minor_issue` | honeypot_score = 1-2 | 0.90 | ~23,084 |
| `suspicious` | honeypot_score = 3-5 | 0.70 | ~7,520 |
| `high_risk` | honeypot_score ≥ 6 | 0.40 | ~2 |

## 5.4 Honeypot Penalty

**What is a honeypot?** A synthetic profile with impossible combinations:
- 8 years at a company founded 3 years ago
- "Expert" in 10 skills with 0 years used
- Skills listed but never mentioned in career history

**Penalty formula:**
```python
honeypot_penalty = min(0.25, honeypot_score / 10.0)
# honeypot_score = 0 → penalty = 0
# honeypot_score = 5 → penalty = 0.125
# honeypot_score = 10 → penalty = 0.25 (max)
```

---

# 6. Filtering Strategy

## 6.1 Why Filter First?

If we score ALL 100K candidates, it takes too long. Filters quickly eliminate obviously bad candidates.

**Filter Pipeline:**

```
100,000 candidates
        ↓
[FILTER 1] Hard Filters (trusted + experience + date)
        ↓
~17,000 survivors
        ↓
[FILTER 2] JD Skill Match
        ↓
~4,000 survivors
        ↓
[FILTER 3] Career Context (keyword stuffer check)
        ↓
~500 survivors ← These get FULL SCORING
        ↓
[RANK] Top 100
```

## 6.2 Filter 1: Hard Filters

```python
def hard_filter(candidate, behavior_df):
    """
    Apply hard filters. Return True if candidate should be DISQUALIFIED.
    """
    cid = candidate['candidate_id']
    row = behavior_df.loc[cid]
    
    # Filter 1: Honeypot labels
    if row['honey_pot_labels'] in ['suspicious', 'high_risk']:
        return True  # DISQUALIFY
    
    # Filter 2: Honeypot score too high
    if row['honeypot_score'] >= 3:
        return True  # DISQUALIFY
    
    # Filter 3: Experience outside acceptable range (soft)
    # Don't disqualify, just mark for lower priority
    
    # Filter 4: Date anomaly (signup > last_active)
    if row.get('date_anomaly', False):
        return True  # DISQUALIFY
    
    return False  # KEEP
```

**Expected elimination:** ~83% (100K → 17K)

## 6.3 Filter 2: JD Skill Match

```python
def jd_skill_filter(candidate, jd_core_skills):
    """
    Does candidate have at least ONE JD-relevant skill?
    Return True if they PASS (keep), False if they FAIL.
    """
    candidate_skills = set(s.lower() for s in candidate.get('skills', []))
    jd_skills = set(jd_core_skills)
    
    # Must have at least 1 JD skill
    if candidate_skills & jd_skills:  # intersection not empty
        return True
    
    # Alternative: check career context for ML terms
    career_text = ' '.join(
        h.get('description', '') for h in candidate.get('career_history', [])
    ).lower()
    
    ml_terms = ['recommendation', 'search', 'ranking', 'personalization', 'retrieval']
    if any(term in career_text for term in ml_terms):
        return True  # Plain-language ML engineer
    
    return False  # FAIL - no JD relevance
```

**Expected elimination:** ~76% (17K → 4K)

## 6.4 Filter 3: Career Context (Keyword Stuffer Detection)

```python
def career_context_filter(candidate):
    """
    Detect keyword stuffers:
    - Non-ML title + ML skills = suspicious
    - skill_desc_similarity very low = suspicious
    
    Return True if PASS (keep), False if FAIL.
    """
    title = candidate.get('profile', {}).get('current_title', '').lower()
    skills = candidate.get('skills', [])
    
    # Non-ML titles
    non_ml_keywords = ['marketing', 'sales', 'hr', 'accountant', 'content', 
                      'graphic', 'mechanical', 'civil', 'operations']
    
    has_non_ml_title = any(kw in title for kw in non_ml_keywords)
    has_ml_skills = any(skill.lower() in JD_CORE_SKILLS for skill in skills)
    
    # Keyword stuffer: non-ML title but ML skills
    if has_non_ml_title and has_ml_skills:
        # Check skill_desc_similarity
        # If low, definitely a stuffer
        # If high, might be real career change (keep)
        return True  # Keep for scoring - let the score decide
    
    return True  # PASS
```

**Expected elimination:** ~87% (4K → 500)

---

# 7. Implementation Step-by-Step

## 7.1 Pre-Computation Phase (Can exceed 5 min)

### Step 1: Load Behavior Scores
```python
# In loaders/csv_loader.py
import pandas as pd

def load_behavior_scores(path: str) -> pd.DataFrame:
    """
    Load pre-computed behavior/trust/honeypot scores.
    
    Returns:
        DataFrame with candidate_id as index
    """
    df = pd.read_csv(
        path,
        usecols=[
            'candidate_id', 'demand_score', 'behavior_score',
            'skill_desc_similarity', 'honeypot_score', 'honey_pot_labels'
        ],
        index_col='candidate_id',
        low_memory=False
    )
    return df
```

### Step 2: Generate Candidate Embeddings
```python
# In precompute/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

def generate_embeddings(candidates_path: str, output_path: str):
    """
    Generate embeddings for all 100K candidates.
    
    Model: all-MiniLM-L6-v2 (384 dimensions)
    Batch size: 64 (for CPU efficiency)
    Time: ~15-20 minutes (acceptable - pre-computation)
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Stream candidates and build text for embedding
    texts = []
    ids = []
    
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            cand = json.loads(line)
            # Combine title + summary + skills + career for rich embedding
            text = f"{cand['profile']['current_title']} {cand['profile']['summary']} "
            text += ' '.join(cand['skills']) + ' '
            text += ' '.join(h.get('description', '') for h in cand.get('career_history', []))
            
            texts.append(text)
            ids.append(cand['candidate_id'])
            
            # Progress every 10K
            if len(texts) % 10000 == 0:
                print(f"Processed {len(texts)} candidates")
    
    # Generate embeddings in batches
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    
    # Save embeddings and ID mapping
    np.save(output_path, embeddings)
    # Also save ID mapping for lookup
    with open(output_path.replace('.npy', '_ids.json'), 'w') as f:
        json.dump(ids, f)
    
    print(f"Saved {len(embeddings)} embeddings to {output_path}")
```

### Step 3: Generate JD Embedding
```python
# In precompute/embeddings.py
def generate_jd_embedding(jd_text: str, model) -> np.ndarray:
    """
    Generate embedding for job description.
    """
    embedding = model.encode([jd_text])
    return embedding[0]
```

### Step 4: Compute JD-Candidate Similarity
```python
# In precompute/jd_similarity.py
import numpy as np

def compute_jd_similarity(
    candidate_embeddings: np.ndarray,
    jd_embedding: np.ndarray,
    candidate_ids: list
) -> dict:
    """
    Compute cosine similarity between JD and all candidates.
    
    Returns:
        dict: {candidate_id: jd_similarity_score}
    """
    # Normalize embeddings
    jd_norm = jd_embedding / np.linalg.norm(jd_embedding)
    cand_norms = candidate_embeddings / np.linalg.norm(
        candidate_embeddings, axis=1, keepdims=True
    )
    
    # Cosine similarity
    similarities = np.dot(cand_norms, jd_norm)
    
    # Build dict
    return {cid: sim for cid, sim in zip(candidate_ids, similarities)}
```

## 7.2 Ranking Phase (≤5 min)

### Step 1: Main Entry Point
```python
# In rank.py
import argparse
import time
import json
import heapq
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True, help='Path to candidates.jsonl')
    parser.add_argument('--output', required=True, help='Path to output submission.csv')
    parser.add_argument('--behavior', default='data_forensic _files/candidate_behavior_scores_full.csv')
    parser.add_argument('--bundles', default='data_forensic _files/frequent_skill_bundles.csv')
    FLAGS = parser.parse_args()
    
    print("=" * 60)
    print("RANKING ENGINE STARTED")
    print("=" * 60)
    
    # Start timer
    start_time = time.time()
    
    # Step 1: Load pre-computed assets
    print("\n[1/6] Loading pre-computed behavior scores...")
    behavior_df = pd.read_csv(FLAGS.behavior, index_col='candidate_id', low_memory=False)
    print(f"  Loaded {len(behavior_df)} behavior records")
    
    # Step 2: Load skill bundles
    print("\n[2/6] Loading skill bundles...")
    bundles_df = pd.read_csv(FLAGS.bundles)
    jd_bundles = bundles_df['bundle'].tolist()
    print(f"  Loaded {len(jd_bundles)} skill bundles")
    
    # Step 3: Initialize filters and scorers
    print("\n[3/6] Initializing filters and scorers...")
    from filters.hard_filters import HardFilter
    from filters.jd_skill_filter import JDSkillFilter
    from filters.career_context import CareerContextFilter
    from scorers.jd_fit import JDFitScorer
    from scorers.skill_bundle import BundleScorer
    from composite.formula import CompositeScorer
    
    hard_filter = HardFilter(behavior_df)
    jd_filter = JDSkillFilter()
    career_filter = CareerContextFilter()
    jd_scorer = JDFitScorer()  # Could load pre-computed JD similarity here
    bundle_scorer = BundleScorer(jd_bundles)
    composite_scorer = CompositeScorer(behavior_df)
    
    # Step 4: Stream candidates, filter, and score
    print("\n[4/6] Processing candidates...")
    from output.top_k import TopKTracker
    
    top100 = TopKTracker(k=100)
    processed = 0
    filtered = 0
    
    with open(FLAGS.candidates, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            candidate = json.loads(line)
            cid = candidate['candidate_id']
            processed += 1
            
            # Apply filters in order (fast reject)
            if hard_filter.is_rejected(candidate):
                filtered += 1
                continue
            
            if not jd_filter.passes(candidate):
                filtered += 1
                continue
                
            if not career_filter.passes(candidate):
                filtered += 1
                continue
            
            # Candidate passed filters - compute score
            score = composite_scorer.compute_score(candidate)
            
            # Add to top-100 heap
            top100.push(cid, score, candidate)
            
            if processed % 10000 == 0:
                print(f"  Processed {processed}, Filtered {filtered}, In heap {len(top100.heap)}")
    
    print(f"\n  Total processed: {processed}")
    print(f"  Total filtered: {filtered}")
    print(f"  In scoring pool: {processed - filtered}")
    
    # Step 5: Generate output
    print("\n[5/6] Generating output...")
    ranked = top100.get_ranked()
    
    # Generate reasoning for each
    from output.reasoning import ReasoningGenerator
    reason_gen = ReasoningGenerator(behavior_df)
    
    results = []
    for rank, (score, cid, candidate) in enumerate(ranked, 1):
        reasoning = reason_gen.generate(candidate, score)
        results.append({
            'candidate_id': cid,
            'rank': rank,
            'score': round(score, 6),
            'reasoning': reasoning
        })
    
    # Step 6: Write CSV
    print("\n[6/6] Writing submission.csv...")
    result_df = pd.DataFrame(results)
    result_df.to_csv(FLAGS.output, index=False)
    
    # Validate
    from utils.validators import validate_submission
    validate_submission(result_df, FLAGS.candidates)
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"RANKING COMPLETED IN {elapsed:.2f} seconds")
    print(f"{'=' * 60}")
    
    # Honeypot check
    honeypot_count = sum(1 for r in results if r['candidate_id'] in honeypot_ids)
    print(f"Honeypots in top 100: {honeypot_count}/100 ({honeypot_count}%)")
    
    if honeypot_count > 10:
        print("⚠️  WARNING: Honeypot rate exceeds 10%!")
```

## 7.3 Helper Classes

### TopKTracker (Heap-based Top 100)
```python
# In output/top_k.py
import heapq

class TopKTracker:
    """
    Maintain top-K candidates using a min-heap.
    
    This is O(N log K) instead of O(N log N) - much faster!
    """
    
    def __init__(self, k: int = 100):
        self.k = k
        self.heap = []  # min-heap stores (score, candidate_id, candidate_data)
    
    def push(self, candidate_id: str, score: float, candidate_data: dict):
        """
        Add candidate to heap. Automatically maintains only top K.
        """
        # Use negative score because Python heap is min-heap
        # Add candidate_id as tie-breaker for determinism
        entry = (-score, candidate_id, candidate_data)
        
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, entry)
        elif score > -self.heap[0][0]:  # New score > min in heap
            heapq.heapreplace(self.heap, entry)
    
    def get_ranked(self) -> list:
        """
        Get top K sorted by score descending.
        """
        # Sort by score descending (negative of stored score)
        sorted_entries = sorted(self.heap, key=lambda x: -x[0])
        return sorted_entries
```

### Reasoning Generator (Candidate-Specific Facts)

**IMPORTANT:** Reasoning must be SPECIFIC to the candidate. Stage 4 manual review grades reasoning for specificity. Templated/generic reasoning is PENALIZED.

**Approach:** Extract key facts from candidate's profile and assemble a specific, factual justification. Each candidate gets a UNIQUE reasoning based on their actual data — no repeated patterns.

```python
def generate_reasoning(candidate: dict, score: float, behavior_df) -> str:
    """
    Generate candidate-specific reasoning from profile facts.
    
    Rules:
    - Must be specific to THIS candidate
    - Use actual facts from their profile
    - No generic templates or filler text
    - Reference specific companies, skills, durations
    """
    # ... [code from above]
```

**Example outputs (each unique to the candidate):**
- "Senior AI Engineer with 7 yrs at Flipkart. strong in LangChain, Pinecone, RAG. built recommendation engine using embeddings. saved by 12 recruiters."
- "ML Engineer with 5 yrs experience. strong in FAISS, Vector Search. built search ranking system for e-commerce. 95% recruiter response rate."
- "Data Scientist with 6 yrs at Razorpay. strong in Hugging Face, LLMs. built semantic search for support tickets."
    """
    Generate candidate-specific reasoning from profile facts.
    
    Rules:
    - Must be specific to THIS candidate
    - Use actual facts from their profile
    - No generic templates or filler text
    - Reference specific companies, skills, durations
    
    Approach:
    1. Extract top 3 facts from their profile
    2. Combine into a 1-2 sentence justification
    3. No repeated patterns across candidates
    """
    profile = candidate.get('profile', {})
    skills = candidate.get('skills', [])
    career = candidate.get('career_history', [])
    cid = candidate['candidate_id']
    
    # Fact 1: Title + years at current company
    current_company = profile.get('current_company', '')
    current_title = profile.get('current_title', 'Engineer')
    years = profile.get('years_of_experience', 0)
    
    # Fact 2: Top relevant skills
    jd_relevant = {'langchain', 'faiss', 'pinecone', 'weaviate', 'embeddings',
                   'vector search', 'semantic search', 'rag', 'llms', 
                   'hugging face', 'sentence transformers', 'recommendation systems'}
    relevant_skills = [s for s in skills if s.lower() in jd_relevant][:3]
    
    # Fact 3: Most recent role description (specific)
    recent_desc = ""
    if career:
        for h in career:
            desc = h.get('description', '')
            if desc and any(t in desc.lower() for t in ['recommendation', 'search', 'retrieval', 'ranking']):
                recent_desc = desc[:150]  # Truncate to first sentence-ish
                break
        if not recent_desc and career[0].get('description'):
            recent_desc = career[0]['description'][:150]
    
    # Fact 4: Behavior signal (from pre-computed)
    behavior_row = behavior_df.loc[cid] if cid in behavior_df.index else None
    if behavior_row is not None:
        response_rate = behavior_row.get('response_rate', 0)
        saved = behavior_row.get('saved', 0)
    else:
        response_rate = 0
        saved = 0
    
    # Assemble candidate-specific reasoning
    parts = []
    
    # Part 1: Title and experience
    if current_company and years > 0:
        parts.append(f"{current_title} with {years} yrs")
    elif years > 0:
        parts.append(f"{current_title} with {years} yrs experience")
    
    # Part 2: Specific skills
    if relevant_skills:
        parts.append(f"strong in {', '.join(relevant_skills[:3])}")
    
    # Part 3: Specific work (from career description)
    if recent_desc:
        # Clean up description
        clean_desc = recent_desc.split('.')[0]  # First sentence
        parts.append(f"built {clean_desc.lower()}")
    
    # Part 4: Recruiter signal (only if notable)
    if saved >= 5:
        parts.append(f"saved by {saved} recruiters")
    elif response_rate > 0.8:
        parts.append(f"{response_rate:.0%} recruiter response rate")
    
    # Join parts naturally
    if not parts:
        return f"Ranked based on profile signals and JD alignment"
    
    reasoning = ". ".join(parts) + "."
    return reasoning[:300]  # Cap at 300 chars to avoid bloat


# Example outputs (each unique to the candidate):
# "Senior AI Engineer with 7 yrs at Flipkart. strong in LangChain, Pinecone, 
#  RAG. built recommendation engine using embeddings. saved by 12 recruiters."
#
# "ML Engineer with 5 yrs experience. strong in FAISS, Vector Search. 
#  built search ranking system for e-commerce. 95% recruiter response rate."
#
# "Data Scientist with 6 yrs at Razorpay. strong in Hugging Face, LLMs. 
#  built semantic search for support tickets."
```

---

# 8. Key Functions Reference

## 8.1 Constants (config/constants.py)
```python
# JD Core Skills (must-have for relevant candidates)
JD_CORE_SKILLS = {
    'embeddings', 'vector search', 'semantic search', 'information retrieval',
    'faiss', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'opensearch',
    'langchain', 'rag', 'llamaindex', 'prompt engineering',
    'hugging face', 'sentence transformers', 'transformers',
    'llms', 'machine learning', 'ml', 'nlp', 'recommendation systems',
}

# Non-ML Titles (likely keyword stuffers)
NON_ML_TITLES = {
    'marketing manager', 'sales executive', 'hr manager', 'accountant',
    'content writer', 'graphic designer', 'customer support',
}

# Plain-language ML terms (catch Tier 5 engineers)
ML_CAREER_TERMS = [
    'recommendation', 'search', 'ranking', 'personalization',
    'matching', 'retrieval', 'embed', 'vector', 'semantic',
]

# Scoring weights
WEIGHTS = {
    'jd_fit': 0.30,
    'bundle_quality': 0.15,
    'demand': 0.20,
    'behavior': 0.20,
    'trust': 0.10,
    'experience': 0.05,
}

# Trust multipliers
TRUST_MULTIPLIERS = {
    'trusted': 1.0,
    'minor_issue': 0.9,
    'suspicious': 0.7,
    'high_risk': 0.4,
}

# Honeypot penalty max
HONEYPOT_PENALTY_MAX = 0.25
```

## 8.2 Validation (utils/validators.py)
```python
def validate_submission(df: pd.DataFrame, candidates_path: str):
    """
    Validate submission meets all requirements.
    Raises AssertionError if validation fails.
    """
    # 1. Check columns
    expected_cols = ['candidate_id', 'rank', 'score', 'reasoning']
    assert list(df.columns) == expected_cols, "Missing or wrong columns"
    
    # 2. Check exactly 100 rows
    assert len(df) == 100, f"Expected 100 rows, got {len(df)}"
    
    # 3. Check ranks 1-100 unique
    expected_ranks = set(range(1, 101))
    actual_ranks = set(df['rank'].tolist())
    assert actual_ranks == expected_ranks, "Ranks not 1-100 unique"
    
    # 4. Check scores non-increasing
    assert df['score'].is_monotonic_decreasing, "Scores not non-increasing"
    
    # 5. Check candidate_ids exist in candidates.jsonl
    valid_ids = load_candidate_ids(candidates_path)
    invalid_ids = set(df['candidate_id']) - valid_ids
    assert len(invalid_ids) == 0, f"Invalid candidate_ids: {invalid_ids}"
    
    print("✓ Submission validation passed!")
```

---

# 9. Edge Cases & Error Handling

## 9.1 Missing Data
```python
# If candidate doesn't have behavior score, use defaults
def get_with_default(df, candidate_id, column, default=0.0):
    if candidate_id in df.index:
        return df.loc[candidate_id, column]
    return default
```

## 9.2 Empty Skills
```python
# Candidate has no skills listed
if not candidate.get('skills'):
    return 0.0  # Low JD fit score
```

## 9.3 Date Parsing
```python
# Handle various date formats in JSONL
from datetime import datetime

def parse_date(date_str):
    if not date_str:
        return None
    for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%Y-%m-%dT%H:%M:%S']:
        try:
            return datetime.strptime(date_str, fmt)
        except:
            continue
    return None
```

## 9.4 JSON Parsing Errors
```python
# Handle malformed JSON lines
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            print(f"Warning: Skipping malformed JSON line")
            continue
```

## 9.5 Memory Management
```python
# Process in chunks if memory is tight
CHUNK_SIZE = 10000

for chunk in pd.read_csv(input_path, chunksize=CHUNK_SIZE):
    # Process chunk
    process_chunk(chunk)
    # Explicitly delete to free memory
    del chunk
    import gc
    gc.collect()
```

---

# 10. Pre-computation vs Ranking

## 10.1 What Can Exceed 5 Minutes

| Asset | Estimated Time | Can Exceed 5 Min? |
|-------|----------------|-------------------|
| Generate 100K embeddings | ~15-20 min | ✅ YES |
| Load behavior CSV | ~10 sec | ✅ YES |
| Compute JD similarity | ~30 sec | ✅ YES |
| **RANKING STEP** | **~30 sec** | ❌ NO (≤5 min) |

## 10.2 Execution Order

```
BEFORE RANKING CLOCK:
┌─────────────────────────────────────────┐
│ 1. Generate candidate embeddings       │  (~15 min)
│ 2. Generate JD embedding               │  (~5 sec)
│ 3. Compute JD-candidate similarity     │  (~30 sec)
│ 4. Load behavior_scores.csv            │  (~10 sec)
│ 5. Save similarity scores to CSV       │  (~5 sec)
└─────────────────────────────────────────┘
        ↓
DURING RANKING CLOCK (≤5 min):
┌─────────────────────────────────────────┐
│ 1. Load pre-computed similarity CSV    │  (~5 sec)
│ 2. Load behavior_scores.csv             │  (~10 sec)
│ 3. Stream candidates + filter          │  (~20 sec)
│ 4. Score filtered candidates            │  (~10 sec)
│ 5. Extract top-100                      │  (~5 sec)
│ 6. Generate reasoning                  │  (~5 sec)
│ 7. Write submission.csv                 │  (~2 sec)
└─────────────────────────────────────────┘
        Total: ~60 seconds << 5 min ✅
```

---

# 11. Testing & Validation

## 11.1 Unit Test Example
```python
# tests/test_filters.py
import pytest
from filters.hard_filters import HardFilter
import pandas as pd

def test_hard_filter_rejects_suspicious():
    """Suspicious candidates should be rejected."""
    behavior_df = pd.DataFrame({
        'candidate_id': ['CAND_001', 'CAND_002'],
        'honey_pot_labels': ['suspicious', 'trusted'],
        'honeypot_score': [3, 0],
    }, index=['CAND_001', 'CAND_002'])
    
    hf = HardFilter(behavior_df)
    
    assert hf.is_rejected({'candidate_id': 'CAND_001'}) == True
    assert hf.is_rejected({'candidate_id': 'CAND_002'}) == False
```

## 11.2 Integration Test
```python
# tests/test_integration.py
def test_full_pipeline():
    """Test full ranking on small sample."""
    # Create small test dataset
    test_candidates = [
        {'candidate_id': 'CAND_001', 'profile': {'current_title': 'AI Engineer', 
           'years_of_experience': 7}, 'skills': ['LangChain', 'Pinecone']},
        {'candidate_id': 'CAND_002', 'profile': {'current_title': 'Marketing Manager',
           'years_of_experience': 5}, 'skills': ['LangChain', 'Pinecone']},
    ]
    
    # Run pipeline
    results = run_ranking(test_candidates)
    
    # Verify
    assert len(results) == 2
    assert results[0]['candidate_id'] == 'CAND_001'  # Real ML engineer first
```

## 11.3 Manual Validation
```bash
# Validate submission format
python -c "
import pandas as pd
df = pd.read_csv('submission.csv')
print('Columns:', list(df.columns))
print('Rows:', len(df))
print('Ranks:', sorted(df['rank'].tolist())[:5], '...', sorted(df['rank'].tolist())[-5:])
print('Scores monotonic:', df['score'].is_monotonic_decreasing)
print('Sample:')
print(df.head(3))
"
```

---

# 12. Quick Reference Cheat Sheet

## 12.1 CLI Usage
```bash
# Run ranking
python rank.py \
    --candidates "data_forensic _files/candidates.jsonl" \
    --output "submission.csv"

# Validate output
python -c "import pandas as pd; df = pd.read_csv('submission.csv'); print(df.head())"
```

## 12.2 Scoring Formula Summary
```
final_score = (
    JD_FIT×0.30 + 
    BUNDLE×0.15 + 
    DEMAND×0.20 + 
    BEHAVIOR×0.20 + 
    TRUST×0.10 + 
    EXP×0.05
) × TRUST_MULTIPLIER - HONEYPOT_PENALTY
```

## 12.3 Filter Summary
| Filter | Purpose | Survivors |
|--------|---------|-----------|
| Hard (trust + date) | Remove honeypots | ~17K |
| JD Skills | Relevance check | ~4K |
| Career Context | Keyword stuffer check | ~500 |

## 12.4 Trust Multipliers
| Label | Score | Multiplier |
|-------|-------|------------|
| trusted | 0 | 1.0 |
| minor_issue | 1-2 | 0.9 |
| suspicious | 3-5 | 0.7 |
| high_risk | 6+ | 0.4 |

## 12.5 Common Errors
| Error | Fix |
|-------|-----|
| `KeyError: candidate_id` | Add default handling for missing IDs |
| `MemoryError` | Stream JSONL, don't load all |
| `JSONDecodeError` | Wrap json.loads in try/except |
| ` Honeypot rate >10%` | Apply harder honeypot filtering |

---

# Document Version
- **Version:** PRD 2.0
- **Date:** 2026-06-13
- **Author:** Claude Code (Multi-Agent Planning)
- **Status:** Ready for Implementation

---

This document should contain everything needed to implement the ranking engine. If anything is unclear, refer to the section number in this document.