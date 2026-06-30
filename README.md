<div align="center">

# Runic
### Intelligent Candidate Discovery & Ranking Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![CPU Only](https://img.shields.io/badge/Compute-CPU--Only-green)
![Runtime](https://img.shields.io/badge/Ranking%20Runtime-%3C60s-brightgreen)
![Memory](https://img.shields.io/badge/RAM-%3C2GB%20during%20ranking-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
[![Open In Colab](https://colab.research.google.com/drive/1G5z7H-Q5jqIWmlmIG1oGczPgJtd6IMo8?usp=sharing)

**Team:** `RUNIC` &nbsp;·&nbsp; **Hackathon:** Intelligent Candidate Discovery & Ranking Challenge &nbsp;·&nbsp; **Environment:** Windows 11 · 16 GB RAM · CPU-only · Python 3.13

</div>

---

## Quick Start & Code Reproducibility

### Setup

```bash
# 1. Clone & enter directory
git clone https://github.com/Theshmphony7580/autorecruit-
cd autorecruit-/Raking_engine

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

### Data

Place your dataset:
```bash
cp your_dataset.jsonl assets/candidates.jsonl
```

**Required fields:** `candidate_id`, `current_title`, `years_of_experience`, `skills[]`, `career_history[]`, `profile.headline`, `profile.summary`, `redrob_signals.*`, `honey_pot_labels`, etc.

**Pre-generated:** `jd_summary.txt` is included in the repo.

---

### Run

#### First Time Only — Generate Embeddings

```bash
# Takes ~1.5–2 hrs for 100K candidates (requires internet for model download)
python -m precompute.embeddings
```

Generates:
- `assets/candidate_embeddings.npy`
- `assets/candidate_embeddings_ids.json`
- `assets/jd_similarity_scores.csv`

#### Every Run — Rank Candidates

```bash
# Full dataset
python Raking_engine\rank.py 

# Or explicit paths
python rank.py --candidates assets/candidates.jsonl --output submission.csv
```

**Output:** `submission.csv` (100 rows: `candidate_id, rank, score, reasoning`)

**Expected terminal output:**

```text
============================================================
RANKING ENGINE STARTED
============================================================

[1/6] Loading pre-computed assets...
[2/6] Initializing filters and scorers...
[3/6] Initializing Top-100 tracker...
[4/6] Processing candidates...
  Processed 10000, Filtered 8525, In heap 100
  ...
[5/6] Generating output...
[6/6] Writing submission.csv...

Running official validation...
✓ Official submission validation passed!

============================================================
RANKING COMPLETED IN 12.97 seconds
============================================================
Honeypots in top 100: 0/100 (0%)
```

---

### Verify Setup

```bash
# Check dependencies
python -c "import pandas, numpy, sentence_transformers; print('✓ OK')"

# Check dataset exists
ls assets/candidates.jsonl

# Check embeddings (post pre-compute)
ls assets/candidate_embeddings.npy

# Test run (10 candidates)
python Raking_engine\rank.py --limit 10
```

---

The system cleanly separates work into two phases:

| Phase | What It Does | When to Run | Runtime |
|---|---|---|---|
| **Pre-Computation** | Generates embeddings + JD similarity scores for all 100K candidates | Once before ranking (requires internet for initial model download) | ~1.5–2 hrs (full 100K pool) |
| **Ranking** | Streams candidates, filters, scores, outputs `submission.csv` | Every run / sandbox (100% offline) | **< 60 seconds** |

---

## Repository Structure

```text
autorecruit-/
├── README.md                         ← You are here
├── submission_metadata.yaml          ← Portal metadata (Section 10.2)
├── jd_summary.txt                    ← Dense token-optimized JD summary for embedding
│
└── Raking_engine/
    ├── rank.py                       ← ★ MAIN ENTRY POINT — produces submission.csv
    ├── requirements.txt              ← Python dependencies + pinned versions
    ├── Dockerfile                    ← Docker sandbox definition (Section 10.5)
    ├── config/
    │   ├── constants.py              ← Single source of truth: weights, thresholds, skill lists
    │   └── paths.py                  ← All file path configuration
    ├── precompute/
    │   ├── embeddings.py             ← Offline: build candidate text + generate .npy embeddings
    │   └── jd_similarity.py         ← Offline: compute cosine similarity against JD embedding
    ├── filters/
    │   ├── hard_filters.py           ← HardFilter class (7 disqualification rules)
    │   ├── jd_skill_filter.py        ← Two-pass ML skill / career-text gate
    │   └── career_context.py        ← Keyword-stuffer detection + penalty
    ├── scorers/
    │   ├── jd_fit.py                 ← JD Fit: semantic similarity + keyword overlap + title context
    │   ├── skill_bundle.py           ← Bundle Quality: co-occurring skill cluster scoring
    │   ├── demand.py                 ← Market Demand: platform engagement signals
    │   ├── behavior.py               ← Behavior: job-search seriousness + reliability
    │   └── trust.py                  ← Trust: profile credibility + identity verification
    ├── composite/
    │   └── formula.py                ← Weighted composite formula + stuffer penalty application
    ├── loaders/
    │   ├── jsonl_stream.py           ← Memory-safe JSONL reader (yields one line at a time)
    │   ├── csv_loader.py             ← Loads pre-computed behavior + JD similarity CSVs
    │   └── embeddings.py             ← Loads candidate_embeddings.npy + IDs mapping
    ├── output/
    │   ├── top_k.py                  ← O(N log K) min-heap tracker
    │   └── reasoning.py             ← Per-candidate reasoning string generator
    ├── utils/
    │   ├── validators.py             ← Post-run submission format validator
    │   ├── normalize.py              ← Experience flat-top linear decay function
    │   └── timing.py                 ← Wall-clock timing context manager
    └── assets/                       ← Pre-computed artifacts (⚠ NOT in repo — must be generated)
        ├── candidate_embeddings.npy        ⚠ Generate via: python -m precompute.embeddings
        ├── candidate_embeddings_ids.json   ⚠ Generated alongside embeddings.npy
        └── jd_similarity_scores.csv        ⚠ Generated alongside embeddings.npy
```

> **WARNING:** The `assets/` directory contains large binary files that are **not committed to the repository**. Run the pre-computation step before running `rank.py`.

---

## System Architecture

```text
candidates.jsonl (100K profiles)
        │
        ▼
┌──────────────────────────────────────────────────────┐
│           STAGE 1 — PRE-COMPUTATION  (offline)        │
│                                                      │
│  precompute/embeddings.py                            │
│   ├─ Build rich candidate text from all schema fields│
│   ├─ Encode via BAAI/bge-small-en-v1.5 (384-dim)    │
│   └─ Save → assets/candidate_embeddings.npy          │
│                                                      │
│  precompute/jd_similarity.py                         │
│   ├─ Encode JD summary → single 384-dim vector       │
│   ├─ dot(candidate_embeddings, jd_embedding)         │
│   └─ Save → assets/jd_similarity_scores.csv          │
└──────────────────────────────────────────────────────┘
        │  (pre-computed assets loaded at startup)
        ▼
┌──────────────────────────────────────────────────────┐
│           STAGE 2 — RANKING STEP  (rank.py)           │
│                                                      │
│  ① Stream candidates.jsonl  (line-by-line, O(1) mem) │
│                                                      │
│  ② Hard Filter Gate                                  │
│     ├─ HardFilter    → honeypot / title / domain /   │
│     │                  consulting / hopper / location │
│     ├─ JDSkillFilter → skills match OR career-text   │
│     └─ StufferPenalty→ keyword-stuffer detection     │
│                                                      │
│  ③ 6-Factor Composite Score                          │
│     ├─ JD Fit Score      30%  (semantic + keyword)   │
│     ├─ Market Demand     20%  (platform signals)     │
│     ├─ Behavior Score    20%  (engagement/reliability)│
│     ├─ Bundle Quality    15%  (co-occurring skills)  │
│     ├─ Trust Score       10%  (profile credibility)  │
│     └─ Experience Score   5%  (YoE band alignment)   │
│                                                      │
│  ④ O(N log K) Min-Heap  → Top 100 in memory          │
│                                                      │
│  ⑤ Reasoning Generator + CSV Writer                  │
└──────────────────────────────────────────────────────┘
        │
        ▼
submission.csv  →  candidate_id, rank, score, reasoning
```

---

## Prerequisites

| Requirement | Detail |
|---|---|
| **Python** | 3.10 or higher (tested on 3.13) |
| **RAM** | 16 GB minimum |
| **Compute** | CPU only — no GPU required or used at any stage |
| **Disk** | ~1 GB free for the generated `assets/` files |
| **Internet** | Required once during pre-computation to download the embedding model (~130 MB). Ranking step is fully offline. |

---

## Dependencies (`requirements.txt`)

| Package | Min Version | Purpose |
|---|---|---|
| `pandas` | 1.5.0 | DataFrame operations, CSV I/O |
| `numpy` | 1.23.0 | Vectorized dot-product similarity |
| `sentence-transformers` | 2.2.0 | BAAI/bge-small-en-v1.5 embedding model |
| `scikit-learn` | 1.1.0 | Utility normalization helpers |
| `tqdm` | 4.60.0 | Progress bars during pre-computation |
| `pytest` | 7.0.0 | Test suite |

---

## System Design — Detailed Explanation

### Stage 0: Offline Pre-Computation & Embedding Engineering &nbsp;`precompute/embeddings.py`

Before the online ranking pipeline runs, candidate profiles and the Job Description are embedded into 384-dimensional $L_2$-normalized vectors using `BAAI/bge-small-en-v1.5`.

#### What Gets Embedded — `build_candidate_text()`
A single pipe-delimited (`|`) string per candidate, ordered by signal strength to fit within the **512-token context limit**:

| # | Section | Schema Fields Used | Notes |
|---|---|---|---|
| 1 | **Context line** | `current_title` · `years_of_experience` · `current_industry` | e.g. `"ML Engineer 7 years exp in Internet/Software"` |
| 2 | **Headline** | `profile.headline` | Full text |
| 3 | **Summary** | `profile.summary` | Truncated to **300 chars** |
| 4 | **Skills** | `skills[].name` | Top **20 skills** sorted by `duration_months + endorsements` |
| 5 | **Education** | `education[0].degree` · `field_of_study` | First degree only |
| 6 | **Certifications** | `certifications[].name` | Top **3** |
| 7 | **Career role 1** | `career_history[0].title` · `company` · `description` | Description truncated to **200 chars** |
| 8 | **Career role 2** | `career_history[1].title` · `company` · `description` | Description truncated to **200 chars** |
| 9 | **Career role 3** | `career_history[2].title` · `company` · `description` | Description truncated to **200 chars** |

> Sorting skills by `duration_months + endorsements` ensures the most semantically meaningful signal occupies the highest-priority token positions within the model's attention window.

---

### Stage 1: Disqualification Filters &nbsp;`filters/hard_filters.py`

Before scoring, every candidate passes through a **binary disqualification gate**. Rejected candidates are discarded immediately — no score computed, keeping the streaming pipeline fast.

| Rule | Logic | JD Basis |
|---|---|---|
| **Honeypot Ban** | `honey_pot_labels` is `suspicious`/`high_risk`, or `honeypot_score >= 3`, or `date_anomaly == True` → reject | ~80 planted honeypots with impossible timelines |
| **Junior Floor** | `years_of_experience < 3.0` → reject | JD requires 5–9 yrs; floor at 3 gives buffer for hidden talent |
| **Non-Engineering Title** | Title contains: *marketing, HR, recruiter, sales, accountant, designer* → reject | JD: *"Marketing Manager is not a fit no matter how perfect their skill list"* |
| **Wrong AI Domain** | Title contains *computer vision / robotics / speech / hardware* AND summary lacks *nlp / llm / information retrieval* → reject | JD: *"primary expertise is CV or speech without significant NLP/IR exposure"* |
| **Consulting-Only Career** | 100% of `career_history` at TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini → reject | JD: *"People who have only worked at consulting firms in their entire career"* |
| **Job Hopper** | `len(career_history) >= 3` AND avg tenure `< 18 months` → reject | JD: *"we need someone who plans to be here for 3+ years"* |
| **Location + Relocation** | Indian candidate not in Pune/Noida/Delhi NCR/Mumbai/Hyderabad AND `willing_to_relocate == False` → reject | JD specifies preferred cities with explicit relocation carve-out |

---

### Stage 2: Skill Validation Gate &nbsp;`filters/jd_skill_filter.py`

Implements the JD's **"hidden talent" philosophy** using a two-pass approach:

**Pass 1 — Skills list match:** If the candidate's `skills[]` array contains *any* skill from `JD_CORE_SKILLS` → **passes immediately**.

**Pass 2 — Full-text ML term search:** If Pass 1 fails, scans `current_title` + `headline` + `summary` + all `career_history[].description` fields for any `ML_CAREER_TERMS`: *recommendation, search, ranking, personalization, matching, retrieval, embed, vector, semantic, collaborative filtering, content-based, similarity* → any match **passes**.

This catches candidates who built recommendation engines described in career text but never tagged "RAG" or "Embeddings" in their skills section.

**`JD_CORE_SKILLS` — the 20-skill reference set:**
```
embeddings          vector search        semantic search     information retrieval
faiss               pinecone             weaviate            qdrant
milvus              opensearch           langchain           rag
llamaindex          prompt engineering   hugging face        sentence transformers
transformers        llms                 nlp                 recommendation systems
bm25                xgboost              learning-to-rank    fine-tuning / lora / qlora
```

---

### Stage 3: Keyword Inflation Penalty &nbsp;`filters/career_context.py`

Candidates with a non-ML title who pad their skills section with AI buzzwords are **penalized rather than hard-banned**, preserving genuine career-switchers:

| Condition | Penalty Applied |
|---|---|
| Non-ML title, 0 ML skills in `skills[]` | `0.0` — not a stuffer, just unqualified |
| Non-ML title + ML skills, but **0** ML terms in career descriptions | `0.7` — strong keyword-stuffing signal |
| Non-ML title + ML skills + **1–2** ML terms in career descriptions | `0.3` — possible genuine transitioner |
| Non-ML title + ML skills + **3+** ML terms in career descriptions | `0.1` — likely a real practitioner with non-standard title |

Applied as: `final_score = base_score − (penalty × 0.15)`, floored at `0.0`.

---

### Stage 4: Composite Scoring Engine &nbsp;`composite/formula.py`

Every candidate that passes all filters receives a composite score **[0.0 → 1.0]**:

```
score = (jd_fit    × 0.30)
      + (demand    × 0.20)
      + (behavior  × 0.20)
      + (bundle    × 0.15)
      + (trust     × 0.10)
      + (experience× 0.05)
      − (stuffer_penalty × 0.15)
```

---

#### Component 1: JD Fit Score — 30% &nbsp;`scorers/jd_fit.py`

| Sub-Signal | Weight | Source | How It Works |
|---|---|---|---|
| **Semantic Similarity** | 50% | `jd_similarity_scores.csv` | Pre-computed cosine similarity between the JD embedding and the candidate's full-profile embedding. Catches hidden talent — candidates who built retrieval systems without using the word "RAG". |
| **Keyword Overlap** | 30% | `skills[].name` vs `JD_CORE_SKILLS` | Count of exact skill matches ÷ 10, capped at 1.0. Rewards candidates who explicitly list core skills. |
| **Title Context** | 20% | `current_title` + `headline` | Binary check: does title/headline contain any ML/IR context word? `1.0` if yes, `0.0` if no. |

---

#### Component 2: Bundle Quality Score — 15% &nbsp;`scorers/skill_bundle.py`

Rewards candidates with **co-occurring skill ecosystems** rather than isolated keyword lists. Bundles were mined via **Apriori frequent itemset analysis** on the full 100K dataset.

**Match rule:** A bundle is "matched" if the candidate possesses **≥ 2 of the 3 skills** in that bundle (partial overlap intentional — rewards depth without requiring a perfect checklist).

**Formula:** `score = min(1.0, matched_bundles / 5.0)` — 5 or more bundle matches = maximum score.

<details>
<summary><strong>Full bundle list (19 bundles, click to expand)</strong></summary>

```
Hugging Face Transformers + LLMs + Vector Search
Hugging Face Transformers + Semantic Search + Vector Search
LangChain + Pinecone + Prompt Engineering
LangChain + Recommendation Systems + Sentence Transformers
Information Retrieval + LLMs + Sentence Transformers
LangChain + Semantic Search + Sentence Transformers
Embeddings + Hugging Face Transformers + LLMs
FAISS + Information Retrieval + LLMs
Information Retrieval + LLMs + Vector Search
LangChain + Pinecone + Sentence Transformers
Information Retrieval + LLMs + Semantic Search
LangChain + Prompt Engineering + Recommendation Systems
LangChain + Prompt Engineering + RAG
Hugging Face Transformers + LLMs + RAG
Embeddings + Information Retrieval + LLMs
LLMs + LangChain + Sentence Transformers
FAISS + Hugging Face Transformers + LLMs
LLMs + Semantic Search + Vector Search
Pinecone + Prompt Engineering + Recommendation Systems
```

</details>

---

#### Component 3: Market Demand Score — 20% &nbsp;`scorers/demand.py`

Measures how much the **recruiting market already wants this candidate**, using 4 `redrob_signals` fields:

| Signal | Ceiling | What It Represents |
|---|---|---|
| `profile_views_received_30d` | 200 views | Organic recruiter interest |
| `search_appearance_30d` | 100 appearances | How frequently profile surfaces in searches |
| `saved_by_recruiters_30d` | 20 saves | Explicit recruiter shortlisting intent |
| `applications_submitted_30d` | 10 applications | Active job-seeking behaviour |

**Formula:** `score = mean(v_norm, s_norm, saves_norm, apps_norm)`

---

#### Component 4: Behavior Score — 20% &nbsp;`scorers/behavior.py`

Measures **job-search seriousness and reliability** — directly addressing the JD note: *"a perfect-on-paper candidate who hasn't logged in for 6 months is not actually available."*

| Signal | Weight | Normalization |
|---|---|---|
| `open_to_work_flag` | **0.25** | Binary: 1.0 / 0.0 |
| `interview_completion_rate` | **0.20** | Direct float [0, 1] |
| `github_activity_score` | **0.20** | ÷ 100; value of −1 (no GitHub) maps to 0.0 |
| `avg_response_time_hours` | **0.15** | `1 − (hours / 72)`, floor 0 — faster = higher |
| `skill_assessment_scores` | **0.10** | Average of all platform assessment scores ÷ 100 |
| `notice_period_days` | **0.10** | `1 − (days / 90)`, floor 0 — shorter = higher |

---

#### Component 5: Trust Score — 10% &nbsp;`scorers/trust.py`

Measures **profile legitimacy and professional credibility**:

| Signal | Weight | Normalization |
|---|---|---|
| `profile_completeness_score` | **0.25** | ÷ 100 |
| `verified_email` + `verified_phone` | **0.25** | Both = 1.0 · one = 0.5 · none = 0.0 |
| `recruiter_response_rate` | **0.20** | Direct float [0, 1] |
| `last_active_date` | **0.15** | Linear decay over 365 days; active today = 1.0, 1-year-inactive = 0.0 |
| `endorsements_received` | **0.10** | ÷ 50, capped at 1.0 |
| `linkedin_connected` | **0.05** | Binary: 1.0 / 0.0 |

---

#### Component 6: Experience Score — 5% &nbsp;`utils/normalize.py`

| Years of Experience | Score | Rationale |
|---|---|---|
| **5 – 9 years** (in-band) | `1.0` | Perfect JD alignment |
| **< 5 years** (below band) | `years / 5` | Linear ramp — partial credit for near-senior candidates |
| **> 9 years** (above band) | `1.0 − (years − 9) / 5` | Gentle decay — hits 0.0 at 14+ years |

> Experience carries only 5% weight — aligned with the JD: *"we'll seriously consider candidates outside the band if other signals are strong."*

---

### Stage 5: O(N log K) Min-Heap Tracker &nbsp;`output/top_k.py`

The engine never accumulates all 100K scored candidates in memory. Instead, it maintains a **fixed-size min-heap of 100 entries** throughout the streaming pass:

- Heap size `< 100` → push unconditionally
- Candidate score `>` current heap minimum → `heapreplace` (pop weakest, push new)
- Otherwise → discard

Peak memory for tracking = **O(K) = O(100)**. Final output: sort the 100-entry heap descending by score. Ties are broken by `candidate_id` lexicographic order (deterministic via Python tuple comparison on `(score, candidate_id, data)` entries).

---

### Stage 6: Automated Reasoning Generation &nbsp;`output/reasoning.py`

For each of the final 100 candidates, a **cluster-aware, data-grounded reasoning string** is assembled from actual profile fields. No inference, no hallucination — every sentence maps to a real JSON value.

| Stage 4 Check | How This Engine Satisfies It |
|---|---|
| **Specific facts** | Lead always includes title, company, YoE, and named JD-matched skills from the profile |
| **JD connection** | Skills mentioned are cross-checked against `JD_CORE_SKILLS` — irrelevant skills are never cited |
| **Honest concerns** | Notice period ≥ 90 days, YoE below band, or over-seniority are explicitly flagged |
| **No hallucination** | Every claim maps to a `profile`, `skills[]`, `career_history[]`, or `redrob_signals` field |
| **Variation** | 4 rotating structural layouts + 3–7 randomized sentence starters per layout; global project deduplication across the 100-candidate cohort |
| **Rank consistency** | Cluster differentiator sentences cite the specific signal (saves, notice period, semantic coverage) that broke a tight score tie |

---

## Hackathon Constraints Compliance

| Constraint | Spec Limit | Our System |
|---|---|---|
| **Ranking runtime** | ≤ 5 minutes | PASS: ~30 seconds |
| **Memory (ranking)** | ≤ 16 GB RAM | PASS: < 2 GB |
| **Compute** | CPU only | PASS: No GPU at any stage |
| **Network (ranking)** | Off — no API calls | PASS: Fully offline |
| **Disk (intermediate)** | ≤ 5 GB | PASS: ~800 MB (`embeddings.npy`) |
| **Output rows** | Exactly 100 | PASS: Enforced by `TopKTracker(k=100)` |
| **Rank range** | 1–100, each once | PASS: Enforced + validated post-run |
| **Score monotonicity** | Non-increasing with rank | PASS: Heap sort guarantees this |
| **Honeypot rate** | < 10% in top 100 | PASS: Actively filtered at Stage 1 |

> **Note on pre-computation time:** The `precompute.embeddings` step may take ~1.5 to 2 hours for the full 100,000 dataset on a standard CPU, but it is a **one-time offline step**. It is **not** counted in the 5-minute ranking budget. The Colab sandbox runs on a small sample, taking only seconds.

---

## Output Format

The `submission.csv` file contains exactly **101 lines** (1 header + 100 data rows):

```csv
candidate_id,rank,score,reasoning
CAND_0077337,1,0.876367,"Working as Staff Machine Learning Engineer at Paytm for 7 years. Technical core includes Semantic Search, QLoRA, Pinecone, and BM25. Notably built and shipped a production recommendation system at a marketplace product, going from offline experimentation to live A/B test in 5 months."
CAND_0081846,2,0.863880,"Currently Lead AI Engineer at Razorpay (6.7 yrs total experience). Strong background across Information Retrieval, LlamaIndex, Elasticsearch, and Vector Search. Past work shows they built a RAG-based ranking pipeline serving 50M+ queries per month."
...
CAND_0031842,100,0.678588,"3.5 years experience as AI Specialist at Ola, skilled in LLMs, Vector Search, and Information Retrieval. Total tenure (3.5 yrs) falls under our 5-year benchmark."
```

Scores are **monotonically non-increasing** from rank 1 to rank 100. Ties broken deterministically by `candidate_id`.

---

## Docker & Sandbox Execution

A `Dockerfile` is included for fully reproducible containerized execution (Section 10.5 of the spec):

```bash
cd Raking_engine

# Build the image
docker build -t autorecruit-ranker .

# Run ranking on your dataset
docker run \
  -v /absolute/path/to/dataset/folder:/data \
  -v /absolute/path/to/assets:/app/assets \
  autorecruit-ranker \
  python rank.py --candidates /data/candidates.jsonl --output /data/submission.csv
```

**Google Colab Sandbox Demo:**

```python
!git clone https://github.com/Theshmphony7580/autorecruit-
%cd autorecruit-/Raking_engine
!pip install -r requirements.txt

# Upload your dataset to the Colab files pane (e.g. sample_100.jsonl)
!python -m precompute.embeddings --candidates sample_100.jsonl
!python rank.py --candidates sample_100.jsonl --output submission.csv
```

---

## Configuration Reference

All tunable parameters live in **one file**: `Raking_engine/config/constants.py`

#### Composite Score Weights

| Parameter | Value | Description |
|---|---|---|
| `WEIGHTS['jd_fit']` | `0.30` | JD Fit component |
| `WEIGHTS['demand']` | `0.20` | Market Demand component |
| `WEIGHTS['behavior']` | `0.20` | Behavior / Engagement component |
| `WEIGHTS['bundle_quality']` | `0.15` | Skill Bundle Quality component |
| `WEIGHTS['trust']` | `0.10` | Trust / Credibility component |
| `WEIGHTS['experience']` | `0.05` | Experience alignment component |

#### Scoring Thresholds

| Parameter | Value | Description |
|---|---|---|
| `JD_EXPERIENCE_MIN` | `5` | Preferred experience band lower bound (years) |
| `JD_EXPERIENCE_MAX` | `9` | Preferred experience band upper bound (years) |
| `EXPERIENCE_DECAY` | `5` | Years outside band before score reaches 0 |
| `BEHAVIOR_RESPONSE_TIME_CEILING_HOURS` | `72` | Response time above which behavior score = 0 |
| `BEHAVIOR_NOTICE_PERIOD_CEILING_DAYS` | `90` | Notice period above which behavior score = 0 |
| `TRUST_ENDORSEMENTS_MAX` | `50` | Endorsements at which trust score caps at 1.0 |
| `TRUST_RECENCY_WINDOW_DAYS` | `365` | Days inactive at which recency component = 0 |
| `JD_REALISTIC_SKILL_MAXIMUM` | `15` | Denominator for keyword overlap normalization |

---

## Design Decisions & JD Alignment

This system was built by reading the JD carefully — not just matching keywords. Five key decisions trace directly to specific JD passages:

**1. Semantic embeddings to catch hidden talent**
> *"A Tier 5 candidate may not use the words 'RAG' or 'Pinecone' but if their career history shows they built a recommendation system at a product company, they're a fit."*

The embedding model encodes the full career history description text, not just the skills section — exactly catching this pattern. The two-pass JD skill filter reinforces this with an ML career-text fallback.

**2. Behavioral down-weighting is mandatory**
> *"A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available. Down-weight them appropriately."*

The Behavior Score (20% weight) uses `open_to_work_flag`, `interview_completion_rate`, `avg_response_time_hours`, and `notice_period_days` to implement this exactly.

**3. Keyword stuffers penalized, not banned**
> *"The right answer is not 'find candidates whose skills section contains the most AI keywords.' That's a trap we've explicitly built into the dataset."*

The stuffer penalty verifies ML terms in actual career descriptions before penalizing — preserving genuine career transitioners while suppressing pure keyword inflation.

**4. Honeypots naturally filtered by scoring**
> *"We expect a good ranking system to naturally avoid them; you don't need to special-case them."*

We do both: hard-ban via forensic labels (Stage 1), and the Trust / Bundle scores naturally penalize impossible profiles.

**5. Consulting-only careers are hard-banned**
> *"People who have only worked at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini) in their entire career. We've had bad fit experiences."*

The consulting-only check verifies the *entire* `career_history` array — a candidate with even one product-company role is not rejected.

---

## AI Tools Declaration

AI tools (Claude, Gemini) were used as coding assistants during development. All architecture decisions, scoring weights, filter logic, and JD analysis were performed by us 
The full data analysis was also done before any building by us and in hummanise way not just the passing to the llm and like that proper analysis of dataset (It is a synthetic and not the real dataset ). The repository reflects genuine engineering and logical reasoning not a single-session LLM dump.

---

## Contact

See [`submission_metadata.yaml`](./submission_metadata.yaml) for primary contact details, team member list, and full portal metadata required by the submission specification.

---

<div align="center">
<sub>Built for the Intelligent Candidate Discovery & Ranking Challenge · Team RUNIC</sub>
</div>
