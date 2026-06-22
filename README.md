<div align="center">

# Runic
### Intelligent Candidate Discovery & Ranking Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![CPU Only](https://img.shields.io/badge/Compute-CPU--Only-green)
![Runtime](https://img.shields.io/badge/Ranking%20Runtime-%3C60s-brightgreen)
![Memory](https://img.shields.io/badge/RAM--%3C2GB%20during%20ranking-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Team:** `RUNIC` &nbsp;·&nbsp; **Hackathon:** Intelligent Candidate Discovery & Ranking Challenge v4 &nbsp;·&nbsp; **Environment:** Windows 11 · 16 GB RAM · CPU-only · Python 3.13

</div>

---

## Overview

This repository contains a **production-grade, CPU-only candidate ranking engine** built for an Intelligent Candidate Discovery Hackathon. Given a pool of **100,000 candidate profiles** (JSONL) and a Job Description for a *Senior AI Engineer* role, the engine outputs a ranked CSV of the **top 100 most suitable candidates** — each with a score and a data-grounded reasoning string.

> **Key design constraint:** The full ranking step (after pre-computation) completes in **under 60 seconds** on a 16 GB CPU machine with no GPU and no network access.

The system separates work into two clearly defined phases:

| Phase | What It Does | When to Run | Runtime |
|---|---|---|---|
| **Pre-Computation** | Generates embeddings + JD similarity scores for all 100K candidates | Once, offline | ~5–10 min |
| **Ranking** | Streams candidates, filters, scores, and outputs `submission.csv` | Every run / sandbox | **< 60 seconds** |

---

## System Architecture

```
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
│   ├─ Encode full JD text → single 384-dim vector     │
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

## Repository Structure

```
autorecruit-/
├── README.md                         ← You are here
├── submission_metadata.yaml          ← Portal metadata 
├── jd_summary.txt                    ← Dense token-optimized JD summary used for embedding
│
└── Raking_engine/
    ├── rank.py                       ← ★ MAIN ENTRY POINT — produces submission.csv
    ├── requirements.txt              ← Python dependencies + pinned versions
    ├── Dockerfile                    ← Docker sandbox definition
    ├── config/
    │   ├── constants.py              ← Single source of truth: all weights, thresholds, skill lists
    │   └── paths.py                  ← All file path configuration (edit if dataset location differs)
    │
    ├── precompute/
    │   ├── embeddings.py             ← Offline: build candidate text + generate .npy embeddings
    │   └── jd_similarity.py         ← Offline: compute cosine similarity against JD embedding
    │
    ├── filters/
    │   ├── hard_filters.py           ← HardFilter class (7 disqualification rules)
    │   ├── jd_skill_filter.py        ← Two-pass ML skill / career-text gate
    │   └── career_context.py        ← Keyword-stuffer detection + penalty
    │
    ├── scorers/
    │   ├── jd_fit.py                 ← JD Fit: semantic similarity + keyword overlap + title context
    │   ├── skill_bundle.py           ← Bundle Quality: co-occurring skill cluster scoring
    │   ├── demand.py                 ← Market Demand: platform engagement signals
    │   ├── behavior.py               ← Behavior: job-search seriousness + reliability
    │   └── trust.py                  ← Trust: profile credibility + identity verification
    │
    ├── composite/
    │   └── formula.py                ← Weighted composite formula + stuffer penalty application
    │
    ├── loaders/
    │   ├── jsonl_stream.py           ← Memory-safe JSONL reader (yields one line at a time)
    │   ├── csv_loader.py             ← Loads pre-computed behavior + JD similarity CSVs
    │   └── embeddings.py             ← Loads candidate_embeddings.npy + IDs mapping
    │
    ├── output/
    │   ├── top_k.py                  ← O(N log K) min-heap tracker
    │   └── reasoning.py             ← Per-candidate reasoning string generator
    │
    ├── utils/
    │   ├── validators.py             ← Post-run submission format validator (100 rows, monotonic, etc.)
    │   ├── normalize.py              ← Experience flat-top linear decay function
    │   └── timing.py                 ← Wall-clock timing context manager
    │
    └── assets/                       ← Pre-computed artifacts (⚠ NOT in repo — must be generated)
        ├── candidate_embeddings.npy        ⚠ Generate via: python -m precompute.embeddings
        ├── candidate_embeddings_ids.json   ⚠ Generated alongside embeddings.npy
        └── jd_similarity_scores.csv        ⚠ Generated alongside embeddings.npy
```

> **WARNING:** The `assets/` directory contains large binary files that are **not committed to the repository**. You must run the pre-computation step (see below) to generate them before running `rank.py`.

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

## Quick Start

### Step 1 — Clone & Install

```bash
git clone https://github.com/Theshmphony7580/autorecruit-
cd autorecruit-/Raking_engine
```

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS / Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Dependencies (`requirements.txt`):**

| Package | Min Version | Purpose |
|---|---|---|
| `pandas` | 1.5.0 | DataFrame operations, CSV I/O |
| `numpy` | 1.23.0 | Vectorized dot-product similarity |
| `sentence-transformers` | 2.2.0 | BAAI/bge-small-en-v1.5 embedding model |
| `scikit-learn` | 1.1.0 | Utility normalization helpers |
| `tqdm` | 4.60.0 | Progress bars during pre-computation |

---

### Step 2 — Place the Dataset

The engine accepts any dataset file name and location using the `--candidates` flag. For convenience, if you do not want to use flags, you can place your dataset in the `assets/` folder:

```
autorecruit-/Raking_engine/assets/candidates.jsonl
```

---

## Step-by-Step Reproduction

### Pre-Computation — Run Once (Offline)

> This step can take 5–10 minutes on a CPU. It only needs to be run **once**. All outputs are cached to disk and reused on every subsequent ranking run.

```bash
# From the Raking_engine/ directory:
python -m precompute.embeddings
```

**What this step does, in order:**

1. **Loads the embedding model** — `BAAI/bge-small-en-v1.5` (33M parameters, 384-dim output, ~130 MB). Downloads from HuggingFace on first run, cached locally thereafter.

2. **Streams `candidates.jsonl` line-by-line** — calls `build_candidate_text()` for each profile to construct a rich, dense text string.

3. **Encodes all 100K candidates** into 384-dimensional **L2-normalized** embedding vectors, in batches of 64.

4. **Saves candidate embeddings** to `assets/candidate_embeddings.npy` (shape: `[100000, 384]`).

5. **Reads the full JD** from `jd_summary.txt` and encodes it into a single 384-dim L2-normalized vector.

6. **Computes cosine similarity** for all 100K candidates in a single vectorized NumPy operation:
   ```python
   similarities = np.dot(candidate_embeddings, jd_embedding)
   # Both are L2-normalized, so dot product == cosine similarity
   ```

7. **Saves similarity scores** to `assets/jd_similarity_scores.csv` (`candidate_id`, `jd_similarity`).

---

#### What Gets Embedded — The `build_candidate_text()` Function

The function builds a single pipe-delimited (`|`) string per candidate, pulling from every meaningful schema field. Fields are ordered by signal strength to fit within the **512-token context limit** of `bge-small-en-v1.5`:

| # | Section | Schema Fields Used | Truncation / Notes |
|---|---|---|---|
| 1 | **Context line** | `current_title` · `years_of_experience` · `current_industry` | e.g. `"ML Engineer 7 years exp in Internet/Software"` |
| 2 | **Headline** | `profile.headline` | Full text — typically short |
| 3 | **Summary** | `profile.summary` | Truncated to **300 chars** to preserve budget for career history |
| 4 | **Skills** | `skills[].name` | Top **20 skills** sorted descending by `duration_months + endorsements` (strongest skills first) |
| 5 | **Education** | `education[0].degree` · `field_of_study` | Highest/first degree only |
| 6 | **Certifications** | `certifications[].name` | Top **3 certifications** |
| 7 | **Career role 1** | `career_history[0].title` · `company` · `description` | Description truncated to **200 chars** |
| 8 | **Career role 2** | `career_history[1].title` · `company` · `description` | Description truncated to **200 chars** |
| 9 | **Career role 3** | `career_history[2].title` · `company` · `description` | Description truncated to **200 chars** |

> **Why sort skills by `duration_months + endorsements`?** The model's attention window is limited. Placing a candidate's longest-held, most-endorsed skills first ensures the most semantically meaningful signal occupies the highest-priority token positions.

---

### Ranking — The Submission Step

> This is the reproducible step validated in the hackathon sandbox. Must complete within **5 minutes** wall-clock.

```bash
# From the Raking_engine/ directory:

# Using explicit paths:
python rank.py --candidates ../data_forensic\ _files/candidates.jsonl --output submission.csv

# Using configured defaults:
python rank.py
```

**Expected terminal output:**

```
============================================================
RANKING ENGINE STARTED
============================================================

[1/6] Loading pre-computed assets...
[2/6] Initializing filters and scorers...
[3/6] Initializing Top-100 tracker...
[4/6] Processing candidates...
  Processed 10000, Filtered 4821, In heap 100
  Processed 20000, Filtered 9654, In heap 100
  ...
[5/6] Generating output...
[6/6] Writing submission.csv...

Running validation...
PASS: Format valid: 100 rows, ranks 1-100, scores non-increasing.

============================================================
RANKING COMPLETED IN 28.4 seconds
============================================================
Honeypots in top 100: 0/100 (0%)
```

The output `submission.csv` is written with exactly **100 data rows**, columns in spec order: `candidate_id, rank, score, reasoning`.


---

## System Design — Detailed Explanation

### Stage 1: Disqualification Filters &nbsp;`filters/hard_filters.py`

Before scoring, every candidate passes through a **binary disqualification gate**. Rejected candidates are discarded immediately — no score is computed, keeping the streaming pipeline fast.

| Rule | Logic | JD Basis |
|---|---|---|
| **Honeypot Ban** | `honey_pot_labels` is `suspicious` / `high_risk`, or `honeypot_score >= 3`, or `date_anomaly == True` → reject | ~80 planted honeypots with impossible timelines in the dataset |
| **Junior Floor** | `years_of_experience < 3.0` → reject | JD requires 5–9 yrs; floor at 3 gives buffer for hidden talent |
| **Non-Engineering Title** | Title contains: *marketing, HR, recruiter, sales, accountant, designer* → reject | JD: *"Marketing Manager is not a fit no matter how perfect their skill list"* |
| **Wrong AI Domain** | Title contains *computer vision / robotics / speech / hardware* AND summary lacks *nlp / llm / information retrieval* → reject | JD: *"primary expertise is CV or speech without significant NLP/IR exposure"* |
| **Consulting-Only Career** | 100% of `career_history` at TCS / Infosys / Wipro / Accenture / Cognizant / Capgemini → reject | JD: *"People who have only worked at consulting firms in their entire career"* |
| **Job Hopper** | `len(career_history) >= 3` AND average tenure `< 18 months` → reject | JD: *"we need someone who plans to be here for 3+ years"* |
| **Location + Relocation** | Indian candidate not in Pune / Noida / Delhi NCR / Mumbai / Hyderabad AND `willing_to_relocate == False` → reject | JD specifies preferred cities with explicit relocation carve-out |

---

### Stage 2: Skill Validation Gate &nbsp;`filters/jd_skill_filter.py`

This filter implements the JD's **"hidden talent" philosophy** using a two-pass approach, ensuring we don't over-filter candidates who have real ML experience but lack exact keyword tags:

**Pass 1 — Skills list match**
If the candidate's `skills[]` array contains *any* skill from the `JD_CORE_SKILLS` reference set → **passes immediately**.

**Pass 2 — Full-text ML term search**
If Pass 1 fails, the filter scans the combined text of `current_title` + `headline` + `summary` + all `career_history[].description` fields for any `ML_CAREER_TERMS`: *recommendation, search, ranking, personalization, matching, retrieval, embed, vector, semantic, collaborative filtering, content-based, similarity* → any match **passes**.

This catches candidates who built recommendation engines described in career text but never tagged "RAG" or "Embeddings" in their skills section.

**`JD_CORE_SKILLS` — the 20-skill reference set:**
```
embeddings          vector search        semantic search     information retrieval
faiss               pinecone             weaviate            qdrant
milvus              opensearch           langchain           rag
llamaindex          prompt engineering   hugging face        sentence transformers
transformers        llms                 nlp                 recommendation systems
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

The highest-weight scoring component. Three sub-signals combine to reward both semantic depth and explicit keyword alignment:

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

**Formula:** `score = mean(v_norm, s_norm, saves_norm, apps_norm)` — average of 4 values normalized to [0, 1].

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

A **flat-top linear decay** aligned to the JD's stated preferred range of 5–9 years:

| Years of Experience | Score | Rationale |
|---|---|---|
| **5 – 9 years** (in-band) | `1.0` | Perfect JD alignment |
| **< 5 years** (below band) | `years / 5` | Linear ramp — partial credit for near-senior candidates |
| **> 9 years** (above band) | `1.0 − (years − 9) / 5` | Gentle decay — hits 0.0 at 14+ years |

> Experience carries only 5% weight — aligned with the JD's note: *"we'll seriously consider candidates outside the band if other signals are strong."*

---

### Stage 5: O(N log K) Min-Heap Tracker &nbsp;`output/top_k.py`

The engine never accumulates all 100K scored candidates in memory. Instead, it maintains a **fixed-size min-heap of 100 entries** throughout the streaming pass:

- Heap size `< 100` → push unconditionally
- Candidate score `>` current heap minimum → `heapreplace` (pop weakest, push new)
- Otherwise → discard

Peak memory for tracking = **O(K) = O(100)**. Final output: sort the 100-entry heap descending by score. Ties are broken by `candidate_id` lexicographic order (deterministic via Python tuple comparison on `(score, candidate_id, data)` entries).

---

### Stage 6: Automated Reasoning Generation &nbsp;`output/reasoning.py`

For each of the final 100 candidates, a **data-grounded reasoning string** is assembled from actual profile fields. No inference, no hallucination — every sentence maps to a real JSON value.

The 4-step algorithm:

1. **Lead:** `"{current_title} at {current_company} with {N} yrs"` — always from the JSON profile.
2. **Skills:** Up to 3 skills that exist in both the candidate's `skills[]` and `JD_CORE_SKILLS`. Skills not in the JD set are never mentioned.
3. **Career highlight:** First sentence from `career_history[].description` containing a retrieval / ranking / search keyword. Falls back to the first sentence of the most recent role description.
4. **Engagement signal:** `saved_by_recruiters_30d >= 5` → mentions save count; else `recruiter_response_rate > 0.8` → mentions response rate %.

Joined with `". "`, capped at **300 characters**. This satisfies Stage 4's *specific facts*, *no hallucination*, *variation*, and *rank consistency* checks.

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
CAND_0042871,1,0.874521,"ML Engineer at DataScale with 7 yrs. Strong in LLMs, FAISS, Sentence Transformers. Built end-to-end vector retrieval pipeline for e-commerce search. 91% recruiter response rate."
CAND_0019884,2,0.861203,"Senior AI Engineer at Razorpay with 6 yrs. Strong in Embeddings, RAG, LangChain. Designed hybrid retrieval ranking system. Saved by 12 recruiters."
...
CAND_0007729,100,0.412000,"Data Scientist at NovaTech with 5 yrs. Strong in Machine Learning. Worked on recommendation systems for fintech. Adjacent skills only — lower engagement signals reduce rank."
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
  -v /absolute/path/to/your/dataset/folder:/data \
  -v /absolute/path/to/assets:/app/assets \
  autorecruit-ranker \
  python rank.py --candidates /data/your_dataset_name.jsonl --output /data/submission.csv
```

**Google Colab Sandbox Demo** (for online end-to-end testing):

A fully reproducible Google Colab sandbox is the recommended way to verify the engine.
1. Open a new Google Colab notebook.
2. Run this setup cell to clone the repository and run the engine on a sample:
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
| `TRUST_ENDORSEMENTS_MAX` | `50` | Endorsements at which trust score is capped at 1.0 |
| `TRUST_RECENCY_WINDOW_DAYS` | `365` | Days inactive at which recency component = 0 |
| `JD_REALISTIC_SKILL_MAXIMUM` | `10` | Denominator for keyword overlap normalization |

---

## Design Decisions & JD Alignment

This system was built by reading the JD carefully — not just matching keywords. Five key decisions trace directly to specific JD passages:

**1. Semantic embeddings to catch hidden talent**
> *"A Tier 5 candidate may not use the words 'RAG' or 'Pinecone' but if their career history shows they built a recommendation system at a product company, they're a fit."*

The embedding model encodes the full career history description text, not just the skills section — exactly catching this pattern. The two-pass JD skill filter reinforces this with an ML career-text fallback.

**2. Behavioral down-weighting is mandatory**
> *"A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available. Down-weight them appropriately."*

The Behavior Score component (20% weight) uses `open_to_work_flag`, `interview_completion_rate`, `avg_response_time_hours`, and `notice_period_days` to implement this exactly.

**3. Keyword stuffers penalized, not banned**
> *"The right answer is not 'find candidates whose skills section contains the most AI keywords.' That's a trap we've explicitly built into the dataset."*

The stuffer penalty function verifies ML terms in actual career descriptions before penalizing — preserving genuine career transitioners while suppressing pure keyword inflation.

**4. Honeypots naturally filtered by scoring**
> *"We expect a good ranking system to naturally avoid them; you don't need to special-case them."*

We do both: hard-ban via forensic labels (Stage 1A), and the Trust / Bundle scores naturally penalize impossible profiles (e.g. "expert" skill with 0 months experience inflates neither score).

**5. Consulting-only careers are hard-banned**
> *"People who have only worked at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini) in their entire career. We've had bad fit experiences."*

The consulting-only check verifies the *entire* `career_history` array — a candidate with even one product-company role is not rejected.

---

## AI Tools Declaration

AI tools (Claude, Gemini) were used as coding assistants during development. All architecture decisions, scoring weights, filter logic, and JD analysis were performed by the team. The repository reflects genuine engineering iteration — not a single-session LLM dump.

---

## Contact

See [`submission_metadata.yaml`](./submission_metadata.yaml) for primary contact details, team member list, and full portal metadata required by Section 10.2 of the submission specification.

---

<div align="center">
<sub>Built for the Redrob Intelligent Candidate Discovery & Ranking Challenge v4 · Team Theshmphony7580</sub>
</div>
---

