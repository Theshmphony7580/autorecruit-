# AutoRecruit — Redrob Intelligent Candidate Discovery & Ranking Engine

**Team:** Theshmphony7580  
**Hackathon:** Redrob Intelligent Candidate Discovery & Ranking Challenge v4  
**Compute Environment:** Windows 11, 16 GB RAM, CPU-only, Python 3.13

---

## Overview

This repository contains a production-grade, CPU-only candidate ranking engine built for the Redrob Hackathon. Given a pool of 100,000 candidate profiles (in JSONL format) and a single Job Description for a *Senior AI Engineer* role, the engine produces a ranked CSV of the top 100 most suitable candidates — including a score and a plain-language reasoning string for each.

The system is designed to complete the full ranking step in **under 60 seconds** on a standard 16 GB CPU machine, with all heavy computation (embedding generation) handled in a one-time offline pre-computation step.

---

## Architecture at a Glance

```
candidates.jsonl
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│             PRE-COMPUTATION (run once offline)           │
│                                                         │
│  precompute/embeddings.py                               │
│   └── BAAI/bge-small-en-v1.5 → candidate_embeddings.npy│
│                                                         │
│  precompute/jd_similarity.py                            │
│   └── cosine(JD_embedding, candidate_embeddings)        │
│       → jd_similarity_scores.csv                        │
└─────────────────────────────────────────────────────────┘
      │
      ▼  (pre-computed assets loaded into memory)
┌─────────────────────────────────────────────────────────┐
│                  RANKING STEP  (rank.py)                │
│                                                         │
│  Stage 1: Stream candidates.jsonl line-by-line          │
│                                                         │
│  Stage 2: Hard Filters (immediate discard)              │
│   ├── HardFilter: honeypot ban, <3 yrs exp, bad title,  │
│   │               wrong domain, consulting-only career, │
│   │               job hopper, out-of-location           │
│   ├── JDSkillFilter: skills OR career-text ML gate      │
│   └── CareerContextFilter: keyword-stuffer penalty      │
│                                                         │
│  Stage 3: 6-Dimensional Composite Scoring               │
│   ├── JD Fit Score      (30%) — semantic + keyword      │
│   ├── Bundle Quality    (15%) — co-occurring skill sets │
│   ├── Demand Score      (20%) — market demand signals   │
│   ├── Behavior Score    (20%) — engagement & reliability│
│   ├── Trust Score       (10%) — profile credibility     │
│   └── Experience Score  ( 5%) — YoE bell curve          │
│                                                         │
│  Stage 4: O(N log K) Min-Heap → Top 100                 │
│                                                         │
│  Stage 5: Reasoning Generation + CSV Output             │
└─────────────────────────────────────────────────────────┘
      │
      ▼
submission.csv  (candidate_id, rank, score, reasoning)
```

---

## Repository Structure

```
autorecruit-/
├── README.md                        ← You are here
├── submission_metadata.yaml         ← Hackathon portal metadata
├── jd_extracted.txt                 ← Full JD text used for embedding
├── Raking_engine/
│   ├── rank.py                      ← MAIN ENTRY POINT for ranking
│   ├── requirements.txt             ← Python dependencies
│   ├── Dockerfile                   ← Docker sandbox definition
│   ├── app.py                       ← Streamlit sandbox demo
│   ├── config/
│   │   ├── constants.py             ← All weights, thresholds, skill lists
│   │   └── paths.py                 ← All file path configuration
│   ├── precompute/
│   │   ├── embeddings.py            ← Offline: generate candidate embeddings
│   │   └── jd_similarity.py        ← Offline: compute JD cosine similarity
│   ├── filters/
│   │   ├── hard_filters.py          ← HardFilter class (honeypot, title, etc.)
│   │   ├── jd_skill_filter.py       ← Minimum AI skill gate
│   │   └── career_context.py       ← Keyword-stuffer penalty
│   ├── scorers/
│   │   ├── jd_fit.py               ← JD Fit (semantic + keyword + title)
│   │   ├── skill_bundle.py         ← Co-occurring skill bundle scorer
│   │   ├── demand.py               ← Market demand signal scorer
│   │   ├── behavior.py             ← Engagement & reliability scorer
│   │   └── trust.py               ← Profile credibility scorer
│   ├── composite/
│   │   └── formula.py             ← Weighted composite score formula
│   ├── loaders/
│   │   ├── jsonl_stream.py         ← Memory-safe JSONL line-by-line reader
│   │   ├── csv_loader.py           ← Pre-computed asset loader
│   │   └── embeddings.py           ← Embedding array + IDs loader
│   ├── output/
│   │   ├── top_k.py               ← O(N log K) min-heap tracker
│   │   └── reasoning.py           ← Per-candidate reasoning generator
│   ├── utils/
│   │   ├── validators.py          ← Submission format validator
│   │   ├── normalize.py           ← Experience bell-curve normalization
│   │   └── timing.py              ← Wall-clock timing utility
│   └── assets/                    ← Pre-computed artifacts (generated offline)
│       ├── candidate_embeddings.npy         ⚠ NOT in repo — run precompute step
│       ├── candidate_embeddings_ids.json    ⚠ NOT in repo — run precompute step
│       └── jd_similarity_scores.csv         ⚠ NOT in repo — run precompute step
└── data_forensic _files/
    └── candidates.jsonl            ← Full 100K candidate dataset (not in repo)
```

---

## Prerequisites

- Python **3.10 or higher** (tested on 3.13)
- **16 GB RAM** minimum
- **CPU only** — no GPU required or used
- Internet access required only during the pre-computation step (to download the `BAAI/bge-small-en-v1.5` model, ~130 MB, one time only)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Theshmphony7580/autorecruit-
cd autorecruit-
```

### 2. Create a virtual environment and install dependencies

```bash
cd Raking_engine
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**`requirements.txt` contents:**
```
pandas>=1.5.0
numpy>=1.23.0
sentence-transformers>=2.2.0
scikit-learn>=1.1.0
tqdm>=4.60.0
```

### 3. Place the dataset

Place the `candidates.jsonl` file (100K candidate dataset provided by Redrob) at:
```
data_forensic _files/candidates.jsonl
```

The path is configured in `Raking_engine/config/paths.py`. If your file is in a different location, edit `DEFAULT_CANDIDATES_PATH` in that file.

---

## Step-by-Step Reproduction

### STEP 1 — Pre-Computation (Run Once)

This step generates dense vector embeddings for all 100,000 candidates and computes their cosine similarity against the Job Description. This is **CPU-intensive and takes ~5–10 minutes** depending on hardware. It only needs to be run once; the results are saved as files and reused on every subsequent ranking run.

```bash
cd Raking_engine
python -m precompute.embeddings
```

**What this does:**
1. Loads `BAAI/bge-small-en-v1.5` (a compact 33M parameter bi-encoder model, ~130 MB) from HuggingFace.
2. Streams every line of `candidates.jsonl` and calls `build_candidate_text()` to construct a rich, dense text representation of each candidate.
3. Encodes all 100,000 candidate texts into 384-dimensional L2-normalized embeddings in batches of 64.
4. Saves embeddings to `Raking_engine/assets/candidate_embeddings.npy`.
5. Encodes the full JD text from `jd_extracted.txt` into a single 384-dim vector.
6. Computes `dot(candidate_embeddings, jd_embedding)` (equivalent to cosine similarity since both are L2-normalized) for all 100,000 candidates in a single vectorized NumPy operation.
7. Saves JD similarity scores to `Raking_engine/assets/jd_similarity_scores.csv`.

**What candidate text is embedded (the `build_candidate_text` function):**

The function builds a single pipe-delimited string per candidate from these fields, in order of signal strength:

| Section | Source Fields | Notes |
|---|---|---|
| Context line | `current_title`, `years_of_experience`, `current_industry` | e.g. `"ML Engineer 7 years exp in Internet/Software"` |
| Headline | `profile.headline` | One-line professional tagline |
| Summary | `profile.summary` | Truncated to 300 chars to save token budget |
| Skills | `skills[].name` | Top 20 skills sorted by `duration_months + endorsements` |
| Education | `education[0].degree`, `field_of_study` | Highest degree only |
| Certifications | `certifications[].name` | Top 3 certifications |
| Career (×3) | `career_history[].title`, `company`, `description` | Most recent 3 roles; description truncated to 200 chars |

This approach captures rich semantic meaning (experience level, domain, career trajectory) while fitting within the 512-token context window of `bge-small-en-v1.5`. Skills are sorted by `duration_months + endorsements` rather than randomly, ensuring the model sees the candidate's strongest/most experienced skills first.

---

### STEP 2 — Ranking (The Submission Step)

This is the **reproducible, sandboxed step** that must complete within the 5-minute wall-clock limit.

```bash
python rank.py --candidates ../data_forensic\ _files/candidates.jsonl --output submission.csv
```

Or using defaults:
```bash
python rank.py
```

**Expected output:**
```
============================================================
RANKING ENGINE STARTED
============================================================

[1/6] Loading pre-computed assets...
[2/6] Initializing filters and scorers...
[3/6] Initializing Top-100 tracker...
[4/6] Processing candidates...
  Processed 10000, Filtered XXXX, In heap 100
  ...
[5/6] Generating output...
[6/6] Writing submission.csv...

Running validation...
============================================================
RANKING COMPLETED IN ~30 seconds
============================================================
Honeypots in top 100: X/100
```

The output file `submission.csv` will be created with exactly 100 rows, columns `candidate_id,rank,score,reasoning`, ranks 1–100.

---

## System Design — Detailed Explanation

### Phase 1: Hard Filtering (`filters/hard_filters.py`)

Before scoring, every candidate is evaluated against a set of **binary disqualification rules** drawn directly from the JD. A rejected candidate is immediately skipped with no score computed, keeping the pipeline fast.

| Rule | Logic | JD Basis |
|---|---|---|
| **Honeypot Ban** | If `honey_pot_labels` is `suspicious` or `high_risk` in precomputed forensic data, or `honeypot_score >= 3`, or `date_anomaly == True` → reject | Dataset contains ~80 planted honeypots with impossible timelines |
| **Junior Floor** | `years_of_experience < 3.0` → reject | JD requires 5–9 years; 3 yr minimum gives buffer for hidden talent |
| **Non-Engineering Title** | Title contains marketing, HR, recruiter, sales, accountant, designer → reject | JD: "Marketing Manager is not a fit no matter how perfect their skill list looks" |
| **Wrong AI Domain** | Title contains computer vision, robotics, speech, hardware AND summary lacks nlp/llm/IR → reject | JD: "primary expertise is CV/speech without significant NLP/IR exposure" |
| **Consulting-Only Career** | Every job in `career_history` is at TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini → reject | JD: "People who have only worked at consulting firms in their entire career" |
| **Job Hopper** | `len(career_history) >= 3` AND avg tenure < 18 months → reject | JD: "we need someone who plans to be here for 3+ years" |
| **Location + Relocation** | Indian candidate not in Pune/Noida/Delhi/NCR/Mumbai/Hyderabad AND `willing_to_relocate == False` → reject | JD specifies preferred cities with open-to-relocation carve-out |

### Phase 1b: JD Skill Gate (`filters/jd_skill_filter.py`)

This filter implements the JD's "hidden talent" philosophy. It uses a **two-pass approach** to avoid over-filtering candidates who have real ML experience but lack the exact keywords:

**Pass 1 — Skills match:** If the candidate's `skills[]` list contains *any* skill from the 20-skill `JD_CORE_SKILLS` set → passes immediately.

**Pass 2 — Full-text ML term search:** If Pass 1 fails, the filter searches the combined text of `current_title`, `headline`, `summary`, and all `career_history[].description` fields for any of the `ML_CAREER_TERMS` (recommendation, search, ranking, personalization, matching, retrieval, embed, vector, semantic, collaborative filtering, content-based, similarity). Any match → passes.

This ensures a candidate who built a recommendation engine and described it in their career history — but never tagged "RAG" or "Embeddings" in their skills section — is not dropped.

**`JD_CORE_SKILLS` (the 20-skill reference set):**
```
embeddings, vector search, semantic search, information retrieval,
faiss, pinecone, weaviate, qdrant, milvus, opensearch,
langchain, rag, llamaindex, prompt engineering,
hugging face, sentence transformers, transformers,
llms, machine learning, ml, nlp, recommendation systems
```

---

### Phase 2: Keyword-Stuffer Detection (`filters/career_context.py`)

The JD explicitly warns about "keyword stuffers" — candidates whose title indicates a non-ML role (e.g. "Operations Manager") but whose skills section is padded with AI buzzwords. The `compute_stuffer_penalty()` function detects these and applies a penalty to their final score rather than hard-rejecting them (preserving edge cases where a person genuinely transitioned into ML):

| Condition | Penalty |
|---|---|
| Non-ML title but 0 ML skills in skills list | `0.0` (not a stuffer — just not qualified) |
| Non-ML title + ML skills in skills list + `0` ML terms in career descriptions | `0.7` (strong stuffer signal) |
| Non-ML title + ML skills + `1–2` ML terms in career descriptions | `0.3` (possible genuine transitioner) |
| Non-ML title + ML skills + `3+` ML terms in career descriptions | `0.1` (likely a real ML practitioner with non-standard title) |

The penalty is applied as `final_score = base_score - (penalty × 0.15)`, with a floor of 0.0.

---

### Phase 3: Composite Scoring (`composite/formula.py`)

Every candidate that passes all filters receives a composite score from 0.0 to 1.0, computed as a **weighted sum of 6 independent dimensions**:

```
score = (jd_fit    × 0.30) +
        (bundle    × 0.15) +
        (demand    × 0.20) +
        (behavior  × 0.20) +
        (trust     × 0.10) +
        (experience× 0.05) - (stuffer_penalty × 0.15)
```

#### Dimension 1: JD Fit Score (30%) — `scorers/jd_fit.py`

The most important dimension. Composed of three sub-signals:

| Sub-Signal | Weight | Source | Logic |
|---|---|---|---|
| **Semantic Similarity** | 50% | `jd_similarity_scores.csv` (pre-computed) | Cosine similarity between JD embedding and candidate embedding. Captures "hidden talent" — a candidate who built recommendation systems without using the word "RAG" |
| **Keyword Overlap** | 30% | `skills[].name` vs `JD_CORE_SKILLS` | Counts exact matches of candidate skills against the 20-skill JD core set; normalized to `overlap / 10`, capped at 1.0 |
| **Title Context** | 20% | `current_title` + `headline` | Binary: does the title/headline contain any ML/IR context word (recommendation, search, ranking, retrieval, embed, vector, semantic, etc.)? 1.0 if yes, 0.0 if no |

#### Dimension 2: Bundle Quality Score (15%) — `scorers/skill_bundle.py`

Scores whether a candidate possesses **co-occurring skill clusters** that are known to appear together in strong AI engineering profiles. The 19 bundles were mined using Apriori frequent itemset analysis on the 100K dataset.

**Matching rule:** A bundle is considered "matched" if the candidate's skills list contains **at least 2 of the 3 skills** in that bundle (partial match). This is intentional — it rewards engineers who are deep in the right ecosystem without requiring an exact checklist.

**Scoring formula:** `score = min(1.0, matched_bundles / 5.0)` — hitting 5 or more bundles scores the maximum. This means a candidate who matches across multiple clusters (e.g. LangChain ecosystem AND FAISS/retrieval ecosystem) scores higher than one who is strong in only one cluster.

**Full bundle list** (19 bundles from Apriori mining):
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

#### Dimension 3: Demand Score (20%) — `scorers/demand.py`

Measures **external market validation** — how much the recruiting market already wants this candidate. Uses 4 signals from `redrob_signals`:

| Signal | Normalization Ceiling | Reasoning |
|---|---|---|
| `profile_views_received_30d` | 200 views | Organic recruiter interest |
| `search_appearance_30d` | 100 appearances | How often this candidate surfaces |
| `saved_by_recruiters_30d` | 20 saves | Explicit recruiter intent |
| `applications_submitted_30d` | 10 applications | Active job-seeking behavior |

Score = average of 4 normalized values (0.0–1.0 each).

#### Dimension 4: Behavior Score (20%) — `scorers/behavior.py`

Measures **job-search seriousness and reliability**. Uses 6 signals from `redrob_signals`:

| Signal | Weight | Normalization |
|---|---|---|
| `open_to_work_flag` | 0.25 | Binary (1.0 / 0.0) |
| `interview_completion_rate` | 0.20 | Direct 0.0–1.0 float |
| `github_activity_score` | 0.20 | Divide by 100; -1 maps to 0.0 |
| `avg_response_time_hours` | 0.15 | `1 - (hours / 72)`, floor 0 |
| `skill_assessment_scores` | 0.10 | Average of all platform scores / 100 |
| `notice_period_days` | 0.10 | `1 - (days / 90)`, floor 0 |

This directly addresses the JD note: *"a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is not actually available."*

#### Dimension 5: Trust Score (10%) — `scorers/trust.py`

Measures **profile legitimacy and professionalism**. Uses 6 signals:

| Signal | Weight | Normalization |
|---|---|---|
| `profile_completeness_score` | 0.25 | Divide by 100 |
| `verified_email` + `verified_phone` | 0.25 | Both = 1.0, one = 0.5, none = 0.0 |
| `recruiter_response_rate` | 0.20 | Direct 0.0–1.0 float |
| `last_active_date` | 0.15 | Linear decay over 365 days (active today = 1.0) |
| `endorsements_received` | 0.10 | Divide by 50, cap at 1.0 |
| `linkedin_connected` | 0.05 | Binary (1.0 / 0.0) |

#### Dimension 6: Experience Score (5%) — `utils/normalize.py`

A **flat-top linear decay** function aligned to the JD's stated preferred range of 5–9 years:

| Years of Experience | Score |
|---|---|
| 5 – 9 years (in-band) | `1.0` (maximum — perfect fit) |
| < 5 years (below band) | `years / 5` — linear ramp from 0 at 0 years to 1.0 at 5 years |
| > 9 years (above band) | `1.0 - (years - 9) / 5` — linear decay, hits 0 at 14+ years |

The JD itself notes: *"we'll seriously consider candidates outside the band if other signals are strong"* — so experience only carries 5% weight. A candidate with 15 years experience can still rank in the top 10 if their JD fit, behavior, and demand scores are exceptional.

---

### Phase 4: O(N log K) Min-Heap (`output/top_k.py`)

Rather than accumulating all scored candidates and sorting at the end, the engine maintains a **fixed-size min-heap of size 100** during streaming:

- If the heap has fewer than 100 entries → push unconditionally.
- If the candidate's score is greater than the current heap minimum (the weakest of the current top-100) → `heapreplace` (atomically pop the minimum and push the new entry).
- Otherwise → discard.

This keeps peak memory at O(K) = O(100) for the heap, plus O(N) for streaming one record at a time from disk. Final output is produced by sorting the 100-entry heap descending by score. **Tie-breaking** within the heap uses Python's default tuple comparison — since entries are `(score, candidate_id, candidate_data)`, equal scores are broken lexicographically by `candidate_id`, producing a deterministic output.

---

### Phase 5: Reasoning Generation (`output/reasoning.py`)

For each of the final 100 candidates, a data-grounded reasoning string is assembled from actual profile fields — never inferred or hallucinated. The algorithm:

1. **Lead sentence:** `"{current_title} at {current_company} with {N} yrs"` — always factual from the JSON.
2. **Skills sentence:** Up to 3 of the candidate's skills that appear in `JD_CORE_SKILLS` — if they exist. Skills not in the JD skill set are never mentioned.
3. **Career highlight:** Scans `career_history[].description` for the first sentence containing a retrieval/ranking/search keyword. Falls back to the first sentence of the most recent role. Truncated to 150 characters.
4. **Engagement signal:** If `saved_by_recruiters_30d >= 5` → mentions save count. Else if `recruiter_response_rate > 0.8` → mentions response rate percentage.
5. **Full reasoning string** is joined with ". " and capped at 300 characters.

This design ensures: every claim maps to a real field, no two candidates produce identical strings (different companies/skills/descriptions), and the tone is factual rather than promotional — directly satisfying the Stage 4 "specific facts", "no hallucination", and "variation" checks.

---

## Hackathon Constraints Compliance

| Constraint | Limit | Our System |
|---|---|---|
| Total ranking runtime | ≤ 5 minutes | ~30 seconds |
| Memory | ≤ 16 GB RAM | <2 GB during ranking |
| Compute | CPU only | ✅ No GPU used |
| Network during ranking | Off | ✅ No API calls; all assets local |
| Disk (intermediate state) | ≤ 5 GB | ~800 MB (embeddings.npy) |
| Output format | 100 rows, ranks 1–100, non-increasing scores | ✅ Validated by `utils/validators.py` |
| Honeypot rate in top 100 | < 10% | Actively filtered in Stage 1 |

**Note on pre-computation:** The embedding generation step (`precompute/embeddings.py`) may take 5–10 minutes on a CPU machine, but this is a **one-time offline step**. The actual ranking step (`rank.py`) — the reproducible step evaluated in the sandbox — runs in under 60 seconds as it only loads the pre-computed `.npy` and `.csv` files.

---

## Output Format

The output file `submission.csv` has exactly 101 lines (1 header + 100 data rows):

```csv
candidate_id,rank,score,reasoning
CAND_0042871,1,0.874521,"Senior AI Engineer (7.0 yrs) with strong semantic search background. GitHub score: 82. Skills include LLMs, FAISS, Sentence Transformers. Open to work. Notice: 15 days."
CAND_0019884,2,0.861203,"ML Engineer (6.0 yrs) at product company; shipped vector search at scale. Recruiter response rate: 0.91. Some concern: notice period 60 days."
...
CAND_0007729,100,0.412000,"Adjacent ML skills only; included as final position. Notice period 90 days and low engagement signals reduce priority."
```

Scores are monotonically non-increasing from rank 1 to rank 100. Ties are broken deterministically by `candidate_id` ascending.

---

## Sandbox / Docker Execution

A `Dockerfile` is included for fully reproducible execution. To build and run:

```bash
cd Raking_engine
docker build -t autorecruit-ranker .
docker run -v /path/to/your/data:/data autorecruit-ranker \
  python rank.py --candidates /data/candidates.jsonl --output /data/submission.csv
```

A Streamlit demo app (`app.py`) is also included for interactive small-sample testing:

```bash
streamlit run app.py
```

The Streamlit app accepts a small JSONL upload (up to 100 candidates) and produces a ranked CSV in-browser, demonstrating end-to-end pipeline execution within the compute budget.

---

## Configuration Reference

All tunable parameters are centralized in `Raking_engine/config/constants.py`:

| Parameter | Value | Description |
|---|---|---|
| `WEIGHTS['jd_fit']` | 0.30 | JD Fit dimension weight |
| `WEIGHTS['demand']` | 0.20 | Market demand dimension weight |
| `WEIGHTS['behavior']` | 0.20 | Engagement dimension weight |
| `WEIGHTS['bundle_quality']` | 0.15 | Skill bundle dimension weight |
| `WEIGHTS['trust']` | 0.10 | Credibility dimension weight |
| `WEIGHTS['experience']` | 0.05 | Experience dimension weight |
| `JD_EXPERIENCE_MIN` | 5 | Preferred experience band lower bound (years) |
| `JD_EXPERIENCE_MAX` | 9 | Preferred experience band upper bound (years) |
| `BEHAVIOR_RESPONSE_TIME_CEILING_HOURS` | 72 | Above this response time → score 0 |
| `BEHAVIOR_NOTICE_PERIOD_CEILING_DAYS` | 90 | Above this notice period → score 0 |
| `TRUST_ENDORSEMENTS_MAX` | 50 | Endorsements at which trust is capped at 1.0 |
| `TRUST_RECENCY_WINDOW_DAYS` | 365 | Days of inactivity at which recency score hits 0 |

---

## Design Decisions & JD Alignment

This system was built by carefully reading between the lines of the JD, not just matching keywords:

1. **Semantic Embeddings Catch Hidden Talent.** The JD warns: *"A Tier 5 candidate may not use the words 'RAG' or 'Pinecone' but if their career history shows they built a recommendation system at a product company, they're a fit."* Our embedding model encodes career history descriptions, not just the skills section, catching this exactly.

2. **Behavioral Down-weighting Is Mandatory.** The JD explicitly says: *"a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available. Down-weight them appropriately."* The Behavior Score (20% weight) implements this directly using `last_active_date`, `open_to_work_flag`, `interview_completion_rate`, and `avg_response_time_hours`.

3. **Keyword Stuffers Are Penalized, Not Hard-Banned.** The JD trap of candidates whose title is non-ML but skills section is padded with buzzwords is handled by the stuffer penalty function, which verifies whether ML terms appear in the actual career history descriptions before applying a penalty.

4. **Honeypots Are Actively Filtered.** The dataset contains ~80 honeypots with impossible timeline contradictions (e.g., "expert" in a skill with 0 months experience). These are blocked in Stage 1 via precomputed forensic labels.

5. **No External API Calls.** The ranking step is fully offline. The embedding model runs locally. No calls to OpenAI, Anthropic, Cohere, or any hosted service are made during ranking.

---

## AI Tools Declaration

AI tools (Claude, Gemini) were used as coding assistants during development. All architecture decisions, scoring weights, filter logic, and JD analysis were performed by the team. The code repository contains genuine engineering iteration, not a single-dump LLM output.

---

## Contact

See `submission_metadata.yaml` at the repository root for primary contact details and full portal metadata.
