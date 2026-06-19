# Redrob "Senior AI Engineer" Ranking Engine
**Product Requirements Document (PRD) & Implementation Guide**

## 1. Executive Summary & Objective
This document outlines the architecture, logic, and step-by-step implementation plan for the **Redrob AI Ranking Engine**. 
The objective is to ingest a dataset of **100,000 synthetic candidate profiles** and output a perfectly formatted CSV containing the **Top 100 candidates** for a "Founding Senior AI Engineer" role.

**Strict Hackathon Constraints:**
- **Hardware:** CPU-only, ≤ 16 GB RAM.
- **Execution Time:** Must run end-to-end in **under 5 minutes** (this dictates our use of pre-computed arrays and streaming instead of runtime LLM processing).
- **Environment:** No network access, no hosted LLM APIs during runtime. 

**The Core Challenge:** 
The dataset is synthetic and riddled with "honeypots" (fake profiles with logical timeline inconsistencies) and "keyword stuffers" (e.g., Marketing Managers who just listed all AI keywords). A naive keyword-count ranker will fail. Our system wins by prioritizing **semantic JD fit**, **strict trust/honeypot filtering**, and **real-world AI skill bundles**.

---

## 2. System Architecture: 5-Stage Cascade Ranker

To meet the 5-minute and 16GB memory constraints, the engine (`rank.py`) will not load all 100k full JSON profiles into memory at once. It uses a **Streaming Multi-Cascade Architecture**.

### 🔹 Stage 1: Streaming Ingestion & Fast Loading
Instead of `pd.read_json()`, we stream the `candidates.jsonl` file line-by-line. We simultaneously load our **pre-computed forensic scores** into memory (which are small, fast Pandas DataFrames/NumPy arrays).

### 🔹 Stage 2: Hard Filtering (The "Trust & Honeypot" Gate)
Before doing any heavy math, we immediately eliminate:
1. **Honeypots:** Candidates flagged with chronological impossibilities (e.g., `signup_date > last_active_date`, overlapping education/experience anomalies).
2. **Keyword Stuffers:** Candidates possessing Tier 3/4 AI skills (like "LLMs", "RAG") but whose job titles and descriptions contain zero matching narrative text.

### 🔹 Stage 3: Semantic JD-Fit (Vector Similarity)
For the candidates that pass Stage 2, we calculate their alignment with the Job Description. 
- **What:** We compare the candidate's pre-computed embedding vector against the JD's embedding vector using **Cosine Similarity**.
- **Why:** This ensures we find people who *do* the work (applied ML context), not just people who *listed* the words.

### 🔹 Stage 4: Forensic Composite Scoring
Every surviving candidate receives a final composite score `[0.0 to 100.0]`. This is a weighted sum:
- **35% - Quality / JD Semantic Fit:** (Stage 3 cosine similarity normalized).
- **30% - Behavioral Score:** Pre-computed metrics defining recruiter demand, responsiveness, and availability.
- **20% - Skill Coherence:** Extra points if their skills map to Apriori-mined bundles (e.g., if they have `LangChain`, do they also have `Vector Search` and `LLMs`?).
- **15% - Trust / Timeline Continuity:** How perfectly their claimed "Years of Experience" matches their career history array.

### 🔹 Stage 5: Selection, Reasoning & Output format
- Sort the final pool descending by composite score.
- Slice the Top 100.
- Generate a concise 1-2 sentence `reasoning` string for each candidate (e.g., *"Top-tier semantic match with strong behavioral signals and verified AI skill bundles."*).
- Export to `submission.csv` strictly adhering to the schema: `candidate_id, rank, score, reasoning`.

---

## 3. Data Flow & Dependencies (What comes from where?)

If you are a Junior Dev picking this up, here is exactly where our data lives and what it does:

| Asset | Location | Purpose | How it's used in `rank.py` |
|-------|----------|---------|----------------------------|
| **Raw Candidates** | `data_forensic _files/candidates.jsonl` | The main 100k profile dataset. | Streamed line-by-line to extract `candidate_id`, `skills`, and textual summary for the final reasoning string. |
| **Behavioral Scores** | `data_forensic _files/candidate_behavior_scores_full.csv` | Pre-computed dataset containing `behavior_score`, `trust_score`, and `honeypot_label`. | Loaded into a Pandas DataFrame, indexed by `candidate_id` for instant O(1) lookups during streaming. |
| **Skill Bundles** | `data_forensic _files/frequent_skill_bundles.csv` | Association rules (Apriori) proving what real AI engineers know. | Used as a dictionary set. If a candidate's skills intersect perfectly with a bundle, they get a multiplier. |
| **Candidate Embeddings** | `data_forensic _files/candidate_embeddings.npy` (or similar) | The mathematical representation of the candidate's text. | Loaded via `numpy.load()` and matrix-multiplied against the JD embedding for rapid cosine similarity. |
| **Job Description** | `jd_extracted.txt` / `job_description.docx` | The target we are matching against. | We will parse this to create the base JD embedding vector. |

---

## 4. Directory & File Structure

```text
F:\autorecruit-\
│
├── Raking_engine/               <-- ALL NEW WORK GOES HERE
│   ├── PRD.md                   <-- (This Document)
│   ├── rank.py                  <-- THE MAIN EXECUTABLE SCRIPT
│   ├── ranker_utils.py          <-- Helper functions (JSONL streaming, math)
│   ├── submission_validator.py  <-- Automated test to guarantee we don't get disqualified
│   └── submission.csv           <-- The final generated output!
│
├── data_forensic _files/        <-- PRE-COMPUTED ARTIFACTS (DO NOT MODIFY)
│   ├── candidates.jsonl         
│   ├── candidate_behavior_scores_full.csv
│   ├── frequent_skill_bundles.csv
│   └── ...
│
└── resources provided by the hackthon/
    ├── submission_spec.docx     <-- Hackathon rules
    └── sample_submission.csv    <-- Reference format
```

---

## 5. Detailed Implementation Plan for `rank.py`

When writing the code, follow this exact sequence:

**Step 1: Argument Parsing & Setup**
- Use `argparse` to accept `--candidates` (path to jsonl) and `--out` (path to submission.csv). 
- *Why:* Makes the script Docker/Submission ready.

**Step 2: Load Pre-computed Forensic Assets into RAM**
- Load `candidate_behavior_scores_full.csv` into a Pandas DataFrame `df_scores`. Set the index to `candidate_id` for extremely fast lookups.
- *Memory note:* A 100k row dataframe with ~10 float columns takes less than 50MB of RAM.

**Step 3: The Streaming Loop (O(N) Processing)**
```python
top_candidates_heap = [] # Use a min-heap to keep track of the Top 100 in O(N log K) time
with open(args.candidates, 'r') as f:
    for line in f:
        candidate = json.loads(line)
        cid = candidate['candidate_id']
        
        # 1. Fetch forensic data from the pre-loaded DataFrame
        forensic_data = df_scores.loc[cid]
        
        # 2. Hard Drop Criteria
        if forensic_data['honeypot_label'] == 1:
            continue # Drop immediately
            
        # 3. Calculate Score
        # score = (0.35 * JD_Fit) + (0.30 * behavior) + ...
        
        # 4. Push to Heap
        # If heap < 100, push. If score > heap_min, pop min and push new.
```

**Step 4: Generate Reasoning Strings**
- For the winning 100 candidates, extract their top skills and key behavioral metric to generate the required reasoning sentence.
- Example: *"Matched JD with 92% semantic fit; top 5% in recruiter responsiveness; verified elite AI skill bundle (LangChain, Pinecone, LLMs)."*

**Step 5: Formatting and Exporting**
- Sort the Top 100 array descending by score.
- Assign a `rank` from 1 to 100.
- Create a Pandas DataFrame with strictly 4 columns: `['candidate_id', 'rank', 'score', 'reasoning']`.
- Export using `df.to_csv(args.out, index=False)`.

---

## 6. Validation & Anti-Disqualification Checklist

The hackathon organizers are strict. If the CSV is slightly malformed, we get an automatic zero. The `submission_validator.py` MUST assert:

1. `df.shape == (100, 4)`
2. `list(df.columns) == ['candidate_id', 'rank', 'score', 'reasoning']`
3. `df['rank'].min() == 1` and `df['rank'].max() == 100`
4. `df['score'].is_monotonic_decreasing == True` (Rank 1 must have the highest score, Rank 100 the lowest).
5. No missing values (`df.isna().sum().sum() == 0`).
6. All `candidate_id`s must be unique.

## 7. Developer Handoff Note
If you are a Junior Dev starting the implementation:
1. Do not try to read `candidates.jsonl` into memory all at once. Use Python's built-in `open()` and `json.loads(line)`.
2. Do not recalculate behavioral scores. The heavy math is already done in the `candidate_behavior_scores_full.csv`.
3. Start by creating `Raking_engine/rank.py` and scaffold the imports and argument parser.
