# %%
import json

candidates = []

with open("F:\\autorecruit-\\data_forensic _files\\candidates.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        candidates.append(json.loads(line))

# %%
import json
from tqdm import tqdm

summaries = []

with open("F:\\autorecruit-\\data_forensic _files\\candidates.jsonl", "r") as f:

    for line in tqdm(f):

        candidate = json.loads(line)

        summaries.append(
            candidate["profile"]["summary"]
        )

print("Total summaries:", len(summaries))

# %% [markdown]
# **How many summaries are literally identical?**

# %%
from collections import Counter

summary_counter = Counter(summaries)


print("Unique summaries:",
      len(summary_counter))
print(summary_counter)

# %%
summaries

# %%
summary_counts = summary_counter.most_common()
summary_counts

# %%


# %%
for summary, count in summary_counter.most_common(50):

    print("\nCOUNT:", count)

    print(summary[:])

    print("="*100)

# %% [markdown]
# **Normalize Summaries**

# %%
import re

def normalize_summary(text):

    text = text.lower()

    text = re.sub(
        r'\d+(\.\d+)?',
        'NUMBER',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()

normalized = [
    normalize_summary(s)
    for s in summaries
]
normalized[:20]

# %%
norm_counter = Counter(normalized)

print(
    "Unique normalized summaries:",
    len(norm_counter)
)

# %% [markdown]
# **View Largest Templates**

# %%
norm_counter.most_common()

# %%
lengths = [
    len(s.split())
    for s in summaries
]

length_counter = Counter(lengths)
length_counter

# %%
import numpy as np

print("Min:", np.min(lengths))
print("Mean:", np.mean(lengths))
print("Median:", np.median(lengths))
print("Max:", np.max(lengths))

# %%
import pandas as pd
pd.Series(normalized)\
.value_counts()\
.head(50)

# %% [markdown]
# ****step 2.2****

# %%
descriptions = []

for c in candidates:

    for job in c["career_history"]:

        descriptions.append(
            job["description"]
        )

print(
    "Total descriptions:",
    len(descriptions)
)

# %%
from collections import Counter

desc_counter = Counter(
    descriptions
)

print(
    "Unique descriptions:",
    len(desc_counter)
)

total = len(descriptions)

unique = len(desc_counter)

print(
    "Reuse Ratio:",
    round(
        unique/total,
        4
    )
)

# %%
for desc, count in desc_counter.most_common(50):

    print("\nCOUNT:", count)

    print(desc[:700])

    print("="*120)

# %%
import re

def normalize_desc(text):

    text = text.lower()

    text = re.sub(
        r'\d+(\.\d+)?',
        'NUMBER',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()

normalized_desc = [
    normalize_desc(d)
    for d in descriptions
]

# %%
normalized_desc

# %%
norm_counter = Counter(
    normalized_desc
)

print(
    "Normalized Templates:",
    len(norm_counter)
)

# %%
for desc, count in norm_counter.most_common():

    print(count)
    print(desc[:500])

# %%
lengths = [
    len(d.split())
    for d in descriptions
]

lengths
import numpy as np

print(np.mean(lengths))
print(np.median(lengths))
print(np.min(lengths))
print(np.max(lengths))

# %% [markdown]
# ***step 2.3***

# %%
import pandas as pd

rows = []

for c in candidates:

    cid = c["candidate_id"]

    for skill in c["skills"]:

        rows.append({
            "candidate_id": cid,
            "skill": skill["name"]
        })

skills_df = pd.DataFrame(rows)
skills_df


# %%
skill_matrix = pd.crosstab(
    skills_df["candidate_id"],
    skills_df["skill"]
)
skill_matrix

# %%
corr_matrix = skill_matrix.corr()
corr_matrix

# %%
from mlxtend.frequent_patterns \
    import apriori


frequent = apriori(
    skill_matrix.astype(bool),
    min_support=0.01,
    use_colnames=True
)

# %%
frequent.sort_values(
    "support",
    ascending=False
)

# %%
frequent[
    frequent["itemsets"]
    .apply(lambda x: len(x) >= 3)
] \
.sort_values(
    "support",
    ascending=False
) \
.head(50)

# %% [markdown]
# ![image.png](attachment:image.png)

# %%
frequent_3 = frequent[
    frequent["itemsets"].apply(
        lambda x: len(x) >= 3
    )
].copy()

frequent_3["bundle"] = (
    frequent_3["itemsets"]
    .apply(lambda x: ", ".join(sorted(list(x))))
)

frequent_3 = frequent_3.sort_values(
    "support",
    ascending=False
)

frequent_3_df = pd.DataFrame({
    "support": frequent_3["support"],
    "bundle": frequent_3["bundle"],})

frequent_3_df[ "bundle" ]
frequent_3_df.to_csv("frequent_skill_bundles.csv", index=False)

# %%
unique_skills = skills_df["skill"].nunique()

print(unique_skills)

# %%
skills_df["skill"].value_counts().head(60)
skills_df["skill"].value_counts().to_csv("skill_frequencies.csv", index=True)

# %%
skills_per_candidate = (
    skills_df.groupby("candidate_id")
             .size()
)

print(skills_per_candidate.describe())

# %%
skills_df.groupby("candidate_id")["skill"] \
         .apply(list) \
         .head(20).to_csv("candidate_skills_2.csv", index=True)

# %% [markdown]
# step 2.4 reverse engineer the templates

# %%
candidate_docs = []

for c in candidates:

    skills = " ".join(
        s["name"]
        for s in c["skills"]
    )

    titles = " ".join(
        job["title"]
        for job in c["career_history"]
    )

    descriptions = " ".join(
        job["description"]
        for job in c["career_history"]
    )

    text = f"""
    {c['profile']['headline']}
    {c['profile']['summary']}
    {skills}
    {titles}
    {descriptions}
    """

    candidate_docs.append(text)
    
candidate_docs

# %%
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = model.encode(
    candidate_docs,
    show_progress_bar=True
)

# %%
embeddings

# %%
import numpy as np
np.save(
    "candidate_embeddings.npy",
    embeddings
)


