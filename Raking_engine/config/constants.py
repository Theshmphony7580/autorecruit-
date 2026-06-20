"""
constants.py
============
Single source of truth for all configuration values.
Every weight, threshold, skill list, and path used by the ranking engine.
"""

# JD Core Skills (must-have for relevant candidates)
JD_CORE_SKILLS = {
    'embeddings', 'vector search', 'semantic search', 'information retrieval',
    'faiss', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'opensearch',
    'langchain', 'rag', 'llamaindex', 'prompt engineering',
    'hugging face', 'sentence transformers', 'transformers',
    'llms', 'machine learning', 'ml', 'nlp', 'recommendation systems',
}

# Specific Disqualifiers defined by the Job Description Policies
JD_DISQUALIFIED_TITLES = [
    'marketing', 'project manager', 'product manager', 
    'hr', 'recruiter', 'sales', 'accountant', 'designer'
]

JD_DISQUALIFIED_DOMAINS = [
    'computer vision', 'robotics', 'speech', 'hardware'
]

JD_ACCEPTED_LOCATIONS = [
    'pune', 'noida', 'delhi', 'ncr', 'mumbai', 'hyderabad', 'new delhi'
]

JD_CONSULTING_FIRMS = [
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini', 'tata consultancy'
]

# Non-ML Titles (penalized for keyword stuffing, but not hard-banned)
NON_ML_TITLES = {
    'marketing manager', 'sales executive', 'hr manager', 'accountant',
    'content writer', 'graphic designer', 'customer support', 'mechanical engineer',
    'civil engineer', 'operations manager', 'business analyst',
}

# Plain-language ML terms (catch Tier 5 engineers)
ML_CAREER_TERMS = [
    'recommendation', 'search', 'ranking', 'personalization',
    'matching', 'retrieval', 'embed', 'vector', 'semantic',
    'collaborative filtering', 'content-based', 'similarity',
]

# JD-RELEVANT SKILL BUNDLES (from apriori mining)
JD_RELEVANT_BUNDLES = [
    "Hugging Face Transformers, LLMs, Vector Search",
    "Hugging Face Transformers, Semantic Search, Vector Search",
    "LangChain, Pinecone, Prompt Engineering",
    "LangChain, Recommendation Systems, Sentence Transformers",
    "Information Retrieval, LLMs, Sentence Transformers",
    "LangChain, Semantic Search, Sentence Transformers",
    "Embeddings, Hugging Face Transformers, LLMs",
    "FAISS, Information Retrieval, LLMs",
    "Information Retrieval, LLMs, Vector Search",
    "LangChain, Pinecone, Sentence Transformers",
    "Information Retrieval, LLMs, Semantic Search",
    "LangChain, Prompt Engineering, Recommendation Systems",
    "LangChain, Prompt Engineering, RAG",
    "Hugging Face Transformers, LLMs, RAG",
    "Embeddings, Information Retrieval, LLMs",
    "LLMs, LangChain, Sentence Transformers",
    "FAISS, Hugging Face Transformers, LLMs",
    "LLMs, Semantic Search, Vector Search",
    "Pinecone, Prompt Engineering, Recommendation Systems",
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

JD_FIT_SUB_WEIGHTS = {
    'semantic_similarity': 0.50,
    'keyword_overlap': 0.30,
    'title_context': 0.20,
}


# Experience scoring limits
JD_EXPERIENCE_MIN = 5   # years
JD_EXPERIENCE_MAX = 9   # years
EXPERIENCE_DECAY = 5    # how many years outside the range before score = 0

JD_REALISTIC_SKILL_MAXIMUM = 10

# Behavior Scorer Normalization Ceilings
BEHAVIOR_RESPONSE_TIME_CEILING_HOURS = 72   # beyond 72h = 0 for responsiveness
BEHAVIOR_NOTICE_PERIOD_CEILING_DAYS  = 90   # beyond 90 days = 0 for availability

# Trust Scorer Normalization Ceilings
TRUST_ENDORSEMENTS_MAX     = 50   # 50+ endorsements = max endorsement score
TRUST_RECENCY_WINDOW_DAYS  = 365  # 1 year inactivity = 0 recency score

TOP_K = 100
OUTPUT_COLUMNS = ['candidate_id', 'rank', 'score', 'reasoning']
SCORE_PRECISION = 6
