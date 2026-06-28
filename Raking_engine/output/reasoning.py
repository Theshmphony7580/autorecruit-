"""
reasoning.py — Cluster-Aware Recruiter Assessment Engine

Solves information architecture challenges across the full 100-candidate cohort:
1. Score Neighborhood & Cluster Analysis: Identifies tight score brackets (<0.003 gap)
   and explicitly states the factual component (saves, notice, stack breadth) that separated tied candidates.
2. Global Evidence Deduplication: Tracks seen project descriptions across the cohort. If synthetic
   dataset repeats a project string, it suppresses verbatim repetition and adapts sentence structure.
3. Natural Sentence Count & Structural Variation: Dynamically varies between 2, 3, and 4 sentences,
   alternating sentence ordering (Constraint-lead, Project-lead, Concise 2-liner, Standard observational).
"""

import os
import sys
import re
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.constants import JD_CORE_SKILLS, JD_EXPERIENCE_MIN, JD_EXPERIENCE_MAX
from utils.safe_extract import safe_str, safe_float, safe_int


class ReasoningGenerator:
    def __init__(self, behavior_df):
        self.behavior_df = behavior_df
        self._jd_skills_lower = {s.lower() for s in JD_CORE_SKILLS}
        self._action_verbs = {
            'built', 'owned', 'designed', 'shipped', 'developed', 'led', 'implemented',
            'trained', 'migrated', 'launched', 'scaled', 'architected', 'engineered',
            'deployed', 'created', 'established', 'spearheaded', 'managed', 'directed',
            'optimised', 'optimized', 'fine-tuned', 'contributed', 'configured'
        }

    def generate(self, candidate: dict, score: float, rank: int = 0, total: int = 100) -> str:
        return self.generate_all([(score, candidate.get('candidate_id', ''), candidate)])[0]

    def generate_all(self, ranked: list) -> list:
        seen_projects = set()
        notes = []
        self._last_starter = -1

        for i, (score, cid, candidate) in enumerate(ranked):
            rank = i + 1
            above = ranked[i - 1] if i > 0 else None
            below = ranked[i + 1] if i < len(ranked) - 1 else None

            seed = int(hashlib.md5(cid.encode()).hexdigest(), 16)
            d = self._extract_facts(candidate, score, rank)

            # Check for project repetition in dataset
            p_txt = d['proof']['text']
            is_dup = False
            if p_txt:
                norm_p = p_txt.lower()[:35]
                if norm_p in seen_projects:
                    is_dup = True
                else:
                    seen_projects.add(norm_p)

            # Check if candidate is in a tight score cluster (< 0.003 diff with neighbor)
            in_cluster = False
            cluster_diff_note = ""
            if below and abs(score - below[0]) <= 0.003:
                in_cluster = True
                cluster_diff_note = self._get_cluster_differentiator(d, below[2])
            elif above and abs(above[0] - score) <= 0.003:
                in_cluster = True

            note, starter_idx = self._assemble_note(d, rank, seed, is_dup, in_cluster, cluster_diff_note, getattr(self, '_last_starter', -1))
            self._last_starter = starter_idx
            notes.append(note)

        return notes

    def _extract_facts(self, candidate: dict, score: float, rank: int) -> dict:
        profile = candidate.get('profile', {})
        skills_raw = candidate.get('skills', [])
        signals = candidate.get('redrob_signals', {})
        career = candidate.get('career_history', [])
        breakdown = candidate.get('_score_breakdown', {})

        title = safe_str(profile.get('current_title', '')).strip() or "ML Engineer"
        company = safe_str(profile.get('current_company', '')).strip() or "Tech"
        years = safe_float(profile.get('years_of_experience', 0))
        yr_str = f"{years:.0f}" if years == int(years) else f"{years:.1f}"

        matched = []
        if isinstance(skills_raw, list):
            for s in skills_raw:
                if isinstance(s, dict) and s.get('name'):
                    sname = safe_str(s['name']).strip()
                    if sname.lower() in self._jd_skills_lower and sname not in matched:
                        matched.append(sname)
        matched = matched[:4]

        notice = safe_int(signals.get('notice_period_days', 90), 90)
        saved = safe_int(signals.get('saved_by_recruiters_30d', 0))
        proof_obj = self._extract_clean_proof(career)

        # Observable concern / trade-off
        concern = ""
        if notice >= 90:
            concern = f"{notice}-day notice period is the primary concern."
        elif years < JD_EXPERIENCE_MIN:
            concern = f"Total tenure ({yr_str} yrs) falls under our {JD_EXPERIENCE_MIN}-year benchmark."
        elif years > JD_EXPERIENCE_MAX + 3:
            concern = f"Seniority ({yr_str} yrs) exceeds typical mid-level target bands."

        return {
            'title': title,
            'company': company,
            'years': yr_str,
            'raw_years': years,
            'skills': matched,
            'proof': proof_obj,
            'notice': notice,
            'saved': saved,
            'concern': concern,
            'breakdown': breakdown
        }

    def _get_cluster_differentiator(self, curr: dict, below_cand: dict) -> str:
        curr_bd = curr.get('breakdown', {})
        below_bd = below_cand.get('_score_breakdown', {})
        if not curr_bd or not below_bd:
            return ""

        if curr_bd.get('demand', 0) > below_bd.get('demand', 0) + 0.003:
            return f"In this tight scoring cluster, higher recruiter interest ({curr['saved']} saves) provided the ranking edge."
        elif curr_bd.get('behavior', 0) > below_bd.get('behavior', 0) + 0.003:
            return f"Within this close score bracket, a {curr['notice']}-day notice period gave them immediate availability advantage over peers."
        elif curr_bd.get('jd', 0) > below_bd.get('jd', 0) + 0.003:
            return "Edged out peer scores in this cluster due to broader semantic retrieval keyword coverage."
        elif curr_bd.get('exp', 0) > below_bd.get('exp', 0) + 0.003:
            return f"Differentiated within this rank cluster by optimal tenure alignment ({curr['years']} yrs)."
        return ""

    def _assemble_note(self, d: dict, rank: int, seed: int, is_dup: bool, in_cluster: bool, cluster_diff: str, prev_idx: int = -1):
        skills = d['skills']
        if not skills:
            sk_str = "Python and machine learning"
        elif len(skills) == 1:
            sk_str = skills[0]
        elif len(skills) == 2:
            sk_str = f"{skills[0]} and {skills[1]}"
        else:
            sk_str = f"{', '.join(skills[:-1])}, and {skills[-1]}"

        t, c, y = d['title'], d['company'], d['years']
        p_txt = d['proof']['text']
        p_act = d['proof']['is_action']
        concern = d['concern']

        # Determine structural layout (Varying information order & sentence count)
        layout = (seed + rank) % 4
        if layout == (prev_idx // 10):
            layout = (layout + 1) % 4

        if is_dup or (layout == 0 and not cluster_diff):
            # 2 Sentences: Identity + Skills & Constraint/Signal with 7 randomized starters
            id_styles = [
                f"{y} years experience as {t} at {c}, skilled in {sk_str}.",
                f"Currently {t} at {c} ({y} yrs total tenure), working across {sk_str}.",
                f"Brings {y} years of professional background at {c} as {t}, with core stack in {sk_str}.",
                f"{y}-year {t} at {c}, demonstrating strong technical overlap in {sk_str}.",
                f"Employed at {c} for {y} years as {t}, covering {sk_str}.",
                f"Solid {y} years of track record as {t} at {c}, aligned with {sk_str}.",
                f"Experienced {t} ({y} yrs at {c}) familiar with {sk_str}."
            ]
            idx = (seed + rank) % len(id_styles)
            if (0 * 10 + idx) == prev_idx:
                idx = (idx + 1) % len(id_styles)
            s1 = id_styles[idx]
            s2 = cluster_diff if cluster_diff else (concern if concern else (f"Active candidate saved by {d['saved']} recruiters recently." if d['saved'] >= 20 else "Strong core technical alignment."))
            return self._clean_text(f"{s1} {s2}"), (0 * 10 + idx)

        if layout == 1 and concern:
            # Constraint Lead with randomized starters
            c_styles = [
                f"Primary constraint is a {d['notice']}-day notice period." if d['notice'] >= 90 else concern,
                f"Note that a {d['notice']}-day notice period reduces immediate availability." if d['notice'] >= 90 else concern,
                f"Main consideration is the {d['notice']}-day onboarding timeline." if d['notice'] >= 90 else concern
            ]
            idx = (seed + rank) % len(c_styles)
            if (1 * 10 + idx) == prev_idx:
                idx = (idx + 1) % len(c_styles)
            s1 = c_styles[idx]
            id_styles = [
                f"Otherwise, they are a {y}-year {t} at {c} with strong overlap across {sk_str}.",
                f"Outside of this, candidate brings {y} years at {c} as {t} covering {sk_str}.",
                f"Technical alignment remains solid across {sk_str} from their {y} years as {t} at {c}."
            ]
            s2 = id_styles[(seed >> 1) % len(id_styles)]
            s3 = f"Built production record where they {p_txt}." if p_act else f"Project evidence highlights: {p_txt}."
            if cluster_diff:
                return self._clean_text(f"{s1} {s2} {cluster_diff}"), (1 * 10 + idx)
            return self._clean_text(f"{s1} {s2} {s3 if p_txt else ''}"), (1 * 10 + idx)

        if layout == 2 and p_txt:
            # Project Lead with randomized starters
            p_styles = [
                f"Directly {p_txt} in their role at {c}." if p_act else f"Key project evidence at {c}: {p_txt}.",
                f"Production track record at {c} shows they {p_txt}." if p_act else f"Practical deliverable highlights: {p_txt}.",
                f"At {c}, candidate successfully {p_txt}." if p_act else f"Notable work at {c} includes: {p_txt}."
            ]
            idx = (seed + rank) % len(p_styles)
            if (2 * 10 + idx) == prev_idx:
                idx = (idx + 1) % len(p_styles)
            s1 = p_styles[idx]
            id_styles = [
                f"Candidate is a {y}-year {t} covering {sk_str}.",
                f"Brings {y} years of tenure as {t}, skilled in {sk_str}.",
                f"Background covers {y} years as {t} with competency in {sk_str}."
            ]
            s2 = id_styles[(seed >> 1) % len(id_styles)]
            s3 = cluster_diff if cluster_diff else concern
            return self._clean_text(f"{s1} {s2} {s3}"), (2 * 10 + idx)

        # Layout 3: Standard Observational with randomized starters
        id_styles = [
            f"Currently {t} at {c} ({y} yrs total experience). Strong background across {sk_str}.",
            f"{t} at {c} bringing {y} years of experience. Demonstrated stack proficiency in {sk_str}.",
            f"Working as {t} at {c} for {y} years. Technical core includes {sk_str}."
        ]
        idx = (seed + rank) % len(id_styles)
        if (3 * 10 + idx) == prev_idx:
            idx = (idx + 1) % len(id_styles)
        s1_2 = id_styles[idx]
        p_styles = [
            f"Past work shows they {p_txt}." if (p_txt and p_act) else (f"Work history notes: {p_txt}." if p_txt else ""),
            f"In production, they {p_txt}." if (p_txt and p_act) else (f"Career summary highlights: {p_txt}." if p_txt else ""),
            f"Notably {p_txt}." if (p_txt and p_act) else (f"Deliverable evidence: {p_txt}." if p_txt else "")
        ]
        s3 = p_styles[(seed >> 1) % len(p_styles)] if p_txt else ""
        s4 = cluster_diff if cluster_diff else concern

        parts = [s1_2]
        if s3:
            parts.append(s3)
        if s4:
            parts.append(s4)

        return self._clean_text(" ".join(parts)), (3 * 10 + idx)

    def _extract_clean_proof(self, career: list) -> dict:
        if not career:
            return {'text': '', 'is_action': False}
        jd_terms = {'ranking', 'search', 'retrieval', 'recommendation', 'embedding', 'vector', 'semantic', 'rag', 'reranking', 'ndcg', 'faiss', 'pinecone', 'weaviate', 'opensearch', 'elasticsearch', 'pipeline', 'llm', 'inference'}
        best_sentence = ""
        for job in career:
            desc = safe_str(job.get('description', ''))
            for sent in re.split(r'(?<=[.!?])\s+', desc):
                clean_s = sent.strip().rstrip('.!?')
                if any(term in clean_s.lower() for term in jd_terms) and 25 <= len(clean_s) <= 145:
                    best_sentence = clean_s
                    break
            if best_sentence:
                break
        if not best_sentence:
            return {'text': '', 'is_action': False}

        dangling_pat = re.compile(r'\s+\b(a|an|the|in|on|at|by|for|from|to|with|into|onto|upon|via|using|serving|including|integrated|and|or|nor|but|yet|so|of)\b\s*$', re.IGNORECASE)
        for _ in range(4):
            if not dangling_pat.search(best_sentence):
                break
            best_sentence = dangling_pat.sub('', best_sentence).strip().rstrip(',;-')

        words = best_sentence.split()
        first_word = words[0].lower() if words else ""
        is_action = first_word in self._action_verbs
        if is_action:
            best_sentence = first_word + best_sentence[len(first_word):]
        return {'text': best_sentence, 'is_action': is_action}

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.replace('..', '.').replace(',.', '.').replace(' ,', ',').replace(' ;', ';')
        if not text.endswith('.'):
            text += '.'
        return text[0].upper() + text[1:]