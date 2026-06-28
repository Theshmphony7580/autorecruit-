import argparse
import time
import os
import sys
import pandas as pd
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.paths import DEFAULT_CANDIDATES_PATH, DEFAULT_BEHAVIOR_CSV, DEFAULT_JD_SIMILARITY, DEFAULT_OUTPUT_PATH
from loaders.jsonl_stream import stream_candidates
from loaders.csv_loader import load_behavior_scores, load_jd_similarity
from filters.hard_filters import HardFilter
from filters.jd_skill_filter import JDSkillFilter
from filters.career_context import CareerContextFilter, compute_stuffer_penalty
from scorers.jd_fit import JDFitScorer
from scorers.skill_bundle import BundleScorer
from composite.formula import CompositeScorer
from output.top_k import TopKTracker
from output.reasoning import ReasoningGenerator
from utils.validators import validate_submission
from utils.timing import Timer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument('--output', default=DEFAULT_OUTPUT_PATH)
    parser.add_argument('--behavior', default=DEFAULT_BEHAVIOR_CSV)
    parser.add_argument('--jd-similarity', default=DEFAULT_JD_SIMILARITY)
    FLAGS = parser.parse_args()
    
    start_time = time.time()
    print("=" * 60)
    print("RANKING ENGINE STARTED")
    print("=" * 60)
    
    print("\n[1/6] Loading pre-computed assets...")
    behavior_df = load_behavior_scores(FLAGS.behavior)
    jd_sim_df = load_jd_similarity(FLAGS.jd_similarity)
    
    print("\n[2/6] Initializing filters and scorers...")
    hard_filter = HardFilter(behavior_df)
    jd_filter = JDSkillFilter()
    career_filter = CareerContextFilter()
    jd_scorer = JDFitScorer()
    bundle_scorer = BundleScorer()
    composite_scorer = CompositeScorer(behavior_df, jd_sim_df, jd_scorer, bundle_scorer)
    
    print("\n[3/6] Initializing Top-100 tracker...")
    top100 = TopKTracker(k=100)
    
    print("\n[4/6] Processing candidates...")
    processed = 0
    filtered = 0
    
    with Timer("Candidate Streaming & Scoring"):
        for candidate in stream_candidates(FLAGS.candidates):
            cid = candidate['candidate_id']
            processed += 1
            
            if hard_filter.is_rejected(candidate):
                filtered += 1
                continue
                
            if not jd_filter.passes(candidate):
                filtered += 1
                continue
                
            if not career_filter.passes(candidate):
                filtered += 1
                continue
                
            stuffer_penalty = compute_stuffer_penalty(candidate)
            score = composite_scorer.compute_score(candidate, cid, stuffer_penalty)
            top100.push(cid, score, candidate)
            
            if processed % 10000 == 0:
                print(f"  Processed {processed}, Filtered {filtered}, In heap {len(top100.heap)}")
                
    print(f"\n  Total processed: {processed}")
    print(f"  Total filtered: {filtered}")
    print(f"  In scoring pool: {processed - filtered}")
    
    print("\n[5/6] Generating output...")
    ranked = top100.get_ranked()
    reason_gen = ReasoningGenerator(behavior_df)
    
    results = []
    explanations = reason_gen.generate_all(ranked)
    for rank, (score, cid, candidate) in enumerate(ranked, 1):
        results.append({
            'candidate_id': cid,
            'rank': rank,
            'score': round(score, 6),
            'reasoning': explanations[rank - 1]
        })
        
    print("\n[6/6] Writing submission.csv...")
    result_df = pd.DataFrame(results)
    result_df.to_csv(FLAGS.output, index=False)
    
    print("\nRunning validation...")
    validate_submission(result_df, FLAGS.candidates)
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"RANKING COMPLETED IN {elapsed:.2f} seconds")
    print(f"{'=' * 60}")
    
    honeypot_ids = set(behavior_df[behavior_df['honey_pot_labels'].isin(['suspicious', 'high_risk'])].index)
    honeypot_count = sum(1 for r in results if r['candidate_id'] in honeypot_ids)
    print(f"Honeypots in top 100: {honeypot_count}/100 ({honeypot_count}%)")
    
    if honeypot_count > 10:
        print("⚠️  WARNING: Honeypot rate exceeds 10%!")

if __name__ == '__main__':
    main()
