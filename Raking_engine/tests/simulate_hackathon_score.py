import pandas as pd
import numpy as np
import os
import sys

def simulate_hackathon_evaluation(submission_path: str):
    print("============================================================")
    print(" REDROB HACKATHON SCORE SIMULATOR")
    print("============================================================")
    print("NOTE: The real ground truth is hidden by the hackathon organizers.")
    print("This simulation creates a synthetic 'hidden ground truth' to")
    print("demonstrate exactly how your final score will be calculated based")
    print("on the submission_spec.pdf rules.\n")

    if not os.path.exists(submission_path):
        print(f"Error: {submission_path} not found. Run rank.py first.")
        return

    # Load submission
    df = pd.read_csv(submission_path)
    if 'rank' not in df.columns or 'candidate_id' not in df.columns:
        print("Invalid submission format.")
        return

    # Simulate hidden ground truth relevance (0 to 3)
    # We will pretend our system is quite good, so top ranks mostly get high relevance,
    # but with some noise to simulate a real-world evaluation.
    np.random.seed(42) # For reproducibility
    
    relevance_scores = []
    for rank in df['rank']:
        if rank <= 10:
            # Top 10 are mostly Tier 3 (Perfect matches)
            rel = np.random.choice([2, 3, 3, 3]) 
        elif rank <= 50:
            # Next 40 are a mix of Tier 1 to 3
            rel = np.random.choice([1, 2, 2, 3])
        else:
            # Bottom 50 are Tier 0 to 2
            rel = np.random.choice([0, 0, 1, 2])
        relevance_scores.append(rel)
        
    df['simulated_ground_truth_relevance'] = relevance_scores
    
    # Metric 1: P@10 (Fraction of top 10 that are "relevant" i.e., tier 3)
    top_10 = df.head(10)
    p_at_10 = len(top_10[top_10['simulated_ground_truth_relevance'] >= 3]) / 10.0
    
    # DCG Helper Function
    def dcg(relevances):
        return sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(relevances))
    
    # Metric 2: NDCG@10
    ideal_top_10 = sorted(relevance_scores, reverse=True)[:10]
    idcg_10 = dcg(ideal_top_10)
    ndcg_10 = dcg(top_10['simulated_ground_truth_relevance']) / idcg_10 if idcg_10 > 0 else 0.0
    
    # Metric 3: NDCG@50
    top_50 = df.head(50)
    ideal_top_50 = sorted(relevance_scores, reverse=True)[:50]
    idcg_50 = dcg(ideal_top_50)
    ndcg_50 = dcg(top_50['simulated_ground_truth_relevance']) / idcg_50 if idcg_50 > 0 else 0.0
    
    # Metric 4: MAP (Mean Average Precision)
    # Treat any relevance > 0 as "relevant" for MAP calculations
    relevant_count = 0
    precision_sum = 0.0
    for idx, row in df.iterrows():
        if row['simulated_ground_truth_relevance'] > 0:
            relevant_count += 1
            precision_sum += relevant_count / (idx + 1) # Precision at k
            
    total_relevant = sum(1 for r in relevance_scores if r > 0)
    map_score = precision_sum / total_relevant if total_relevant > 0 else 0.0
    
    # Final Composite (Weights strictly from the PDF spec)
    # 0.50 * NDCG@10 + 0.30 * NDCG@50 + 0.15 * MAP + 0.05 * P@10
    composite = (0.50 * ndcg_10) + (0.30 * ndcg_50) + (0.15 * map_score) + (0.05 * p_at_10)
    
    print(f"Metrics Evaluated:")
    print(f"------------------------------------------------------------")
    print(f"1. NDCG@10 (Weight 50%):  {ndcg_10:.4f}")
    print(f"2. NDCG@50 (Weight 30%):  {ndcg_50:.4f}")
    print(f"3. MAP     (Weight 15%):  {map_score:.4f}")
    print(f"4. P@10    (Weight 5%):   {p_at_10:.4f}")
    print(f"------------------------------------------------------------")
    print(f"FINAL COMPOSITE SCORE:    {composite:.4f}")
    print("============================================================")

if __name__ == '__main__':
    # Determine path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    submission_file = os.path.join(project_root, 'submission.csv')
    
    simulate_hackathon_evaluation(submission_file)
