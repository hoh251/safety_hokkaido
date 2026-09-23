def calculate_hit_rate(results, expected_ids):
    """
    Hit Rate checks if ANY of the correct expected_ids were found in the results.
    1.0 means it was found, 0.0 means the database completely missed it.
    """
    retrieved_ids = [res["chunk_id"] for res in results]
    for exp_id in expected_ids:
        if exp_id in retrieved_ids:
            return 1.0
    return 0.0

def calculate_mrr(results, expected_ids):
    """
    Mean Reciprocal Rank (MRR) checks HOW HIGH the correct answer was ranked.
    If it was Rank #1, score is 1.0. If Rank #2, score is 0.5. If Rank #3, score is 0.33.
    """
    retrieved_ids = [res["chunk_id"] for res in results]
    for rank, ret_id in enumerate(retrieved_ids):
        if ret_id in expected_ids:
            return 1.0 / (rank + 1)
    return 0.0
