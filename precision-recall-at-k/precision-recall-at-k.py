def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    # Write code here

    recommended = recommended[:k]
    
    recommended = set(recommended)
    relevant = set(relevant)

    union = recommended & relevant
    precision_score = len(union) / k

    recall_score = len(union) / len(relevant)

    return [precision_score, recall_score]