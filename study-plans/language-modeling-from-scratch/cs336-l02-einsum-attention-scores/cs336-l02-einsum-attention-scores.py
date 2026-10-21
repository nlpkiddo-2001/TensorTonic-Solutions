import torch
import numpy as np

def attention_scores(q: torch.Tensor, k: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Returns scores of shape (batch, heads, query_length, key_length).

    shape = q.shape
    
    """

    B, s_q, D = q.shape
    _ ,s_k, _ = k.shape

    head_dim = D // num_heads

    q = q.reshape(B, s_q, num_heads, head_dim) 
    k = k.reshape(B, s_k, num_heads, head_dim) 

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    scores = q @ k.transpose(-2, -1)

    return scores / math.sqrt(head_dim)

    
    
    
