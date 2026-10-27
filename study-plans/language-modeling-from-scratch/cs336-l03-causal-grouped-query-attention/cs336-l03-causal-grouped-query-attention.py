import math
import torch

def causal_gqa(q, k, v):
    """
    Returns: causal grouped-query attention tensor
    """
    batch_size, query_heads, sequence_length, head_width = q.shape
    kv_heads = k.shape[1]
    group_size = query_heads // kv_heads

    grouped_q = q.float().reshape(batch_size, kv_heads, group_size, sequence_length, head_width)
    key_float = k.float()
    value_float = v.float()
    scores = torch.einsum("bhgsd,bhtd->bhgst", grouped_q, key_float) / math.sqrt(head_width)
    causal_mask = torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=q.device),
        diagonal=1,
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    grouped_output = torch.einsum("bhgst,bhtd->bhgsd", probabilities, value_float)
    return grouped_output.reshape(batch_size, query_heads, sequence_length, head_width).to(q.dtype)