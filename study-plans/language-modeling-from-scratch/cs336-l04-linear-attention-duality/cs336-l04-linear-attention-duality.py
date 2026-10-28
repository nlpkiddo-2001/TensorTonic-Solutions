import torch

def linear_attention_duality(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> dict:
    """
    Returns a dict of tensors: parallel_output, recurrent_output, final_state.
    """
    B, S, Dk = q.shape
    Dv = v.shape[-1]

    scores = q @ k.transpose(-2, -1)

    mask = torch.tril(
        torch.ones((S,S), device=q.device, dtype=torch.bool)
    )

    scores = scores.masked_fill(~mask, 0)

    parallel_output = scores @ v

    

    # recurrent state starts
    state = q.new_zeros(B, Dk, Dv)
    recurrent_state = []
    
    for t in range(S):
        state = state + (
            k[:, t, :, None] * v[:, t, None, :]
        )

        y_t = torch.einsum(
            "bd,bdv->bv", q[:, t, :], state
        )

        recurrent_state.append(y_t)

    recurrent_output = torch.stack(recurrent_state, dim=1)

    final_state = torch.einsum('bsd,bsv->bdv', k, v)
    
    return {
        "parallel_output" : parallel_output,
        "recurrent_output" : recurrent_output,
        "final_state" : final_state
    }

    
