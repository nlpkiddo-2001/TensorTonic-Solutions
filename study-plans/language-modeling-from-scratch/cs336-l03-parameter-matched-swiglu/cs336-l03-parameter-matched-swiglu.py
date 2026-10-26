import torch
import math 
import torch.nn.functional as F

def parameter_matched_swiglu(
    x: torch.Tensor, w_g: torch.Tensor, w_v: torch.Tensor,
    w_o: torch.Tensor, base_params: int,
) -> dict:
    """
    Returns output, hidden_width, and parameter_count in a dictionary.
    """
    d = x.shape[-1] 
    H = min(
        w_g.shape[1],
        w_v.shape[1],
        w_o.shape[0]
    )

    h = min(H, max(1, math.floor(base_params / (3 * d) + 0.5)))

    w_g_selected = w_g[:, :h]
    w_v_selected = w_v[:, :h]
    w_o_selected = w_o[:h, :]

    gate = F.silu(x @ w_g_selected)
    value = x @ w_v_selected 

    output =  (gate * value) @ w_o_selected

    return {
        "output" : output,
        "hidden_width" : int(h),
        "parameter_count" : int(3 * d * h)
    }
    
