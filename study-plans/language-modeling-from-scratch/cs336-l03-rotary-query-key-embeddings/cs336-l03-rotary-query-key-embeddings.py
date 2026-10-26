import torch

def rotary_embed(
    q: torch.Tensor, k: torch.Tensor,
    positions: torch.Tensor, inv_freq: torch.Tensor,
) -> dict:
    """
    Returns q_rotated and k_rotated tensors in a dictionary.
    """

    def rotate(x):
        B, H, S, D = x.shape
        x_dtype = x.dtype

        # splitting into pairs
        pairs = x.float().reshape(B, H, S, D // 2, 2)
        x_0, x_1 = pairs[..., 0], pairs[..., 1]

        # converting positions, inv to same dtype and reshaping into (B, S)
        pos = positions.to(device=x.device, dtype=torch.float32)
        freq = inv_freq.to(device=x.device, dtype=torch.float32)

        if pos.ndim == 1:
            pos = pos.unsqueeze(0)

        # calculate theta
        theta = pos[:, None, :, None] * freq[None, None, None, :]

        # calculate cos and sin
        cos = torch.cos(theta)
        sin = torch.sin(theta)

        # calculate new positions
        y0 = x_0 * cos - x_1 * sin
        y1 = x_0 * sin + x_1 * cos

        # stack them
        return torch.stack((y0, y1), dim=-1).reshape(B,H,S,D).to(x_dtype)


    return {
        "q_rotated" : rotate(q),
        "k_rotated" : rotate(k)
    }

        

    
