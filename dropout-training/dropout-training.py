import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.array(x)
    rand = rng.random(x.shape) if rng is not None else np.random.random(x.shape)
    scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0
    mask = np.where(rand >= p, scale, 0.0)
    output = x * mask
    return (output, mask)