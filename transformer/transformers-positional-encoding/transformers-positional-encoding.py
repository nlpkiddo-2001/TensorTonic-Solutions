import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here

    base=10000.0
    col_idx = np.arange(d_model)
    i = col_idx // 2
    divisors = np.power(base, (2*i)/d_model)
    positions = np.arange(seq_length)[:,np.newaxis]
    angles = positions / divisors
    PE = np.where(col_idx%2==0, np.sin(angles), np.cos(angles))
    return PE.astype(float)