import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    col_idx = np.arange(d_model)

    i = col_idx // 2 # let's say if dim=512, then i = 256 because i (0,1) in same pair

    div_terms = np.power(base, (2*i) / d_model)

    positions = np.arange(seq_len)[:,np.newaxis]

    angles = positions / div_terms

    PE = np.where(col_idx % 2 == 0, np.sin(angles), np.cos(angles))

    return PE.astype(float)
    