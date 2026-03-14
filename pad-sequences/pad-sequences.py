import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if not seqs:
        return np.array((0, 0), dtype=int)

    if not max_len:
        max_len = max(len(arr) for arr in seqs)

    result = np.full((len(seqs), max_len), fill_value=pad_value, dtype=int)

    for i, seq in enumerate(seqs):
        length = min(len(seq), max_len)
        result[i, :length] = seq[:length]

    return result