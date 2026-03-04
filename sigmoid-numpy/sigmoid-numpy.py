import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here

    x = np.asarray(x, dtype=float)
    
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))
    