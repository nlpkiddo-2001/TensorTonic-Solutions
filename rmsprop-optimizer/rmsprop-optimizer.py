import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # Write code here

    w = np.array(w, dtype=float)
    g = np.array(g, dtype=float)
    s = np.array(s, dtype=float)
    
    s_new = beta * s + (1 - beta) * g**2

    w_new = w - lr / (np.sqrt(s_new + eps)) * g

    return (w_new, s_new)