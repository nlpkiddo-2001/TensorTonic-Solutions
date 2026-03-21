import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Your code here
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    norm = (x - mean) / np.sqrt(var + eps)
    return gamma * norm + beta
    

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    Q = Q @ W_q
    K = K @ W_k
    V = V @ W_v

    Q_h = Q.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K_h = K.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V_h = V.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    scores = softmax(Q_h @ K_h.transpose(0, 1, 3, 2) / np.sqrt(d_k), axis=-1)
    attended = scores @ V_h
    concat = attended.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    return concat @ W_o
    

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    hidden = np.maximum(0, x @ W1 + b1)    
    return hidden @ W2 + b2 

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    attn_out = multi_head_attention(x, x, x ,  W_q, W_k, W_v, W_o, num_heads)
    x = layer_norm(x + attn_out, gamma1, beta1)

    ff_out = feed_forward(x, W1, b1, W2, b2)
    x = layer_norm(x + ff_out, gamma2, beta2)

    return x