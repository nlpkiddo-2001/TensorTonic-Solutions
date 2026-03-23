import numpy as np

def relu(x):
    return np.maximum(0, x)

class IdentityBlock:
    """
    Identity Block: F(x) + x
    Used when input and output dimensions match.
    """
    
    def __init__(self, channels: int):
        self.channels = channels
        # Simplified: using dense layers instead of conv for demo
        self.W1 = np.random.randn(channels, channels) * 0.01
        self.W2 = np.random.randn(channels, channels) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = ReLU(W2 @ ReLU(W1 @ x)) + x
        """
        original_shape = x.shape
        batch = original_shape[0]
        channels = original_shape[1]
        
        # Flatten spatial dims into one
        x_flat = x.reshape(batch, channels, -1)        # (batch, C, spatial)
        
        # Transpose to (batch, spatial, C) for matmul, then back
        out = x_flat.transpose(0, 2, 1) @ self.W1.T    # (batch, spatial, C)
        out = relu(out)
        out = out @ self.W2.T                           # (batch, spatial, C)
        
        out = out.transpose(0, 2, 1)                    # (batch, C, spatial)
        out = out.reshape(original_shape)
        
        # Skip connection — NO final ReLU (identity must pass through as-is)
        return out + x
        