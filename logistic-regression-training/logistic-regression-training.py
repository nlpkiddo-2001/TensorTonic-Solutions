import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for i in range(steps):
        linear_model = np.dot(X, weights) + bias 
        y_pred = _sigmoid(linear_model)

        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)

        weights -= lr * dw
        bias -= lr * db

    return (weights, bias)
    

    
        