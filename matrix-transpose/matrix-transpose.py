import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.asarray(A)
    
    rows, cols = A.shape
    new_np_arr = np.zeros((cols, rows))

    for i in range(rows):
        for j in range(cols):
            new_np_arr[j][i] = A[i][j]

    return new_np_arr
    
