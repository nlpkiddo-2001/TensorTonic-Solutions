import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    points = np.array(points, dtype=float)
    single = points.ndim == 1

    if single:
        points = points[np.newaxis, :]

    ones = np.ones((points.shape[0], 1))
    ph = np.hstack([points, ones])

    transformed = (T @ ph.T).T
    result = transformed[:, :3]

    return result[0] if single else result
    