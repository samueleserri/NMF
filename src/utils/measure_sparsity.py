import numpy as np

def measure_sparsity(X: np.ndarray, tol: float = 1e-5) -> float:
    """
    Measure sparsity treating values with absolute magnitude <= tol as zero.

    Parameters
    ----------
    X : np.ndarray
        Input array.
    tol : float, optional
        Threshold below which values are considered zero.

    Returns
    -------
    float
        Fraction of entries equal to (approximately) zero: 1 - (nonzeros / total).
    """
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    nnz = np.count_nonzero(np.abs(X) > tol)
    return 1.0 - nnz / float(X.size)