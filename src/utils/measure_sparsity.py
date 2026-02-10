import numpy as np
from scipy import sparse

def measure_sparsity(X: np.ndarray, tol: float = 1e-10) -> float:
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
    if sparse.issparse(X):
        nnz = X.data.shape[0] #type: ignore
        total = X.shape[0] * X.shape[1]
        return 1.0 - nnz / float(total)
    else:
        X = np.asarray(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        nnz = np.count_nonzero(np.abs(X) > tol)
        return 1.0 - nnz / float(X.size)