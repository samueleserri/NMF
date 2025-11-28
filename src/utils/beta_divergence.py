import numpy  as np

"""
    This file contains the implementation of the beta divergence and the relative loss used in Multiplicative Updates for β-NMF
    -----------------------
    The beta loss is D_\beta(V, WH, beta) = sum_{i,j} d_\beta(V[i,j], WH[i,j])
    where d_\beta is the beta divergence
    d_\beta(x,y) = x/y - log(x/y) - 1    if beta = 0
    d_\beta(x,y) = x * log(x/y) - x + y    if beta = 1
    d_\beta(x,y) = (x^beta + (beta-1) y^beta - beta x y^{beta-1}) / (beta (beta-1))
    ------------------
    references: https://arxiv.org/abs/1010.1763

"""

def beta_divergence(x, y, beta: float):
    """
      - beta == 0: d(x,y) = x/y - log(x/y) - 1   (Itakura-Saito)
      - beta == 1: d(x,y) = x * log(x/y) - x + y (KL)
      - otherwise: d(x,y) = (x^beta + (beta-1) y^beta - beta x y^{beta-1}) / (beta (beta-1))
    """
    eps = 1e-10
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_safe = np.maximum(y, eps)

    if beta == 0:
        return x / y_safe - np.log(x / y_safe) - 1
    elif beta == 1:
        # define 0*log(0/...) = 0 via np.where to avoid NaNs
        term = np.where(x > 0, x * np.log(x / y_safe), 0.0)
        return term - x + y
    else:
        return (x**beta + (beta - 1) * (y_safe**beta) - beta * x * (y_safe**(beta - 1))) / (beta * (beta - 1))


def beta_loss(X: np.ndarray, Y: np.ndarray, beta: float) -> float:
    """
    Sum of element-wise beta-divergence between X and Y (vectorized).
    """
    return float(np.sum(beta_divergence(X, Y, beta)))

