import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Optional
from scipy import sparse
from nmf import NMF

class SparseNMF(NMF):

    def __init__(self, V: sparse.csr_array, r: int, init: Optional[str] = "random",
                 max_iter: int = 1000, tol: float = 1e-4, T: int  = 10, random_state: Optional[int] = None,
                beta: float = 2, W0: Optional[np.ndarray] = None, H0: Optional[np.ndarray] = None,
                ):
        """
        Sparse Non-negative Matrix Factorization (Sparse NMF) with L1 regularization.

        Parameters:
        - V: Input non-negative data matrix (m x n) as a sparse matrix.
        - r: Rank for the factorization.
        - init: Initialization method for W and H ('random' or 'nndsvd').
        - max_iter: Maximum number of iterations.
        - tol: Tolerance for the stopping condition.
        - random_state: Seed for random number generator.
        - alpha_W: Regularization parameter for W (L1 penalty).
        - alpha_H: Regularization parameter for H (L1 penalty).
        - beta: Beta divergence parameter (0, 1, or 2).
        """
        self.V = V 
        if V.min() < 0:
            raise ValueError("Input matrix V contains negative elements.")
        self.rank = r
        self.m, self.n = V.shape # type: ignore
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.T = T
        self.beta = beta
        match init:
            case "random":
                self.W = np.random.rand(self.m, self.rank)
                self.H = np.random.rand(self.rank, self.n)
            case "nndsvd":
                self.__NNSVD_init()
            case "custom":
                if W0 is None or H0 is None:
                    raise ValueError("Custom initialization requires W0 and H0.")
                self.W = W0
                self.H = H0
            case _:
                raise ValueError("Invalid initialization method.")
        self.errors = []  # To store reconstruction errors
        self.V_norm = self.compute_frobenius_norm(V)
   
    def compute_frobenius_norm(self, M) -> float:
        """Compute the Frobenius norm of a (possibly sparse) matrix."""
        if sparse.isspmatrix(M):
            return float(np.sqrt((M.data ** 2).sum()))
        else:
            return float(np.linalg.norm(M))
            

    
    