import numpy as np
import time

from nmf.NonNegMatrix import NonNegMatrix
from nmf.NMF import NMF

class SepNMF(NMF):
    """
    Separable Non-negative Matrix Factorization (SepNMF)

    Implementation for factorizing a non-negative matrix V under the
    separability assumption: there exists and index set K with |K| = r s.t. V = V(:,K)H.

    
    The class provides a simple implementation of the Successive Nonnegative
    Projection Algorithm (SNPA). SNPA iteratively:
      1. selects the column j* with largest squared L2 norm in the current
         residual R,
      2. adds j* to the set K,
      3. solves a non-negative least-squares problem to compute coefficients
         H for V ≈ V[:, K] @ H,
      4. updates the residual R = V - V[:, K] @ H,
      5. repeats until `rank` columns are selected.

    Example
    -------
    model = SepNMF(V, rank=10)
    model.fit("SNPA")
    W, H = model.W, model.H

    References
    ----------
    - Gillis, N., & Vavasis, S. A. (2014). Fast and robust recursive algorithms
      for separable nonnegative matrix factorization. IEEE TPAMI.
    """

    def __init__(self, V: NonNegMatrix, rank: int, max_iter: int = 1000, tol: float = 0.0001, T: int = 10):
        super().__init__(V, rank, max_iter, tol, T)
    
    def fit(self, solver: str) -> None:
        """
        Fit the model using the chosen solver.

        Parameters
        ----------
        solver : str
            One of {"MU", "HALS", "SNPA"} selecting the update algorithm.
        """
        print(f"Fitting with {solver} algorithm")
        start_time = time.perf_counter()
        match solver:
            case "MU":
                self.W = np.abs(np.random.normal(self.m, self.rank)) # set W and H to random matrices if using MU
                self.H = np.abs(np.random.normal(self.rank, self.n))
                super().__mu_update__()
            case "HALS":
                super().__HALS_update__()
            case "SNPA":
                self.__SNPA_update__()
            case _:
                raise ValueError("Solver not found")
        end_time = time.perf_counter()
        self.fit_time = end_time - start_time
        self.n_iter = max(0, len(self.errors) - 1)
        self.time_per_iter = self.fit_time/self.n_iter if self.n_iter > 0 else float('inf')
        print(f"Fit completed in {self.fit_time:.4f} s, iterations: {self.n_iter}, avg time/iter: {self.time_per_iter:.4e} s")




    def __SNPA_update__(self) -> None:
        
        K = []
        R = self.V
        super().__compute_error__()
        for i in range(self.rank):
            # j_star = argmax_j ||R[:, j]||_2^2
            j_star = np.argmax(np.sum(R**2, axis=0))
            K.append(j_star)
            # H = argmin_{Y >= 0} ||V - V[:, K] Y||_2
            V_K = self.V[:, K]
            H = np.linalg.lstsq(V_K, self.V, rcond=None)[0]
            H = np.maximum(H, 0)  # enforce non-negativity
            self.H = H
            # R = V - V[:, K] H
            R = self.V - V_K @ H
            self.errors.append(np.linalg.norm(R))
        self.W = self.V[:, K]