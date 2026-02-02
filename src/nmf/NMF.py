import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Optional
from nmf.NonNegMatrix import NonNegMatrix
from utils.beta_divergence import beta_loss
from sklearn.decomposition import TruncatedSVD

class NMF:
    """"
    Non-negative Matrix Factorization (NMF).
    Description
    -----------
    This class implements a basic NMF solver with the following update strategies:
    - Multiplicative Updates (MU)
    - beta Multiplicative Updates (beta_MU)
    - Hierarchical Alternating Least Squares (HALS)
    - Alternating Least Squares (ALS)
    Given a non-negative input matrix V (m x n) and a target factorization rank r,
    the model approximates V ≈ W @ H with W (m x r) and H (r x n), both constrained
    to be non-negative.
    Parameters
    ----------
    V : NonNegMatrix
        Non-negative data matrix to factorize with shape (m, n).
    rank : int
        Target factorization rank r (number of components).
    max_iter : int, optional (default=1000)
        Maximum number of iterations for the chosen update algorithm.
    tol : float, optional (default=1e-4)
        Relative tolerance used in the stopping criterion:
            |e(t - T) - e(t)| <= tol * e(t)
    T : int, optional (default=10)
        Lag used in the stopping criterion: compare the current error with the
        error T iterations before.
    column_stochastic: bool, if True the input model is normalized s.t. 1^T V = 1

    init: str, (default = "random")
            Initialization method for factor matrices W and H. Supported values:
            - "random": Random initialization with uniform distribution.
            - "nndsvd": Non-negative Double Singular Value Decomposition initialization.
            - "custom": User will set W and H manually.
    W0: Optional[np.ndarray], (default = None)
         Initial matrix for W if init = "custom"
    H0: Optional[np.ndarray], (default = None)
         Initial matrix for H if init = "custom"
    Attributes
    ----------
    V : NonNegMatrix
        Input matrix.
    rank : int
        Factorization rank.
    max_iter : int
        Maximum allowed iterations.
    tol : float
        Tolerance for convergence test.
    T : int
        Lag parameter for convergence test.
    m : int
        Number of rows of V.
    n : int
        Number of columns of V.
    W : NonNegMatrix
        Left factor matrix of shape (m, rank).
    H : NonNegMatrix
        Right factor matrix of shape (rank, n).
    errors : list[float]
        History of relative reconstruction errors:
            e(t) = ||V - WH|| / ||V||
    V_norm : float
        Frobenius norm of the input matrix V used to normalize errors.
    Public Methods
    --------------
    fit(solver="HALS")
        Run factorization using the specified solver. Supported solvers:
        - "MU": multiplicative updates
        - "HALS": hierarchical alternating least squares
        - "ALS": alternating least squares
        - "beta_MU": beta-divergence multiplicative updates (requires beta parameter)
        Raises ValueError if an unsupported solver string is provided.
    plot_errors()
        Plot the stored relative error history on a logarithmic y-scale.
    reconstruct() -> NonNegMatrix
        Return the current reconstruction W @ H as a NonNegMatrix.
    get_final_error() -> float
        Return the most recent (final) relative reconstruction error.
    get_factors() -> tuple[NonNegMatrix, NonNegMatrix]
        Return the factor matrices (W, H).
    Implementation details
    ----------------------
    Multiplicative Updates (MU)
        The MU implementation follows the classical Lee & Seung multiplicative update
        rules.
    Hierarchical Alternating Least Squares (HALS)
        HALS updates each column/row of W and H in turn using closed-form updates
        that enforce non-negativity by taking a maximum with zero.
    Alternating least squares (ALS)
        ALS solves for W and H alternately using least squares and then projecting the solutions.
    Projected Gradient Descent (PGD)
        PGD updates W and H by performing a gradient descent step followed by a
        projection onto the non-negative orthant.
    Beta-divergence Multiplicative Updates (beta_MU)
        The beta_MU implementation follows the update rules for minimizing the
        beta-loss between V and WH.
        see: arXiv:1010.1763 for the details: # * https://arxiv.org/abs/1010.1763 * #
    Stopping criterion
    ------------------
    After each iteration the relative error e(t) is appended to self.errors. The
    algorithm stops early if
        |e(t - T) - e(t)| <= tol * e(t)
    for t >= T. Otherwise the process continues up to max_iter iterations.
    -----------------
    Example
    -----------------
    Basic usage:
        model = NMF(V, rank=10, max_iter=500, tol=1e-5, T=5)
        model.fit(solver="HALS")
        V_approx = model.reconstruct()
    """

    def __init__(self, V: np.ndarray, rank: int, max_iter: int = 1000, tol: float = 1e-4, T: int = 10, column_stochastic : bool = False, init: str = "random", W0: Optional[np.ndarray] = None, H0: Optional[np.ndarray] = None) -> None:
        """
        Initialize the NMF model with the input matrix and parameters.
        Parameters
        ----------
        V : NonNegMatrix
            Non-negative data matrix to factorize with shape (m, n).
        rank : int
            Target factorization rank r.
        max_iter : int, optional (default=1000)
            Maximum number of iterations for the chosen update algorithm.
        tol : float, optional (default=1e-4)
            Relative tolerance used in the stopping criterion:
                |e(t - T) - e(t)| <= tol * e(t)
        T : int, optional (default=10)
            Lag used in the stopping criterion: compare the current error with the
            error T iterations before.  
        column_stochastic: bool, (default = True)
                           Normalize the input data matrix X s.t. 1^T V = 1 i.e. each column sum to 1. This property is called column stochasticicty 
        init: str, (default = "random")
              Initialization method for factor matrices W and H. Supported values:
              - "random": Random initialization with uniform distribution.
              - "nndsvd": Non-negative Double Singular Value Decomposition initialization.
              - "custom": User will set W and H manually.
        """
        try:
            self.V = NonNegMatrix(V / (V.sum(axis=0, keepdims=True) + 1e-10)) if column_stochastic else NonNegMatrix(V)
        except ValueError as e:
            raise ValueError("Input matrix V must be non-negative.") from e
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol
        self.T = T
        self.m, self.n = V.shape
        self.errors = []
        self.V_norm = np.linalg.norm(V, "fro")
        match init:
            case "random":
                self.W = NonNegMatrix(np.random.rand(self.m, self.rank))
                self.H = NonNegMatrix(np.random.rand(self.rank, self.n))
            case "nndsvd":
                self.__NNSVD_init()
            case "custom":
                if W0 is None or H0 is None:
                    raise ValueError("Please provide both W0 and H0 for custom initialization.")
                if W0.shape != (self.m, self.rank):
                    raise ValueError(f"W0 must have shape ({self.m}, {self.rank}).")
                if H0.shape != (self.rank, self.n):
                    raise ValueError(f"H0 must have shape ({self.rank}, {self.n}).")
                self.W = NonNegMatrix(W0)
                self.H = NonNegMatrix(H0)
            case _:
                raise ValueError(f"Initialization method {init} not recognized.")
    
    def fit(self, solver: str, beta: Optional[float] = None):
        """
        Fit the NMF model using the selected update solver.
        Parameters
        ----------
        solver : str, optional
            The name of the update algorithm to use. Supported values:
            - "HALS" : Hierarchical Alternating Least Squares 
            - "ALS" : Alternating Least Squares
            - "MU"   : Multiplicative Updates
            - "beta_MU" : beta-divergence multiplicative updates (requires beta parameter)
            - "PGD" : Projected Gradient Descent
        beta: float, optional 
            required if the solver is beta_MU
        Returns
        -------
        None
            The method updates in-place factor matrices.
        Raises
        ------
        ValueError
            If `solver` is not one of the supported solver names.
        --------------      
        Usage example:
            model.fit(solver = "MU")
        """
        print(f"Fitting with {solver} algorithm")
        start_time = time.perf_counter()
        match solver:
            case "MU":
                self.__mu_update()
            case "HALS":
                self.__HALS_update()
            case "ALS":
                self.__ALS_update()
            case "PGD":
                self.__PGD_update()
            case "beta_MU":
                if beta is None:
                    raise ValueError("provide a value for beta")
                if beta < 0:
                    raise ValueError("beta must be non-negative")
                if beta == 2:
                    self.__mu_update()
                else:
                    self.__beta_update(beta)
            case _:
                raise ValueError(f"Solver {solver} not found")
        end_time = time.perf_counter()
        self.fit_time = end_time - start_time
        self.n_iter = max(0, len(self.errors) - 1)
        self.time_per_iter = self.fit_time/self.n_iter if self.n_iter > 0 else float('inf')
        print(f"Fit completed in {self.fit_time:.4f} s, iterations: {self.n_iter}, avg time/iter: {self.time_per_iter:.4e} s")
        if self.n_iter == self.max_iter:
            print("Max iter reached: you may try to increase the value")

    def plot_errors(self):  
        """
        Plot the normalized reconstruction error history on a logarithmic y-scale. 
        Return:
            None 
        --------------      
        Usage example:
                model.plot_errors()
        """
        # plot (y axis in log scale)
        plt.figure(figsize=(6,4))
        plt.plot(range(len(self.errors)), self.errors, '-o', markersize=3)
        plt.yscale('log')
        plt.xlabel("Iteration")
        plt.ylabel("Relative error (log scale)")
        plt.title("NMF reconstruction error")
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.show()

    
    def reconstruct(self) -> NonNegMatrix:
        """
        This method reconstructs the matrix V from W and H.
        """
        return NonNegMatrix(self.W @ self.H) 
        
    
    def __mu_update(self) -> None:
        """
        This method performs the multiplicative update algorithm for NMF.
        stopping criterion:
            |e(t - T) - e(t)| ≤ tol*e(t)
        Formula:
            W = W * (V H^T) / (W H H^T)  # * is the element-wise product between matrices
            H = H * (W^T V) / (W^T W H)
        """

        self.__compute_Fro_error() # e(0)

        for t in range(self.max_iter):
            # first block
            W_num = self.V @ self.H.T
            W_den = self.W @ (self.H @ self.H.T)
            self.W = NonNegMatrix(np.multiply(self.W, W_num / (W_den + 1e-10))) # update W
            # second block
            H_num = self.W.T @ self.V
            H_den = (self.W.T @ self.W) @ self.H
            self.H = NonNegMatrix(np.multiply(self.H, H_num / (H_den + 1e-10))) # update H
            # error at step t
            self.__compute_Fro_error()
            if t >= self.T and np.abs(self.errors[t - self.T] - self.errors[t]) <= self.tol*self.errors[t]:
                break

    def __beta_update(self, beta: float) -> None:
        """
        stopping criterion:
            |e(t - T) - e(t)| ≤ tol*e(t)
        Multiplicative updates for beta-divergence:
            H = H * (W^T[(WH)^(beta - 2) * V]) / (W^T[(WH)^(beta -1)])
            W = W * ([(WH)^(beta - 2) * V] H^T) / ([(WH)^(beta - 1)] H^T)
        """
        print(f"value of beta: {beta}")
        self.V_beta_div = beta_loss(self.V, np.mean(self.V)*np.ones(self.V.shape), beta)
        self.errors.append(beta_loss(self.V, self.W @ self.H, beta)/ (self.V_beta_div + 1e-10))
        eps = 1e-10
        for t in range(self.max_iter):
            WH = np.maximum(self.W @ self.H, 0)
            # element-wise powers
            WH_beta_m2 = np.power(WH, beta - 2)
            WH_beta_m1 = np.power(WH, beta - 1)
            # block 1
            W_num = (np.multiply(WH_beta_m2, self.V)) @ self.H.T
            W_den = (WH_beta_m1) @ self.H.T
            W_new = np.multiply(self.W, W_num / W_den + 1e-10)
            W_new = np.maximum(W_new, eps) 
            self.W = NonNegMatrix(W_new) # update W

            # recompute WH after W update
            WH = np.maximum(self.W @ self.H, 0)
            WH_beta_m2 = np.power(WH, beta - 2)
            WH_beta_m1 = np.power(WH, beta - 1)

            # block 2
            H_num = self.W.T @ (WH_beta_m2 * self.V)
            H_den = self.W.T @ (WH_beta_m1)
            H_new = np.multiply(self.H, H_num / H_den + 1e-10)
            H_new = np.maximum(H_new, eps)
            self.H = NonNegMatrix(H_new) # update H

            # error at step t (beta-divergence)
            rel_err = beta_loss(self.V, self.W @ self.H, beta)/(self.V_beta_div + 1e-10)
            self.errors.append(rel_err)

            if t >= self.T and np.abs(self.errors[t - self.T] - self.errors[t]) <= self.tol * self.errors[t]:
                break


    
    def __ALS_update(self) -> None:
        """
        Input: V, W(0), H(0)
        V.size =  (m x n), 
        W(0).size = (m x r) 
        H(0).size = (r x n)
        Output: An NMF solution (W, H) ≥ 0 such that ∥V - WH∥_F is minimized.
        for i = 1, 2, . . . do
            W <- max (0, argmin_{W} ||V - WH||_F^2)
            H <- max (0, argmin_{H} ||V - WH||_F^2)
        end for
        """
        # initial error
        self.__compute_Fro_error()  # e(0)

        for t in range(self.max_iter):
            # Solve for W: min_{W >= 0} ||V - WH||_F^2
            HHT = self.H @ self.H.T
            VHT = self.V @ self.H.T
            for i in range(self.m):
                self.W[i, :] = np.maximum(0, np.linalg.solve(HHT + 1e-10 * np.eye(self.rank), VHT[i, :].T).T)

            # Solve for H: min_{H >= 0} ||V - WH||_F^2
            WTW = self.W.T @ self.W
            WTV = self.W.T @ self.V
            for j in range(self.n):
                self.H[:, j] = np.maximum(0, np.linalg.solve(WTW + 1e-10 * np.eye(self.rank), WTV[:, j]))

            # error at step t
            self.__compute_Fro_error()
            if t >= self.T and np.abs(self.errors[t - self.T] - self.errors[t]) <= self.tol * self.errors[t]:
                break

    def __PGD_update(self) -> None:
        """
        Input: V, W(0), H(0)
        V.size =  (m x n), 
        W(0).size = (m x r) 
        H(0).size = (r x n)
        Output: An NMF solution (W, H) ≥ 0 such that ∥V - WH∥_F is minimized.
        for i = 1, 2, . . . do
            LW = ∥H∥_2^2
            W <- max (0, W - 1/LW * ((W H - V) H^T))
            LH = ∥W∥_2^2
            H <- max (0, H - 1/LH * (W^T(WH - V))) 
        end for
        """
        # initial error
        self.__compute_Fro_error()  # e(0)

        for t in range(self.max_iter):
            LW = (np.linalg.norm(self.H, 2) ** 2) + 1e-10
            LH = (np.linalg.norm(self.W, 2) ** 2) + 1e-10
            # Gradient w.r.t W: ((W H - V) H^T)
            grad_W = (self.W @ self.H - self.V) @ self.H.T
            self.W = NonNegMatrix(np.maximum(0, self.W - grad_W / LW))
            # Gradient w.r.t H: (W^T (W H - V))
            grad_H  = self.W.T @ (self.W @ self.H - self.V)
            self.H = NonNegMatrix(np.maximum(0, self.H - grad_H / LH))

            # error at step t
            self.__compute_Fro_error()
            if t >= self.T and np.abs(self.errors[t - self.T] - self.errors[t]) <= self.tol * self.errors[t]:
                break
            



    def __HALS_update(self) -> None:
        """
        formula W[:,k] <- max(0, VH^T[:,k] - sum_{l \not= k}W[:,l](HH^T)[l,k])
                H[j,:] <- max(0, W^TV[j,:] - sum_{l \not= j}W^TW[k,l](H)[l,:])
        """
        self.__compute_Fro_error()
        for t in range(self.max_iter):
            # compute VH^T and HH^T
            VHT = self.V @ self.H.T
            HHT = self.H @ self.H.T
            # block 1
            for k in range(self.rank):
                sum = 0
                for l in range(self.rank):
                    if l != k:
                        sum += self.W[:,l] * HHT[l,k]
                self.W[:,k] = np.maximum(0, (VHT[:,k] - sum)/(HHT[k,k] + 1e-10)) # update column of W
            # compute W^TV and W^TW
            WTV = self.W.T @ self.V
            WTW = self.W.T @ self.W
            # block 2
            for j in range(self.rank):
                sum = 0
                for l in range(self.rank):
                    if l != j:
                        sum += WTW[k,l] * self.H[l,:]
                self.H[k,:] = np.maximum(0, (WTV[k,:] - sum)/(WTW[k,k] + 1e-10)) # update row of H
            # error at step t
            self.__compute_Fro_error()
            if t >= self.T and np.abs(self.errors[t - self.T] - self.errors[t]) <= self.tol*self.errors[t]:
                break            



    def __compute_Fro_error(self) -> None:
        """
        Private method.
        Compute the current relative error and append it to self.errors.

        The relative error is the Frobenius norm of the residual normalized by the
        Frobenius norm of the input matrix V:
            e(t) = ||V - WH|| / ||V||.
        """
        rel_err = np.linalg.norm(self.V - self.W @ self.H, "fro")/ (self.V_norm +1e-10)
        self.errors.append(rel_err)


    def __NNSVD_init(self) -> None:
        """
        Non-negative Double Singular Value Decomposition (NNDSVD) initialization.
        Reference:
        Boutsidis, C., & Gallopoulos, E. (2008). SVD-based initialization: A head start for
        nonnegative matrix factorization. Pattern Recognition, 41(4), 1350-1362.
        --------------
        """
        svd = TruncatedSVD(n_components=self.rank)
        U = svd.fit_transform(self.V) 
        S = svd.singular_values_ 
        VT = svd.components_
        self.W = np.zeros((self.m, self.rank)) # dimensions m x r
        self.H = np.zeros((self.rank, self.n)) # dimensions r x n
        self.W[:, 0] = np.sqrt(S[0]) * np.maximum(0, U[:, 0])
        self.H[0, :] = np.sqrt(S[0]) * np.maximum(0, VT[0, :])
        for j in range(1, self.rank):
            u = U[:,j]
            v = VT[j,:]
            u_pos = np.maximum(0, u)
            u_neg = np.maximum(0, -u)
            v_pos = np.maximum(0, v)
            v_neg = np.maximum(0, -v)
            u_pos_norm = np.linalg.norm(u_pos)
            u_neg_norm = np.linalg.norm(u_neg)
            v_pos_norm = np.linalg.norm(v_pos)
            v_neg_norm = np.linalg.norm(v_neg)
            m_pos = u_pos_norm * v_pos_norm
            m_neg = u_neg_norm * v_neg_norm
            if m_pos >= m_neg:
                self.W[:, j] = np.sqrt(S[j] * m_pos) * (u_pos / (u_pos_norm + 1e-10))
                self.H[j, :] = np.sqrt(S[j] * m_pos) * (v_pos / (v_pos_norm + 1e-10))
            else:
                self.W[:, j] = np.sqrt(S[j] * m_neg) * (u_neg / (u_neg_norm + 1e-10))
                self.H[j, :] = np.sqrt(S[j] * m_neg) * (v_neg / (v_neg_norm + 1e-10))
        print("NNDSVD initialization completed.")


    def get_final_error(self) -> float:
        """
        This method returns the final error of the reconstruction.
        """
        return self.errors[-1]
          

    def get_factors(self) -> dict:
        """
        This method returns the factor matrices W and H.
        """
        return {"W" : self.W, "H": self.H}
