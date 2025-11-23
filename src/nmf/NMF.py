import numpy as np
import time
import matplotlib.pyplot as plt
from nmf.NonNegMatrix import NonNegMatrix


class NMF:
    """"
    Non-negative Matrix Factorization (NMF).
    Description
    -----------
    This class implements a basic NMF solver with two update strategies:
    - Multiplicative Updates (MU)
    - Hierarchical Alternating Least Squares (HALS)
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
            e(t) = ||V - W @ H||_F / ||V||_F
    V_norm : float
        Frobenius norm of the input matrix V used to normalize errors.
    Public Methods
    --------------
    fit(solver="HALS")
        Run factorization using the specified solver. Supported solvers:
        - "MU": multiplicative updates
        - "HALS": hierarchical alternating least squares
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
        The implementation computes intermediate quantities (e.g. V @ H.T, H @ H.T,
    Stopping criterion
    ------------------
    After each iteration the relative error e(t) is appended to self.errors. The
    algorithm stops early if
        |e(t - T) - e(t)| <= tol * e(t)
    for t >= T. Otherwise the process continues up to max_iter iterations.
    
    Notes
    -----------------
    In the constructor the matrices W,H are initialized as the identity matrix W = I^{m x r}, H = I^{r x n},
    This initialization does not work for the MU solver so in the fit method if "MU" is solved W,H are set to be random (np.random.rand) samples from a Uniform([0,1])
    I have noticed that the random initialization work poorly with HALS when fitting a grey scale image.
    -----------------
    Example
    -----------------
    Basic usage:
        model = NMF(V, rank=10, max_iter=500, tol=1e-5, T=5)
        model.fit(solver="HALS")
        W, H = model.get_factors()
        V_approx = model.reconstruct()
    """

    def __init__(self, V: NonNegMatrix, rank: int, max_iter: int = 1000, tol: float = 1e-4, T: int = 10):
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
        """
        self.V = V
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol
        self.T = T
        self.m, self.n = V.shape
        # Initialize W and H as Identity matrix
        self.W = NonNegMatrix(np.eye(self.m, self.rank))
        self.H = NonNegMatrix(np.eye(self.rank, self.n))
        self.errors = []
        self.V_norm = np.linalg.norm(V, "fro")
    
    def fit(self, solver: str = "HALS"):
        """
        Fit the NMF model using the selected update solver.
        Parameters
        ----------
        solver : str, optional
            The name of the update algorithm to use. Supported values:
            - "HALS" : Hierarchical Alternating Least Squares 
            - "MU"   : Multiplicative Updates
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
                self.W = np.random.rand(self.m, self.rank) # set W and H to random matrices if using MU
                self.H = np.random.rand(self.rank, self.n)
                self.__mu_update__()
            case "HALS":
                self.__HALS_update__()
            case _:
                raise ValueError("Solver not found")
        end_time = time.perf_counter()
        self.fit_time = end_time - start_time
        self.n_iter = max(0, len(self.errors) - 1)
        self.time_per_iter = self.fit_time/self.n_iter if self.n_iter > 0 else float('inf')
        print(f"Fit completed in {self.fit_time:.4f} s, iterations: {self.n_iter}, avg time/iter: {self.time_per_iter:.4e} s")
            

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
        plt.ylabel("Frobenius norm ||V - W H||_F/||V||_F (log scale)")
        plt.title("NMF reconstruction error")
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.show()

    
    def reconstruct(self) -> NonNegMatrix:
        """
        This method reconstructs the matrix V from W and H.
        """
        return NonNegMatrix(self.W @ self.H) 
        
    
    def __mu_update__(self) -> None:
        """
        This method performs the multiplicative update algorithm for NMF.
        stopping criterion:
            |e(t - T) - e(t)| ≤ tol*e(t)
        Formula:
            W = W * (V H^T) / (W H H^T)  # * is the element-wise product between matrices
            H = H * (W^T V) / (W^T W H)
        """

        self.__compute_error__() # e(0)

        for t in range(self.max_iter):
            # first block
            W_num = self.V @ self.H.T
            W_den = self.W @ (self.H @ self.H.T)
            self.W = np.multiply(self.W, W_num / (W_den + 1e-10)) # update W
            # second block
            H_num = self.W.T @ self.V
            H_den = (self.W.T @ self.W) @ self.H
            self.H = np.multiply(self.H, H_num / (H_den + 1e-10)) # update H
            # error at step t
            self.__compute_error__()
            if t >= self.T and np.abs(self.errors[t - self.T] - self.errors[t]) <= self.tol*self.errors[t]:
                break


    def __HALS_update__(self) -> None:
        """
        formula W[:,k] <- max(0, VH^T[:,k] - sum_{l \not= k}W[:,l](HH^T)[l,k])
                H[j,:] <- max(0, W^TV[j,:] - sum_{l \not= j}W^TW[k,l](H)[l,:])
        """
        self.__compute_error__()
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
            self.__compute_error__()
            if t >= self.T and np.abs(self.errors[t - self.T] - self.errors[t]) <= self.tol*self.errors[t]:
                break            



    def __compute_error__(self) -> None:
        """
        Private method.
        Compute the current relative error and append it to self.errors.

        The relative error is the Frobenius norm of the residual normalized by the
        Frobenius norm of the input matrix V:
            e(t) = ||V - WH|| / ||V||.
        """
        rel_err = np.linalg.norm(self.V - self.W @ self.H, "fro")/self.V_norm
        # rel_err = (self.V_norm -2*np.dot(self.W @ self.V, self.H) + np.dot(self.W.T @ self.W, self.H @ self.H.T))/(self.V_norm)
        self.errors.append(rel_err)
    

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
