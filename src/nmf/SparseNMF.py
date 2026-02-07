import numpy as np
from typing import Optional
from scipy import sparse
from nmf import NMF

class SparseNMF(NMF):

    def __init__(self, V: sparse.csr_array, r: int, init: Optional[str] = "random",
                 max_iter: int = 1000, tol: float = 1e-4, T: int  = 10, 
                beta: float = 2, W0: Optional[np.ndarray] = None, H0: Optional[np.ndarray] = None,
                ):
        """
        See NMF class for documentation.
        This class extends NMF to handle sparse input matrices efficiently; it is compatible with the scipy sparse array interface.
        It is recommended to use this class whenever the input matrix is sparse.
        For more details see: https://docs.scipy.org/doc/scipy/reference/sparse.html
        """
        self.V = V 
        if V.min() < 0:
            raise ValueError("Input matrix V contains negative elements.")
        self.rank = r
        self.m, self.n = V.shape # type: ignore
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
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
        self.errors = [] 
        self.V_norm = self.compute_frobenius_norm(V)
   
   
   
   
    def compute_frobenius_norm(self, M) -> float:
        """
        # * Compute the Frobenius norm of a (possibly sparse) matrix.
        In case of a sparse matrix we don't need to visit all the elements, we can use the attributes of the sparse matrix class provided by scipy and 
        compute the norm by using the definition
        
        """
        if sparse.issparse(M):
            return float(np.sqrt((M.data ** 2).sum()))
        else:
            return float(np.linalg.norm(M))
            

    
    