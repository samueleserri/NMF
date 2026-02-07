import time
from nmf import NMF
from typing import Optional
import numpy as np

from utils.beta_divergence import beta_loss

class RegularizedNMF(NMF):

    """
    This class extends the NMF class to include a regularization term in the objective function. The regularization term is added to the error calculation and can be controlled by two regularization parameters alpha_W and alpha_H.
    The regularization term is defined as:
    R(W,H) = alpha_W * f_W(W) + alpha_H * f_H(H)
    where f_W and f_H are the regularization functions for W and H respectively.
    """

    def __init__(self, V: np.ndarray, rank: int, max_iter: int = 1000, tol: float = 1e-4, T: int = 10, column_stochastic : bool = False, init: str = "random", W0: Optional[np.ndarray] = None, H0: Optional[np.ndarray] = None, alpha_W: float = 0.0, alpha_H: float = 0.0, regularizer: Optional[str] = "ell_1", custom_regularizer: Optional[callable] = None) -> None: # type: ignore
        super().__init__(V, rank, max_iter, tol, T, column_stochastic, init, W0, H0)
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.regularizer = regularizer
        self.custom_regularizer = custom_regularizer
    
    def _compute_Fro_error(self) -> None:
        rel_err = beta_loss(self.V, self.W @ self.H, 2)/(self.V_norm + 1e-10)
        reg_err = self.__compute_regularization_error()
        self.errors.append(rel_err + reg_err)

    
    def _compute_Beta_error(self, beta: float) -> None:
        rel_err = beta_loss(self.V, self.W @ self.H, beta)/(self.V_beta_div + 1e-10)
        reg_err = self.__compute_regularization_error()
        self.errors.append(rel_err + reg_err)

    def __compute_regularization_error(self) -> float:
        """
        Compute the regularization error based on the selected regularizer and the current values of W and H. The regularization error is added to the Frobenius norm error to compute the total error at each iteration.
        """
        match self.regularizer:
            case "ell_1":
                reg_W = self.alpha_W * np.sum(np.abs(self.W))
                reg_H = self.alpha_H * np.sum(np.abs(self.H))
            case "ell_2":
                reg_W = self.alpha_W * np.sum(self.W**2)
                reg_H = self.alpha_H * np.sum(self.H**2)
            case "custom":
                if self.custom_regularizer is None:
                    raise ValueError("Custom regularizer function not provided")
                reg_W = self.alpha_W * self.custom_regularizer(self.W)
                reg_H = self.alpha_H * self.custom_regularizer(self.H)
            case _:
                raise ValueError("Invalid regularizer. Supported values are 'ell_1', 'ell_2', and 'custom'.")
        return reg_W + reg_H


    
