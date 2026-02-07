from nmf import RegularizedNMF
import numpy as np

"""
This file contains an example of how to use the RegularizedNMF class to perform non-negative matrix factorization with a custom regularization term. The custom regularizer used in this example is the elasticity regularizer, which is defined as R(X) = sum(|X|) / (sum(X^2) + epsilon). The regularization parameters alpha_W and alpha_H are set to 0.01 for both W and H.
"""

def eleasticity_regularizer(X: np.ndarray) -> float:
    """
    Custom regularizer function that computes the elasticity of the input matrix X. 
    formula: R(X) = sum(|X|) / (sum(X^2) + epsilon)
    """
    return np.sum(np.abs(X)) / (np.sum(X**2) + 1e-10)


V = np.random.rand(250, 100)

reg_model = RegularizedNMF(V, 80, max_iter=100, alpha_W=0.01, alpha_H=0.01, regularizer="custom", custom_regularizer=eleasticity_regularizer)
reg_model.fit(solver="beta_MU", beta = 2)
reg_model.plot_errors()