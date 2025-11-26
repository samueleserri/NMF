import pandas as pd

from nmf import NonNegMatrix, NMF
from utils import display, measure_sparsity

"""
This file reproduces the results obtained by Daniel D. Lee & H. Sebastian Seung in their paper: https://www.nature.com/articles/44565 .
The dataset CBCL consists in a 2429 x 361 pixels where each column represent a 19 x 19 grey scale image of a face. In total there are 361 pictures.  
The NMF model with MU algorithm is applied with a factorization rank r = 49. After the fitting two matrices W of size (2429 x 49) and H of size (49 x 361) are obtained.
Due to the non-negative constraints of the NMF model the matrix W can still be interpreted in the same way as the original matrix, each column of the W can be reshaped
to a 19 x 19 matrix and displayed. The main result of Lee & Seung is that NMF is able to learn a representation of faces by "parts".
TODO: reconstruction and sparsity measures.
"""

def load_dataset() -> NonNegMatrix:
    return  NonNegMatrix(pd.read_csv("data/CBCL.csv", header=None).to_numpy())


def fit_model(rank:int, show: bool = False, solver: str = "MU") -> NMF:
    V = load_dataset()
    sparsity = measure_sparsity(V)
    print(f"sparsity of the data matrix: {sparsity}")
    model = NMF(V, rank)
    model.fit(solver)
    if show: # show original images if set to True
        display(V[:, :rank], perrow=7, Li=19, Co=19, bw=0, show=True)
    return model

    

def run_example() -> None:
    reconstruction_rank = 49
    fitted_model = fit_model(reconstruction_rank)
    display(fitted_model.W[:,:reconstruction_rank], perrow=7,Li=19, Co=19, bw=0, show=True)
    print(f"reconstruction error: {fitted_model.get_final_error()}")
    print(f"sparsity of W:{measure_sparsity(fitted_model.W)}")



if __name__ == "__main__":
    run_example()