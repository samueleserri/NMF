import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nmf import NonNegMatrix, NMF
from utils import display, measure_sparsity

"""
This file reproduces the results obtained by Daniel D. Lee & H. Sebastian Seung in their paper: https://www.nature.com/articles/44565 .
The dataset CBCL consists in a 2429 x 361 pixels where each column represent a 19 x 19 grey scale image of a face. In total there are 361 pictures.  
The NMF model with MU algorithm is applied with a factorization rank r = 49. After the fitting two matrices W of size (2429 x 49) and H of size (49 x 361) are obtained.
Due to the non-negative constraints of the NMF model the matrix W can still be interpreted in the same way as the original matrix, each column of the W can be reshaped
to a 19 x 19 matrix and displayed. The main result of Lee & Seung is that NMF is able to learn a representation of faces by "parts".
TODO: weights visualization.
"""

def load_dataset() -> NonNegMatrix:
    """
    Load the CBCL face dataset from csv file and return it as NonNegMatrix
    return: NonNegMatrix of shape (2429, 361)
    2429 = 19 x 19 pixels
    361 = number of images
    -------------------
    dataset path: data/CBCL.csv
    -------------------
    """
    return  NonNegMatrix(pd.read_csv("data/CBCL.csv", header=None).to_numpy())


def fit_model(rank:int, show: bool = False, solver: str = "beta_MU", beta: float = 1) -> NMF:
    """
    param rank: factorization rank
    param show: shows the original images before fitting the model if true
    param solver: NMF solver to use (in the original paper MU is used)
    return: fitted NMF model
    """ 
    V = load_dataset()    
    if show: # original images displayed if True
        display(V[:, :rank], perrow=7, Li=19, Co=19, bw=0, show=True)
    sparsity = measure_sparsity(V)
    print(f"sparsity of the data matrix: {sparsity}")
    # instantiate and fit model
    model = NMF(V, rank)
    model.fit(solver, beta)
    return model

    
def reconstruct_face(idx:int ,model: NMF) -> NonNegMatrix:
    """
    param idx: index of the face to reconstruct (0 <= idx < n)
    param model: fitted NMF model
    return: reconstructed face as NonNegMatrix
    -------------------
    V.shape = m x n
    W.shape = m x r
    H.shape = r x n
    # in this case n is the number of faces and m is the dimension of the flattened image (19 x 19 for this dataset)
    ------------------
    if V \approx WH then the j-th column of V can be written as V(:,j) = WH(:,j)
    this is a linear combination of the column of W weighted with the values of the matrix H
    i.e. V[:,j] = H[1,j]W[:,1] + H[2,j]W[:,2]+...+H[r,j]W[:,r]
    furthermore note that the number of columns of H is precisely the number of images in V so there is a one-to-one mapping between them. 
    """
    reconstructed = model.W @ model.H[:,idx]
    V = load_dataset()
    print(f"Error in reconstructing face {idx}: {np.linalg.norm(V[:,idx]-reconstructed)/np.linalg.norm(V[:,idx])}")
    return NonNegMatrix(reconstructed)



def run_example(show: bool = False) -> None:
    reconstruction_rank = 49
    fitted_model = fit_model(reconstruction_rank)
    if show:
        display(fitted_model.W[:,:reconstruction_rank], perrow=7,Li=19, Co=19, bw=0, show=True)
    print(f"reconstruction error: {fitted_model.get_final_error()}")
    print(f"sparsity of W:{measure_sparsity(fitted_model.W)}")
    # heat map of the weights in H
    # plt.imshow(fitted_model.H[:,0], aspect='auto', cmap='grey')
    # plt.colorbar()
    # plt.title("Heatmap of the weights in H")
    # plt.xlabel("Images")
    # plt.ylabel("Components")
    # plt.show()
    
    # reconstruct_face(2, fitted_model)



if __name__ == "__main__":
    run_example(True)