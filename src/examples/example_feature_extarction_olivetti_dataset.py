from sklearn.datasets import fetch_olivetti_faces

from nmf import NMF, NonNegMatrix
from utils import display

def load_dataset():
    V, _  = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=42)
    return NonNegMatrix(V.T)
 
    

def fit_model(n_components, show=False):
    faces = load_dataset() # Shape: (4096, 400)
    if show:
        display(faces, perrow=40, height=64, width=64, column_order=False)
    
    model = NMF(faces, rank=n_components)
    model.fit("beta_MU", 2) 

    basis_images = model.W
    feature_weights = model.H
 
    print(f"Basis shape (W): {model.W.shape}")
    print(f"Features shape (H): {feature_weights.shape}")   
    display(basis_images, perrow=n_components//2, height=64, width=64, column_order=False)


    return model


def run_example():
    n_components = 30
    show = True
    model = fit_model(n_components=n_components, show=show)    
    if show:
        model.plot_errors()
    print(f"reconstruction error {model.get_final_error()}")

if __name__ == "__main__":
    run_example()