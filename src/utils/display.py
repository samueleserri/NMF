import math
import numpy as np
import matplotlib.pyplot as plt

def display(V, perrow, Li, Co, bw=0, show=True, column_order=True):
    """
    Display columns of V as Li x Co images arranged in a grid with `perrow` images per row.
    V : (m x r) numpy array where m == Li*Co
    perrow : number of images per row
    Li, Co : image height and width
    bw : if 0 (default) display high intensities as black (MATLAB behaviour),
         if 1 display high intensities as white
    show : if True, show the figure with matplotlib
    Returns the assembled display image Ass (float values in [0,1]).

    # credits: I have adapted the function found in N.Gillis repository of his book to display images in a CBCL-like dataset:
    #  https://gitlab.com/ngillis/nmfbook
    """
    V = np.maximum(V, 0.0)
    m, r = V.shape
    # Normalize columns to have maximum 1 (avoid division by zero)
    for i in range(r):
        mx = V[:, i].max()
        if mx > 0:
            V[:, i] = V[:, i] / mx

    # grid geometry
    n_rows = math.ceil(r / perrow)
    sep = 1  # separator thickness (1 pixel) between images
    grid_h = n_rows * Li + (n_rows - 1) * sep
    grid_w = perrow * Co + (perrow - 1) * sep

    # initialize with ones (background)
    Ass = np.ones((grid_h, grid_w), dtype=float)

    idx = 0
    for row in range(n_rows):
        for col in range(perrow):
            if idx >= r:
                break
            top = row * (Li + sep)
            left = col * (Co + sep)
            if column_order:
                patch = V[:, idx].reshape((Li, Co), order='F')
            else:
                patch = V[:, idx].reshape((Li, Co))
            Ass[top:top+Li, left:left+Co] = patch
            idx += 1
    if show:
        plt.figure(figsize=(8, 8))
        if bw == 1:
            plt.imshow(Ass, cmap='gray', vmin=0, vmax=1, aspect='equal')
        else:
            plt.imshow(1.0 - Ass, cmap='gray', vmin=0, vmax=1, aspect='equal')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return Ass