from sklearn.decomposition._nmf import _beta_divergence as skl_beta
from utils.beta_divergence import beta_divergence as my_beta
import numpy as np


def test(beta):
    for _ in range(10**3):
        x, w, h = np.random.uniform(low=0, high=10), np.random.uniform(low=0, high=10), np.random.uniform(low=0, high=10)
        x1 = my_beta(x,w*h, beta)
        x2 = skl_beta(x,w,h,beta)
        # print(x1,x2)
        assert np.abs(x1 - x2) <= 1e-10
if __name__ == "__main__":
    test(3)