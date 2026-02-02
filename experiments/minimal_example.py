import numpy as np
from nmf import NMF

V = np.random.rand(100, 20)@ np.random.rand(20, 100)
model = NMF(V, rank=20, max_iter=10000)
model.fit(solver="beta_MU", beta=1)
print(f"Final error: {model.get_final_error()}")
model.plot_errors()