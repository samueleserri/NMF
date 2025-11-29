import matplotlib.pylab as plt
from nmf import NMF, NonNegMatrix
from utils import Data_Gen, measure_sparsity

# TODO: comments

def run_experimemts(params: dict, solver: str , distr : str = "normal", beta: float = 1) -> None:
    
    data_generator = Data_Gen(params)
    V = data_generator.generate_data(distr)
    # histogram of the data matrix
    plt.hist(V, bins=15); plt.title("Histogram of values in V"); plt.xlabel("Value");plt.ylabel("Frequency"); plt.show()
    print(f"V.shape = {V.shape}, V sparsity: {measure_sparsity(V)}")
    model = NMF(V, r, 10**4) # * model instantiation *
    model.fit(solver, beta)
    model.plot_errors()
    print(f"final error: {model.get_final_error()}")
    print(f"W sparsity: {measure_sparsity(model.W)}")
    print(f"H sparsity: {measure_sparsity(model.H)}")


if __name__ == "__main__":
    m : int = 100
    n : int = 30
    r : int = min(m,n)
    distribution : str = "normal"
    solver : str = "beta_MU"
    beta: float = 0
    run_experimemts({"m": m, "n": n, "r": min(m,n), "random_state" : 42}, solver = solver, distr=distribution, beta = beta)

