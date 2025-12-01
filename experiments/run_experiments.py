import matplotlib.pylab as plt
from nmf import NMF, NonNegMatrix
from utils import Data_Gen, measure_sparsity

# TODO: comments

def run_experiments(params: dict, solver: str , distr : str = "normal", beta: float = 1) -> None:
    
    data_generator = Data_Gen(params)
    V = data_generator.generate_data(distr)
    # histogram of the data matrix
    plt.hist(V, bins=n); plt.title("Histogram of values in V"); plt.xlabel("Value");plt.ylabel("Frequency"); plt.show()
    print(f"V.shape = {V.shape}, V sparsity: {measure_sparsity(V)}")
    model = NMF(V, r, 10**4, column_stochastic=False) # * model instantiation *
    model.fit(solver, beta)
    model.plot_errors()
    print(f"final error: {model.get_final_error()}")
    print(f"W sparsity: {measure_sparsity(model.W)}")
    print(f"H sparsity: {measure_sparsity(model.H)}")
    # print(model.W.sum())
        # run for different values of r and plot error vs rank
    ranks = [i for i in range(10, min(m,n)+10, 10)]
    errors = []
    for rank in ranks:
        print(f"Running experiment for rank = {rank}")
        model = NMF(V, rank, 10**4, column_stochastic=False)
        model.fit(solver, beta)
        final_error = model.get_final_error()
        errors.append(final_error)
        print(f"final error for rank {rank}: {final_error}")
    plt.plot(ranks, errors, marker='o') 
    plt.xlabel("Rank")
    plt.ylabel("Final Error")
    plt.title("Final Error vs Rank")
    plt.show()

if __name__ == "__main__":
    m : int = 100
    n : int = 50
    r : int = min(m,n)
    distribution : str = "uniform"
    solver : str = "beta_MU"
    beta: float = 1
    run_experiments({"m": m, "n": n, "r": min(m,n), "random_state" : 42}, solver = solver, distr=distribution, beta = beta)