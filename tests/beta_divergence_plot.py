import numpy as np
import matplotlib.pyplot as plt

from utils.beta_divergence import beta_divergence


def plt_divergence():
    # Plotting and comparing the beta divergence for different values of beta
    x = np.linspace(0, 5, 100)
    y = np.zeros(x.shape)

    name_map = {0.0: 'Itakura-Saito', 1.0: 'Kullback-Leibler', 2.0: 'Frobenius norm'}
    beta_vals = [0.0, 1.0, 2.0, 3.0, 4.0, 10.0]

    # Plotting the graph
    for beta in beta_vals:
        for i, xi in enumerate(x):
            y[i] = beta_divergence(1, xi, beta)
        name = f'beta = {beta}: {name_map.get(beta, "beta="+str(beta))}'

        plt.plot(x, y, label=name)

    plt.xlabel("x")
    plt.ylabel("D(1, x)")
    plt.title("Beta-Divergence(1, x)")
    plt.legend(loc='upper center')
    plt.grid(True)
    plt.axis([0.0, 4.0, 0.0, 3.0]) #type: ignore

    # Displaying the graph
    plt.show()

if __name__ == "__main__":
  plt_divergence()