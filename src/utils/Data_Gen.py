import numpy as np
from nmf.NonNegMatrix import NonNegMatrix

class Data_Gen:
    def __init__(self, m: int  , n: int , r: int, random_state = None) -> None:
        """
        param 
        size: tuple (m : int, n: int) representing the dimensions of the generated data matrix.
        distr: distribution type for generating data, default is 'uniform'.
        random_state: int a seed for reproducing the results.
        """
        self.m = m
        self.n = n
        self.r = r
        if random_state is not None:
            # set random seed  for reproducibility
            self.seed = np.random.seed(random_state)
            self.rng = np.random.default_rng(self.seed)

    def generate_data(self, distr: str = "normal") -> NonNegMatrix:
        """
        Docstring for generate_data
        
        :param self: Description
        :return: Description
        :rtype: NonNegMatrix
        """
        self.data = None
        match distr:
            case 'uniform':
                data = np.random.uniform(low=0.0, high=1.0, size=(self.m, self.n)) # @ np.random.uniform(low=0.0, high=1.0, size=(self.r, self.n))
            case 'normal':
                data = np.abs(np.random.normal(loc=0.0, scale=1.0, size=(self.m, self.n))) #  @ np.abs(np.random.normal(loc=0.0, scale=1.0, size=(self.r, self.n)))
            case 'poisson':
                data = np.random.poisson(lam=5.0, size=(self.m, self.n)) # @ np.random.poisson(lam=5.0, size=(self.r, self.n))
            case _:
                raise ValueError(f"Unsupported distribution type: {distr}")
        # data = data/np.max(data)  # normalize data to [0, 1]
        self.data = NonNegMatrix(data)
        return self.data
    

        

    
    