import numpy as np
from nmf import NonNegMatrix

class Data_Gen:
    def __init__(self, size : (tuple), distr: str ='uniform', random_state = None) -> None:
        """
        param 
        size: tuple (m : int, n: int) representing the dimensions of the generated data matrix.
        distr: distribution type for generating data, default is 'uniform'.
        random_state: int a seed for reproducing the results.
        """
        self.size = size
        self.distr = distr 
        if random_state is not None:
            # set random seed  for reproducibility
            np.random.seed(random_state)
     

    def generate_data(self) -> NonNegMatrix:
        """
        Docstring for generate_data
        
        :param self: Description
        :return: Description
        :rtype: NonNegMatrix
        """
        self.data = None
        match self.distr:
            case 'uniform':
                data = np.random.uniform(low=0.0, high=1.0, size=self.size)
            case 'normal':
                data = np.abs(np.random.normal(loc=0.0, scale=1.0, size=self.size))
            case 'poisson':
                data = np.random.poisson(lam=5.0, size=self.size)
            case _:
                raise ValueError(f"Unsupported distribution type: {self.distr}")
        
        self.data = NonNegMatrix(data)
        return self.data
    

        

    
    