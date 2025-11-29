import numpy as np
from nmf.NonNegMatrix import NonNegMatrix

class Data_Gen:
    def __init__(self, params: dict) -> None:
        """
        Initialize data generator with parameters provided in a dict.

        Expected keys in params:
            - m: int (number of rows)
            - n: int (number of columns)
            - r: int (rank, optional for some generators)
            - random_state: optional int seed for reproducibility

        Example:
            Data_Gen({'m': 10, 'n': 5, 'r': 3, 'random_state': 42})
        """
        self.params = dict(params)
        try:
            self.m = int(self.params['m'])
            self.n = int(self.params['n'])
            self.r = int(self.params['r'])
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {e}") from e
        random_state = self.params.get('random_state', None)
        if random_state is not None:
            self.rng = np.random.default_rng(random_state)
        else:
            self.rng = None

    def generate_data(self, distr: str = "normal") -> NonNegMatrix:
        """
        Generate a non-negative data matrix based on the distribution type.

        :param distr: 'uniform' | 'normal' | 'poisson'
        :return: NonNegMatrix of shape (m, n)
        """
        rng = self.rng if self.rng is not None else np.random

        match distr:
            case 'uniform':
                data = rng.uniform(low=0.0, high=1.0, size=(self.m, self.n))
            case 'normal':
                # take absolute to ensure non-negativity
                data = np.abs(rng.normal(loc=0.0, scale=1.0, size=(self.m, self.n)))
            case 'poisson':
                data = rng.poisson(lam=5.0, size=(self.m, self.n))
            case _:
                raise ValueError(f"Unsupported distribution type: {distr}")

        self.data = NonNegMatrix(data)
        return self.data





