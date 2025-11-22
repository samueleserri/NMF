import numpy as np

class NonNegMatrix(np.ndarray):
    """
    This class represent a matrix with non-negative entries and it is an extension of the numpy ndarray.
    """

    def __new__(cls, input_array: np.ndarray) -> "NonNegMatrix":
        # Create the ndarray instance of our class
        obj = np.asarray(input_array).view(cls)
        # Check for non-negativity
        if np.any(obj < 0):
            raise ValueError("All entries must be non-negative.")
        return obj

    def __array_finalize__(self, obj):
        # This method is called whenever a new array is created
        if obj is None:
            return