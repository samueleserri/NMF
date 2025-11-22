import numpy as np

class NonNegMatrix(np.ndarray):
    """
    NumPy ndarray subclass that enforces non-negativity on construction.
    """

    def __new__(cls, input_array):
       
        arr = np.asarray(input_array)

        # ensure numeric float dtype and at least 2-D
        try:
            arr = arr.astype(float, copy=False)
        except Exception as e:
            raise TypeError("Input cannot be converted to a numeric ndarray") from e

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        # check non-negativity
        if np.any(arr < 0):
            raise ValueError("All entries must be non-negative.")

        # create view as subclass
        obj = arr.view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
