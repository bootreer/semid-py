import numpy as np
from numpy._typing import NDArray


def validate_matrix(mat: NDArray):
    assert len(mat.shape) == 2 and mat.shape[0] == mat.shape[1], (
        "Input matrix must be square"
    )

    assert np.isin(mat, [0, 1]).all(), "Input matrix must only contain 0's and 1's."
    assert (np.diag(mat) == 0).all(), "Input matrix must have 0's along the diagonal."


def validate_matrices(L: NDArray, O: NDArray):  # noqa: E741
    assert L.shape == O.shape, "L and O must have the same shape."

    assert len(L.shape) == 2 and L.shape[0] == L.shape[1], (
        "L and O must both be square matrices."
    )

    assert np.isin(L, [0, 1]).all() and np.isin(O, [0, 1]).all(), (
        "L and O must only contain 1's and 0's."
    )
    assert (np.diag(L) == 0).all() and (np.diag(O) == 0).all(), (
        "L and O must have 0's along their diagonals."
    )


def matrix_to_edgelist(adj: np.ndarray):
    pass
