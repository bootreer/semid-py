from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


def validate_matrix(mat: NDArray):
    if len(mat.shape) != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Input matrix must be square")

    if not np.isin(mat, [0, 1]).all():
        raise ValueError("Input matrix must only contain 0's and 1's.")

    if not (np.diag(mat) == 0).all():
        raise ValueError("Input matrix must have 0's along the diagonal.")


def validate_matrices(L: NDArray, O: NDArray):  # noqa: E741
    if L.shape != O.shape:
        raise ValueError("L and O must have the same shape.")

    if len(L.shape) != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("L and O must both be square matrices.")

    if not (np.isin(L, [0, 1]).all() and np.isin(O, [0, 1]).all()):
        raise ValueError("L and O must only contain 1's and 0's.")

    if not ((np.diag(L) == 0).all() and (np.diag(O) == 0).all()):
        raise ValueError("L and O must have 0's along their diagonals.")


def matrix_to_edgelist(adj: np.ndarray):
    pass


def reshape_arr(
    matrix: NDArray | list[int], nodes: Optional[int] = None
) -> NDArray[np.int32]:
    if isinstance(matrix, np.ndarray) and matrix.ndim > 1:
        if matrix.ndim != 2:
            raise ValueError("Matrix must be 2-dimensional")
        return matrix.astype(np.int32)
    else:
        if nodes is None:
            raise ValueError("node count required for 1D array or list")
        return np.array(matrix, dtype=np.int32).reshape((nodes, nodes))


@dataclass
class TrekSystem:
    """Result from trek system computation."""

    system_exists: bool
    active_from: list[int]


@dataclass
class BiNodesResult:
    """Result containing bidirected and incoming nodes."""

    bi_nodes: list[int]
    in_nodes: list[int]


@dataclass
class CComponent:
    """
    A c-component from Tian decomposition.

    Attributes:
        internal: Nodes in the bidirected component (external vertex IDs)
        incoming: Nodes that are parents of internal nodes (external vertex IDs)
        top_order: Topological ordering of internal + incoming nodes (external vertex IDs)
        L: Directed adjacency matrix for this component
        O: Bidirected adjacency matrix for this component
    """

    internal: list[int]
    incoming: list[int]
    top_order: list[int]
    L: NDArray[np.int32]  # noqa: E741
    O: NDArray[np.int32]  # noqa: E741
