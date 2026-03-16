"""Shared utility types and helper functions."""
from dataclasses import dataclass
from itertools import combinations
from typing import Iterator

import numpy as np
from numpy.typing import NDArray


def subsets_of_size(items: list, size: int) -> Iterator[list]:
    """Generate all subsets of a given size."""
    for subset in combinations(items, size):
        yield list(subset)


def validate_matrix(mat: NDArray[np.int32]) -> None:
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Input matrix must be square")

    if ((mat != 0) & (mat != 1)).any():
        raise ValueError("Input matrix must only contain 0's and 1's.")

    if mat.diagonal().any():
        raise ValueError("Input matrix must have 0's along the diagonal.")


def validate_matrices(L: NDArray[np.int32], O: NDArray[np.int32]) -> None:  # noqa: E741
    if L.shape != O.shape:
        raise ValueError("L and O must have the same shape.")

    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("L and O must both be square matrices.")

    if ((L != 0) & (L != 1)).any() or ((O != 0) & (O != 1)).any():
        raise ValueError("L and O must only contain 1's and 0's.")

    if L.diagonal().any() or O.diagonal().any():
        raise ValueError("L and O must have 0's along their diagonals.")


def reshape_arr(
    matrix: NDArray[np.int32] | list[int], nodes: int | None = None
) -> NDArray[np.int32]:
    if isinstance(matrix, np.ndarray) and matrix.ndim > 1:
        if matrix.ndim != 2:
            raise ValueError("Matrix must be 2-dimensional")
        return matrix.astype(np.int32)
    else:
        if nodes is None:
            raise ValueError("node count required for 1D array or list")
        return np.array(matrix, dtype=np.int32).reshape((nodes, nodes))


@dataclass(slots=True)
class TrekSystem:
    """Result from trek system computation."""

    system_exists: bool
    active_from: list[int]


@dataclass(slots=True)
class BiNodesResult:
    """Result containing bidirected and incoming nodes."""

    bi_nodes: list[int]
    in_nodes: list[int]


@dataclass(slots=True)
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
