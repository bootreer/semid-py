"""Base utility functions for identification."""

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from semid.utils import CComponent, validate_matrices
from .types import IdentifierResult


def tian_sigma_for_component(
    Sigma: NDArray, internal: list[int], incoming: list[int], top_order: list[int]
) -> NDArray:
    """
    Globally identify the covariance matrix of a C-component

    The Tian decomposition of a mixed graph G allows one to globally identify
    the covariance matrices Sigma' of special subgraphs of G called c-components.
    This function takes the covariance matrix Sigma corresponding to G and
    a collection of node sets which specify the c-component, and returns the
    Sigma' corresponding to the c-component.

    Args:
        `Sigma`: Covariance matrix for the mixed graph G
        `internal`: Indices of internal nodes (in the bidirected component)
        `incoming`: Indices of incoming nodes (parents of internal, not in component)
        `top_order`: Topological ordering of internal + incoming nodes

    Returns:
        Transformed covariance matrix ordered by top_order
    """
    if not incoming:
        # No incoming nodes, just reorder by topological order
        return Sigma[np.ix_(top_order, top_order)]

    n = len(top_order)
    internal_set = set(internal)
    new_sigma_inv = np.zeros((n, n))

    for j, node in enumerate(top_order):
        if node in internal_set:
            if j == 0:
                new_sigma_inv[j, j] += 1 / Sigma[node, node]
            else:
                # Get indices of previous nodes in topological order
                prev_nodes = top_order[:j]

                Sigma_prev = Sigma[np.ix_(prev_nodes, prev_nodes)]
                # Solve: Sigma_prev @ x = Sigma[prev_nodes, node]
                x = np.linalg.solve(Sigma_prev, Sigma[prev_nodes, node])

                # Schur complement: Sigma[node,node] - Sigma[node,prev] @ Sigma_prev^-1 @ Sigma[prev,node]
                schur = Sigma[node, node] - Sigma[node, prev_nodes] @ x
                schur_inv = 1 / schur

                # Update precision matrix
                new_sigma_inv[j, j] += schur_inv

                # Compute mean contribution
                mean_mat = x * schur_inv

                # Update off-diagonal terms
                new_sigma_inv[j, :j] -= mean_mat
                new_sigma_inv[:j, j] = new_sigma_inv[j, :j]

                new_sigma_inv[:j, :j] += np.outer(x, mean_mat)

    incoming_indices = [i for i, node in enumerate(top_order) if node in incoming]
    new_sigma_inv[np.ix_(incoming_indices, incoming_indices)] += np.eye(len(incoming))
    new_sigma = np.linalg.inv(new_sigma_inv)

    return new_sigma


def create_identifier_base_case(
    L: NDArray,
    O: NDArray,  # noqa: E741
) -> Callable[[NDArray], IdentifierResult]:
    """
    Creates a base case identifier that returns NaN for unidentified parameters.

    Args:
        `L`: Directed adjacency matrix (1 where edge exists, 0 otherwise)
        `O`: Bidirected adjacency matrix (1 where edge exists, 0 otherwise)

    Returns:
        Function that takes covariance matrix and returns dict with Lambda and Omega
    """
    validate_matrices(L, O)
    n = L.shape[0]

    def identifier(Sigma: NDArray) -> IdentifierResult:
        Lambda = np.full((n, n), np.nan)
        Lambda[L == 0] = 0

        Omega = np.full((n, n), np.nan)
        Omega[(O == 0) & ~np.eye(n, dtype=bool)] = 0

        return IdentifierResult(Lambda, Omega)

    return identifier


def create_simple_bidir_identifier(
    base_identifier: Callable[[NDArray], IdentifierResult],
) -> Callable[[NDArray], IdentifierResult]:
    """
    Identify bidirected edges if all directed edges are identified.

    Creates an identifier function that assumes that all directed edges have
    already been identified and then is able to identify all bidirected edges
    simultaneously.

    Args:
        `base_identifier`: Identifier that identifies all directed edges

    Returns:
        Function that identifies everything
    """

    def identifier(Sigma: NDArray) -> IdentifierResult:
        identified_params = base_identifier(Sigma)
        Lambda = identified_params.Lambda.copy()
        Omega = identified_params.Omega.copy()

        if not np.any(np.isnan(Lambda)):
            I_minus_Lambda = np.eye(Lambda.shape[0]) - Lambda
            Omega = I_minus_Lambda.T @ Sigma @ I_minus_Lambda

        return IdentifierResult(Lambda, Omega)

    return identifier


def tian_identifier(
    id_funcs: list[Callable[[NDArray], IdentifierResult]],
    c_components: list[CComponent],
) -> Callable[[NDArray], IdentifierResult]:
    """
    Identifies components in a tian decomposition.

    Creates an identification function which combines the identification
    functions created on a collection of c-components into a identification
    for the full mixed graph.

    Args:
        id_funcs: List of identifier functions, one per c-component
        c_components: List of CComponent objects as returned by tian_decompose

    Returns:
        Combined identifier function for the full graph
    """

    def identifier(Sigma: NDArray) -> IdentifierResult:
        m = Sigma.shape[0]
        Lambda = np.full((m, m), np.nan)
        Omega = np.full((m, m), np.nan)

        for i, c_comp in enumerate(c_components):
            internal = c_comp.internal
            incoming = c_comp.incoming
            top_order = c_comp.top_order

            new_sigma = tian_sigma_for_component(Sigma, internal, incoming, top_order)

            identified_params = id_funcs[i](new_sigma)
            Lambda_comp = identified_params.Lambda
            Omega_comp = identified_params.Omega

            # Map internal node indices in topOrder to positions in Lambda_comp
            internal_indices_in_top = [top_order.index(node) for node in internal]
            Lambda[np.ix_(top_order, internal)] = Lambda_comp[
                :, internal_indices_in_top
            ]
            # Set non-topOrder entries to 0 (matching R implementation)
            non_top_order = [j for j in range(m) if j not in top_order]
            if non_top_order:
                Lambda[np.ix_(non_top_order, internal)] = 0

            Omega[np.ix_(internal, internal)] = Omega_comp[
                np.ix_(internal_indices_in_top, internal_indices_in_top)
            ]
            # Set non-internal entries to 0 (matching R implementation)
            non_internal = [j for j in range(m) if j not in internal]
            if non_internal:
                Omega[np.ix_(non_internal, internal)] = 0
                Omega[np.ix_(internal, non_internal)] = 0

        return IdentifierResult(Lambda, Omega)

    return identifier
