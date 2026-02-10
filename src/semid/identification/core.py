"""Core identification algorithms."""

from typing import Callable

import numpy as np

from semid.mixed_graph import MixedGraph

from .base import (
    create_identifier_base_case,
    create_simple_bidir_identifier,
    tian_identifier,
)
from .types import GenericIDResult, SEMIDResult
from .htc import htc_identify_step


def general_generic_id(
    mixed_graph: MixedGraph,
    id_step_functions: list[Callable],
    tian_decompose: bool = False,
) -> GenericIDResult:
    """
    General template for generic identification algorithms.

    Applies the identifier functions in list `id_step_functions` to identify
    as many parameters as possible

    Args:
        `mixed_graph`: The mixed graph to identify
        `id_step_functions`: List of identification step functions to apply
        `tian_decompose`: Whether to use Tian decomposition (default: False).
                        In general, enabling this will make the algorithm
                        faster and more powerful

    Returns:
        GenericIDResult with solved/unsolved edges and identifier function
    """
    m = mixed_graph.num_nodes

    if not tian_decompose:
        unsolved_parents = [
            list(np.flatnonzero(mixed_graph.d_adj[:, i])) for i in range(m)
        ]
        solved_parents = [[] for _ in range(m)]
        identifier = create_identifier_base_case(mixed_graph.d_adj, mixed_graph.b_adj)

        while True:
            changed = False
            for id_step_function in id_step_functions:
                id_result = id_step_function(
                    mixed_graph, unsolved_parents, solved_parents, identifier
                )

                if id_result.identified_edges:
                    changed = True
                    unsolved_parents = id_result.unsolved_parents
                    solved_parents = id_result.solved_parents
                    identifier = id_result.identifier
                    break

            if not changed:
                break

        total_unsolved = sum(len(parents) for parents in unsolved_parents)

        if total_unsolved == 0:
            identifier = create_simple_bidir_identifier(identifier)
            solved_siblings = [
                list(np.flatnonzero(mixed_graph.b_adj[i, :])) for i in range(m)
            ]
            unsolved_siblings = [[] for _ in range(m)]
        else:
            solved_siblings = [[] for _ in range(m)]
            unsolved_siblings = [
                list(np.flatnonzero(mixed_graph.b_adj[i, :])) for i in range(m)
            ]
    else:
        unsolved_parents = [[] for _ in range(m)]
        solved_parents = [[] for _ in range(m)]
        solved_siblings = [[] for _ in range(m)]
        c_components = mixed_graph.tian_decompose()

        comp_results = []
        identifiers = []

        for c_comp in c_components:
            comp_graph = MixedGraph(c_comp.L, c_comp.O)
            comp_result = general_generic_id(
                comp_graph, id_step_functions, tian_decompose=False
            )

            comp_results.append(comp_result)
            identifiers.append(comp_result.identifier)

            # Map results back to original indices
            top_order = c_comp.top_order

            for local_idx, orig_node in enumerate(top_order):
                solved_parents_local = comp_result.solved_parents[local_idx]
                for p_local in solved_parents_local:
                    p_orig = top_order[p_local]
                    if p_orig not in solved_parents[orig_node]:
                        solved_parents[orig_node].append(p_orig)

                unsolved_parents_local = comp_result.unsolved_parents[local_idx]
                for p_local in unsolved_parents_local:
                    p_orig = top_order[p_local]
                    if p_orig not in unsolved_parents[orig_node]:
                        unsolved_parents[orig_node].append(p_orig)

                solved_siblings_local = comp_result.solved_siblings[local_idx]
                for s_local in solved_siblings_local:
                    s_orig = top_order[s_local]
                    if s_orig not in solved_siblings[orig_node]:
                        solved_siblings[orig_node].append(s_orig)

        identifier = tian_identifier(identifiers, c_components)

        unsolved_siblings = [
            [
                j
                for j in range(m)
                if mixed_graph.b_adj[i, j] and j not in solved_siblings[i]
            ]
            for i in range(m)
        ]

    return GenericIDResult(
        solved_parents,
        unsolved_parents,
        solved_siblings,
        unsolved_siblings,
        identifier,
        mixed_graph,
        tian_decompose,
    )


# Identification step functions


def semid(
    mixed_graph: MixedGraph,
    test_global_id: bool = True,
    test_generic_non_id: bool = True,
    id_step_functions: list[Callable] | None = None,
    tian_decompose: bool = False,
) -> SEMIDResult:
    """
    Identifiability of linear structural equation models.

    Tests both global and generic identifiability of linear structural equation models,
    and attempts to identify as many parameters as possible using the provided
    identification algorithms.

    Args:
        `mixed_graph`: The mixed graph to analyze
        `test_global_id`: Whether to test global identifiability
        `test_generic_non_id`: Whether to test generic non-identifiability
        `id_step_functions`: List of identification step functions to use.
                          Defaults to [htc_identify_step]
        `tian_decompose`: Whether to use Tian decomposition (default False)

    Returns:
        SEMIDResult with global/generic identifiability and identified edges

    Example:
        >>> import numpy as np
        >>> from semid import MixedGraph, semid
        >>> # Create Verma graph
        >>> L = np.array([[0, 1, 1, 0],
        ...               [0, 0, 1, 0],
        ...               [0, 0, 0, 1],
        ...               [0, 0, 0, 0]], dtype=np.int32)
        >>> O = np.array([[0, 0, 0, 0],
        ...               [0, 0, 0, 1],
        ...               [0, 0, 0, 0],
        ...               [0, 1, 0, 0]], dtype=np.int32)
        >>> graph = MixedGraph(L, O)
        >>> result = semid(graph)
        >>> print(result)
    """
    if id_step_functions is None:
        id_step_functions = [htc_identify_step]

    # Test global identifiability
    is_global_id = None
    if test_global_id:
        is_global_id = mixed_graph.global_id()

    # Test generic non-identifiability
    is_generic_non_id = None
    if test_generic_non_id:
        if tian_decompose:
            is_generic_non_id = False
            c_components = mixed_graph.tian_decompose()
            for c_comp in c_components:
                comp_graph = MixedGraph(c_comp.L, c_comp.O)
                if comp_graph.non_htc_id():
                    is_generic_non_id = True
                    break
        else:
            is_generic_non_id = mixed_graph.non_htc_id()

    # Run generic identification
    generic_id_result = None
    if id_step_functions:
        generic_id_result = general_generic_id(
            mixed_graph, id_step_functions, tian_decompose
        )

    return SEMIDResult(
        is_global_id,
        is_generic_non_id,
        generic_id_result,
        mixed_graph,
        tian_decompose,
    )
