"""Trek separation identification algorithm."""

from itertools import combinations
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from semid_py.mixed_graph import MixedGraph
from semid_py.utils import IdentifierResult, IdentifyStepResult

from .types import GenericIDResult
from .core import general_generic_id


def create_trek_separation_identifier(
    id_func: Callable[[NDArray], IdentifierResult],
    sources: list[int],
    targets: list[int],
    node: int,
    parent: int,
    solved_parents: list[int],
) -> Callable[[NDArray], IdentifierResult]:
    """
    Create an trek separation identification function

    A helper function for `trekSeparationIdentifyStep`, creates an
    identifier function based on its given parameters. This created identifier
    function will identify the directed edge from 'parent' to 'node.'

    Args:
        `id_func`: Previous identifier function to compose with
        `sources`: Source nodes for the trek system
        `targets`: Target nodes (excluding parent and node)
        `node`: The node whose incoming edge we're identifying
        `parent`: The parent whose edge to node we're identifying
        `solved_parents`: Already solved parents of node

    Returns:
        An identification function
    """

    def identifier(Sigma: NDArray) -> IdentifierResult:
        identified_params = id_func(Sigma)
        Lambda = identified_params.Lambda.copy()
        Omega = identified_params.Omega.copy()

        SigmaMinus = Sigma.copy()
        for solved_parent in solved_parents:
            SigmaMinus[sources, node] = (
                SigmaMinus[sources, node]
                - SigmaMinus[sources, solved_parent] * Lambda[solved_parent, node]
            )

        subSigmaNode = SigmaMinus[np.ix_(sources, targets + [node])]
        subSigmaParent = SigmaMinus[np.ix_(sources, targets + [parent])]

        det_node = np.linalg.det(subSigmaNode)
        det_parent = np.linalg.det(subSigmaParent)

        Lambda[parent, node] = det_node / det_parent

        return IdentifierResult(Lambda, Omega)

    return identifier


def _find_trek_separation_edge(
    mixed_graph: MixedGraph,
    node: int,
    parent: int,
    solved_parents_of_node: list[int],
    all_nodes: list[int],
    non_i_descendants: list[int],
    max_subset_size: int,
) -> tuple[list[int], list[int]] | None:
    """
    Search for valid sources and targets to identify edge parent -> node.

    Returns:
        Tuple of (sources, targets) if edge can be identified, None otherwise
    """
    m = len(all_nodes)

    for k in range(1, min(max_subset_size, m) + 1):
        # Generate all k-subsets of sources
        source_sets = list(combinations(all_nodes, k))

        # Generate all (k-1)-subsets of non-descendants excluding parent
        target_candidates = [n for n in non_i_descendants if n != parent]
        target_sets = list(combinations(target_candidates, k - 1))

        for sources in source_sets:
            sources = list(sources)
            for targets in target_sets:
                targets = list(targets)

                system_with_j = mixed_graph.get_trek_system(
                    from_nodes=sources, to_nodes=targets + [parent]
                )

                if system_with_j.system_exists:
                    # Build avoid_right_edges: edges from (parent, solved_parents) to node
                    to_remove_on_right = []
                    for p in [parent] + solved_parents_of_node:
                        to_remove_on_right.append((p, node))

                    # Check if trek system exists from sources to (targets + node)
                    # avoiding edges from parent and solved parents to node
                    system_without_edges = mixed_graph.get_trek_system(
                        from_nodes=sources,
                        to_nodes=targets + [node],
                        avoid_right_edges=to_remove_on_right,
                    )

                    if not system_without_edges.system_exists:
                        return (sources, targets)

    return None


def trek_separation_identify_step(
    mixed_graph: MixedGraph,
    unsolved_parents: list[list[int]],
    solved_parents: list[list[int]],
    identifier: Callable[[NDArray], IdentifierResult],
    max_subset_size: int = 3,
) -> IdentifyStepResult:
    """
    Perform one iteration of trek separation identification.

    A function that does one step through all the nodes in a mixed graph
    and tries to identify new edge coefficients using trek-separation as
    described in Weihs, Robeva, Robinson, et al. (2017).

    Args:
        `mixed_graph`: The mixed graph
        `unsolved_parents`: List of unsolved parent edges for each node
        `solved_parents`: List of solved parent edges for each node
        `identifier`: Current identifier function
        `max_subset_size`: Maximum subset size to consider (default 3)

    Returns:
        dict with:
            - `identified_edges`: List of newly identified edges as (parent, child) tuples
            - `unsolved_parents`: Updated unsolved parents
            - `solved_parents`: Updated solved parents
            - `identifier`: Updated identifier function
    """
    if max_subset_size <= 0:
        raise ValueError("max_subset_size must be >= 1")

    m = mixed_graph.num_nodes
    identified_edges = []
    all_nodes = list(range(m))

    for i in all_nodes:
        unsolved_before = unsolved_parents[i]
        component = mixed_graph.strongly_connected_component(i)
        if not unsolved_before or len(component) != 1:
            continue

        i_descendants = mixed_graph.descendants([i])
        non_i_descendants = [n for n in all_nodes if n not in i_descendants]

        for j in unsolved_before:
            # Try to identify edge j -> i using helper function
            result = _find_trek_separation_edge(
                mixed_graph,
                i,
                j,
                solved_parents[i],
                all_nodes,
                non_i_descendants,
                max_subset_size,
            )

            if result is not None:
                sources, targets = result
                identified_edges.append((j, i))

                identifier = create_trek_separation_identifier(
                    identifier,
                    sources,
                    targets,
                    i,
                    j,
                    solved_parents[i],
                )

                solved_parents[i] = sorted(solved_parents[i] + [j])
                unsolved_parents[i] = [p for p in unsolved_parents[i] if p != j]

    return IdentifyStepResult(
        identified_edges,
        unsolved_parents,
        solved_parents,
        identifier,
    )


def trek_sep_id(
    mixed_graph: MixedGraph,
    max_subset_size: int = 3,
    tian_decompose: bool = False,
) -> GenericIDResult:
    def ts_step(graph, unsolved, solved, identifier):
        return trek_separation_identify_step(
            graph, unsolved, solved, identifier, max_subset_size
        )

    return general_generic_id(mixed_graph, [ts_step], tian_decompose)
