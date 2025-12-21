"""Edgewise identification algorithm."""

from itertools import combinations
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from semid.mixed_graph import MixedGraph

from .types import GenericIDResult, IdentifierResult, IdentifyStepResult

from .htc import htc_identify_step
from .core import general_generic_id
from .trek_separation import trek_separation_identify_step


def create_edgewise_identifier(
    id_func: Callable[[NDArray], IdentifierResult],
    sources: list[int],
    targets: list[int],
    node: int,
    solved_node_parents: list[int],
    source_parents_to_remove: list[list[int]],
) -> Callable[[NDArray], IdentifierResult]:
    """
    Create an edgewise identification function.

    A helper function for `edgewise_identify_step`, creates an identifier
    function based on its given parameters. This created identifier will
    identify the directed edges from `targets` to `node`.

    Args:
        `id_func`: Previous identifier function to compose with
        `sources`: Source nodes of the half-trek system
        `targets`: Target nodes of the half-trek system (should be parents of `node`)
        `node`: The node whose incoming edges we're identifying
        `solved_node_parents`: Already-solved parents of node (from previous steps)
        `source_parents_to_remove`: A list of parents of sources that should have their edge
                                   to their respective source removed

    Returns:
        An identification function
    """

    def identifier(Sigma: NDArray) -> IdentifierResult:
        identified = id_func(Sigma)
        Lambda = identified.Lambda.copy()
        Omega = identified.Omega.copy()

        SigmaMinus = Sigma.copy()

        for source_idx, source in enumerate(sources):
            parents_to_remove = source_parents_to_remove[source_idx]
            if parents_to_remove:
                SigmaMinus[source, :] = (
                    Sigma[source, :]
                    - Lambda[parents_to_remove, source] @ Sigma[parents_to_remove, :]
                )

        if solved_node_parents:
            SigmaMinus[sources, node] = (
                SigmaMinus[np.ix_(sources, [node])].flatten()
                - SigmaMinus[np.ix_(sources, solved_node_parents)]
                @ Lambda[solved_node_parents, node]
            )

        try:
            Lambda[targets, node] = np.linalg.solve(
                SigmaMinus[np.ix_(sources, targets)], SigmaMinus[sources, node]
            )
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "In identification, found near-singular system. Is the input matrix generic?"
            ) from e

        return IdentifierResult(Lambda, Omega)

    return identifier


def _find_edgewise_subset(
    mixed_graph: MixedGraph,
    node: int,
    unsolved: list[int],
    allowed_nodes: list[int],
    htr_from_allowed_or_tr_from_unsolved: list[set[int]],
    subset_size_control: int,
) -> tuple[list[int], list[int]] | None:
    """
    Search for a valid subset and active sources for edgewise identification.

    Returns:
        Tuple of (subset, active_from) if found, None otherwise
    """
    n_unsolved = len(unsolved)
    # Search both small subsets (1 to subset_size_control) and large subsets
    small_sizes = list(range(1, min(subset_size_control, n_unsolved) + 1))
    large_sizes = list(
        range(
            n_unsolved,
            max(n_unsolved - subset_size_control + 1, 1) - 1,
            -1,
        )
    )
    subset_sizes = sorted(set(small_sizes + large_sizes))

    for k in subset_sizes:
        subsets = combinations(unsolved, k)

        for subset in subsets:
            subset = list(subset)

            # Filter allowed nodes for this subset
            allowed_for_subset = []
            for idx, a in enumerate(allowed_nodes):
                reachable = htr_from_allowed_or_tr_from_unsolved[idx]
                intersection_with_unsolved = reachable & set(unsolved)

                if intersection_with_unsolved.issubset(set(subset)):
                    allowed_for_subset.append(a)

            if len(allowed_for_subset) < len(subset):
                continue

            half_trek_result = mixed_graph.get_half_trek_system(
                from_nodes=allowed_for_subset, to_nodes=subset
            )

            if half_trek_result.system_exists:
                return (subset, half_trek_result.active_from)

    return None


def edgewise_identify_step(
    mixed_graph: MixedGraph,
    unsolved_parents: list[list[int]],
    solved_parents: list[list[int]],
    identifier: Callable[[NDArray], IdentifierResult],
    subset_size_control: int = 3,
) -> IdentifyStepResult:
    """
    Perform one iteration of edgewise identification.

    A function that does one step through all the nodes in a mixed graph
    and tries to identify new edge coefficients using the existence of
    half-trek systems as described in Weihs, Robeva, Robinson, et al. (2017).

    Args:
        `mixed_graph`: The mixed graph
        `unsolved_parents`: List of unsolved parent edges for each node
        `solved_parents`: List of solved parent edges for each node
        `identifier`: Current identifier function
        `subset_size_control`: Controls subset search size (default 3)

    Returns:
        dict with:
            - `identified_edges`: List of newly identified edges as (parent, child) tuples
            - `unsolved_parents`: Updated unsolved parents
            - `solved_parents`: Updated solved parents
            - `identifier`: Updated identifier function
    """
    identified_edges = []
    all_nodes = list(range(mixed_graph.num_nodes))

    for i in all_nodes:
        unsolved = unsolved_parents[i]
        if not unsolved:
            continue

        htr_from_node = set(mixed_graph.htr_from([i]))

        allowed_nodes = []
        for j in all_nodes:
            if i == j:
                continue
            if mixed_graph.is_sibling(i, j):
                continue

            # Check if j has half-trek to at least one unsolved parent of i
            htr_from_j = set(mixed_graph.htr_from([j]))
            if len(htr_from_j & set(unsolved)) == 0:
                continue

            # Check that i's half-trek reachable nodes don't intersect with unsolved parents of j
            if len(htr_from_node & set(unsolved_parents[j])) > 0:
                continue

            allowed_nodes.append(j)

        if not allowed_nodes:
            continue

        # Compute half-trek reachable nodes for each allowed node (for filtering later)
        # Also includes trek-reachable from unsolved parents of each allowed node
        htr_from_allowed_or_tr_from_unsolved = []
        for a in allowed_nodes:
            reachable = set(mixed_graph.htr_from([a]))
            for unsolved_parent in unsolved_parents[a]:
                reachable |= set(mixed_graph.tr_from([unsolved_parent]))
            htr_from_allowed_or_tr_from_unsolved.append(reachable)

        result = _find_edgewise_subset(
            mixed_graph,
            i,
            unsolved,
            allowed_nodes,
            htr_from_allowed_or_tr_from_unsolved,
            subset_size_control,
        )

        if result is not None:
            subset, active_from = result

            for parent in subset:
                identified_edges.append((parent, i))

            source_parents_to_remove = [
                solved_parents[source] for source in active_from
            ]

            identifier = create_edgewise_identifier(
                identifier,
                active_from,
                subset,
                i,
                solved_parents[i],
                source_parents_to_remove,
            )

            solved_parents[i] = sorted(set(solved_parents[i]) | set(subset))
            unsolved_parents[i] = [p for p in unsolved_parents[i] if p not in subset]

    return IdentifyStepResult(
        identified_edges,
        unsolved_parents,
        solved_parents,
        identifier,
    )


def edgewise_id(
    mixed_graph: MixedGraph,
    subset_size_control: int = 3,
    tian_decompose: bool = False,
) -> GenericIDResult:
    """
    Determines which edges in a mixed graph are edgewiseID-identifiable

    Uses the edgewise identification criterion of Weihs, Robeva, Robinson, et al.
    (2017) to determine which edges in a mixed graph are generically
    identifiable.

    Args:
        `mixed_graph`: `MixedGraph` object representing the linear structural equation model
        `subset_size_control`: Controls subset search size (default 3)
                           Searches subsets of sizes 1..k and n-k+1..n
        `tian_decompose`: Whether to use Tian decomposition (Default: False)

    Returns:
        GenericIDResult with identified edges and identifier function
    """

    def eid_step(
        mixed_graph: MixedGraph,
        unsolved_parents: list[list[int]],
        solved_parents: list[list[int]],
        identifier: Callable[[NDArray], IdentifierResult],
    ) -> IdentifyStepResult:
        return edgewise_identify_step(
            mixed_graph,
            unsolved_parents,
            solved_parents,
            identifier,
            subset_size_control,
        )

    return general_generic_id(mixed_graph, [eid_step], tian_decompose)


def edgewise_ts_id(
    mixed_graph: MixedGraph,
    subset_size_control: int = 3,
    max_subset_size: int = 3,
    tian_decompose: bool = True,
) -> GenericIDResult:
    """
    Determines which edges in a mixed graph are edgewiseID+TS identifiable

    Uses the edgewise+TS identification criterion of Weihs, Robeva, Robinson, et
    al. (2017) to determine which edges in a mixed graph are generically
    identifiable. In particular this algorithm iterates between the half-trek,
    edgewise, and trek-separation identification algorithms in an attempt to
    identify as many edges as possible, this may be very slow.

    Args:
        `mixed_graph`: The mixed graph to analyze
        `subset_size_control`: Max subset size for edgewise (default 3)
        `max_subset_size`: Max subset size for trek separation (default 3)
        `tian_decompose`: Whether to use Tian decomposition (default True)

    Returns:
        GenericIDResult with identified edges and identifier function
    """

    def eid_step(graph, unsolved, solved, identifier):
        return edgewise_identify_step(
            graph, unsolved, solved, identifier, subset_size_control
        )

    def ts_step(graph, unsolved, solved, identifier):
        return trek_separation_identify_step(
            graph, unsolved, solved, identifier, max_subset_size
        )

    return general_generic_id(
        mixed_graph, [htc_identify_step, eid_step, ts_step], tian_decompose
    )
