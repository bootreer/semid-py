"""Half-trek criterion (HTC) identification."""

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from semid_py.mixed_graph import MixedGraph
from semid_py.utils import IdentifierResult, IdentifyStepResult

from .types import GenericIDResult


def create_htc_identifier(
    id_func: Callable[[NDArray], IdentifierResult],
    sources: list[int],
    targets: list[int],
    node: int,
    htr_sources: list[int],
) -> Callable[[NDArray], IdentifierResult]:
    """
    Creates an htc idenfication function.

    Creates an identifier function based on its given paramets. This created
    identifier function will identify the directed edges from `targets` to `node`.

    Args:
        `id_func`: Previous identifier function to compose with
        `sources`: Source nodes of the half-trek system
        `targets`: Target nodes of the half-trek system (should be parents of `node`)
        `node`: The node whose incoming edges we're identifying
        `htr_sources`: Sources that are half-trek reachable from node

    Returns:
        An identification function
    """

    def identifier(Sigma: NDArray) -> IdentifierResult:
        identified_params = id_func(Sigma)
        Lambda = identified_params.Lambda.copy()
        Omega = identified_params.Omega.copy()

        SigmaMinus = Sigma.copy()
        for source in htr_sources:
            # Remove contribution: Sigma[source, :] - Lambda[:, source]^T @ Sigma
            SigmaMinus[source, :] = Sigma[source, :] - Lambda[:, source] @ Sigma

        # solving SigmaMinus[sources, targets] @ x = SigmaMinus[sources, node] for Lambda[targets, node]
        try:
            Lambda[targets, node] = np.linalg.solve(
                SigmaMinus[np.ix_(sources, targets)], SigmaMinus[sources, node]
            )
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "In identification, found near-singular system. Is the input matrix generic?"
            ) from e

        if not np.any(np.isnan(Lambda)):
            I_minus_Lambda = np.eye(len(Lambda)) - Lambda
            Omega = I_minus_Lambda.T @ Sigma @ I_minus_Lambda

        return IdentifierResult(Lambda, Omega)

    return identifier


def htc_identify_step(
    mixed_graph: MixedGraph,
    unsolved_parents: list[list[int]],
    solved_parents: list[list[int]],
    identifier: Callable[[NDArray], IdentifierResult],
) -> IdentifyStepResult:
    """
    Perform one iteration of HTC idenfication.

    This function performs one step through all the nodes in the mixed graph
    and tries to identify new edge coefficients using the existence of half-trek
    systems as described in Foygel, Draisma, Drton (2012).

    Args:
        `mixed_graph`: The mixed graph
        `unsolved_parents`: List of unsolved parent edges for each node
        `solved_parents`: List of solved parent edges for each node
        `identifier`: Current identifier function

    Returns:
        dict with:
            - `identified_edges`: List of newly identified edges as (parent, child) tuples
            - `unsolved_parents`: Updated unsolved parents
            - `solved_parents`: Updated solved parents
            - `identifier`: Updated identifier function
    """
    identified_edges = []
    all_nodes = list(range(mixed_graph.num_nodes))

    # Find nodes that are already solved (no unsolved parents)
    solved_nodes = [i for i in all_nodes if len(unsolved_parents[i]) == 0]

    # TODO: frozen dict in python 3.15
    for i in all_nodes:
        if i in solved_nodes:
            continue

        htr_from_node = set(mixed_graph.htr_from([i]))

        siblings_of_i = set(np.flatnonzero(mixed_graph.b_adj[i, :]))

        # Allowed nodes: solved nodes that are not siblings of i,
        # plus nodes not half-trek-reachable from i
        allowed_nodes = []
        for node in all_nodes:
            if node in solved_nodes and node not in siblings_of_i:
                allowed_nodes.append(node)
            elif node not in htr_from_node:
                allowed_nodes.append(node)

        # Get parents of node i
        node_parents = unsolved_parents[i] + solved_parents[i]

        if len(allowed_nodes) < len(node_parents):
            continue  # Not enough allowed nodes

        # Check if half-trek system exists
        half_trek_result = mixed_graph.get_half_trek_system(
            from_nodes=allowed_nodes, to_nodes=node_parents
        )

        if half_trek_result.system_exists:
            # All parents of i are now identified
            for parent in node_parents:
                if parent in unsolved_parents[i]:
                    identified_edges.append((parent, i))

            # Update solved/unsolved
            solved_parents[i] = node_parents
            unsolved_parents[i] = []

            # Create HTC identifier for this node
            active_from = half_trek_result.active_from
            # htr_sources are nodes that are both in active_from AND half-trek-reachable from i
            htr_sources = [node for node in active_from if node in htr_from_node]

            identifier = create_htc_identifier(
                identifier,
                active_from,
                node_parents,
                i,
                htr_sources,
            )

    return IdentifyStepResult(
        identified_edges,
        unsolved_parents,
        solved_parents,
        identifier,
    )


def htc_id(
    mixed_graph: MixedGraph,
    tian_decompose: bool = False,
) -> GenericIDResult:
    """Run HTC identification on a mixed graph."""
    from .core import general_generic_id

    return general_generic_id(mixed_graph, [htc_identify_step], tian_decompose)
