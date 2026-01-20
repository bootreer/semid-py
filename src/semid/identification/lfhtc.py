from itertools import combinations
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from semid.latent_digraph import LatentDigraph

from .types import IdentifierResult, LfhtcIdentifyStepResult, LfhtcIDResult


def subsets_of_size(items: list, size: int):
    """Generate all subsets of a given size."""
    for subset in combinations(items, size):
        yield list(subset)


def validate_latent_nodes_are_sources(graph: LatentDigraph) -> None:
    """
    Validate that latent nodes in a LatentDigraph are sources.

    Raises:
        ValueError: If any latent node has a parent
    """
    for node in graph.latent_nodes():
        if graph.parents([node]):
            raise ValueError(
                f"Latent node {node} in input graph is not a source node "
                "(i.e., it has a parent)."
            )


def latent_digraph_has_simple_numbering(graph: LatentDigraph) -> None:
    """
    Check that a LatentDigraph has appropriate node numbering.

    Verifies that observed nodes are numbered 0 to num_observed-1 and
    latent nodes are numbered num_observed to num_observed+num_latents-1.

    Raises:
        ValueError: If the graph doesn't have simple numbering
    """
    observed_nodes = graph.observed_nodes()
    latent_nodes = graph.latent_nodes()

    expected_observed = list(range(graph.num_observed))
    expected_latent = list(
        range(graph.num_observed, graph.num_observed + graph.num_latents)
    )

    if observed_nodes != expected_observed or latent_nodes != expected_latent:
        raise ValueError(
            "Currently only latent graphs whose vertices are numbered from 0 "
            f"to {graph.num_observed + graph.num_latents - 1} in order are supported."
        )


def create_lf_identifier_base_case(
    graph: LatentDigraph,
) -> Callable[[NDArray], IdentifierResult]:
    """
    Create a latent-factor identifier base case.

    Creates an identifier that returns matrices with NA values for unidentified
    parameters. This is used as a base case when building more complex
    identification functions for latent-factor graphs.

    Args:
        `graph`: A LatentDigraph object representing the latent-factor graph.
               All latent nodes should be source nodes (i.e., have no parents).

    Returns:
        A function that takes a covariance matrix and returns a dict with:
            - `Lambda`: Matrix with NA for unknown coefficients, 0 where no edge exists
            - `Omega`: Matrix with NA for error covariances between siblings
    """
    L = graph.adj
    observed_nodes = graph.observed_nodes()
    latent_nodes = graph.latent_nodes()
    num_observed = len(observed_nodes)

    # Collect pairs of siblings (children of same latent parent) for Omega
    children_of_latents = [graph.children([node]) for node in latent_nodes]
    omega_indices = []
    for children in children_of_latents:
        # Add all pairs of children (siblings)
        for pair in combinations(children, 2):
            omega_indices.append(pair)

    # Remove duplicates
    omega_indices = list(set(omega_indices))

    def identifier(Sigma: NDArray) -> IdentifierResult:
        Lambda = np.full((num_observed, num_observed), np.nan)
        Lambda[L[observed_nodes, :][:, observed_nodes] == 0] = 0

        Omega = np.zeros((num_observed, num_observed))
        for i, j in omega_indices:
            Omega[i, j] = np.nan
            Omega[j, i] = np.nan
        np.fill_diagonal(Omega, np.nan)

        return IdentifierResult(Lambda=Lambda, Omega=Omega)

    return identifier


def create_lf_htc_identifier(
    id_func: Callable[[NDArray], IdentifierResult],
    v: int,
    Y: list[int],
    Z: list[int],
    parents: list[int],
    reachable_y: list[int],
) -> Callable[[NDArray], IdentifierResult]:
    """
    Create a latent-factor half-trek criterion identification function.

    Creates an identifier function based on its given parameters. This created
    identifier will identify the directed edges from 'parents' to 'v'.

    Args:
        `id_func`: Identification of edge coefficients often requires that other
                edge coefficients already be identified. This argument should be
                a function that produces all such identifications. The newly created
                identifier will return these identifications along with its own.
        `v`: The node for which all incoming edges are to be identified
           (the tails of which are parents).
        `Y`: The sources of the latent-factor half-trek system.
        `Z`: The nodes that are reached from Y via a latent-factor half-trek of the
           form y <- h -> z where h is an element of L.
        `parents`: The parents of node v (observed parents).
        `reachable_y`: The nodes in `Y` which are latent-factor half-trek reachable
                    from `Z` or `v` by avoiding the nodes in L. All incoming edges to
                    these nodes should be identified by `id_func` for the newly created
                    identification function to work.

    Returns:
        An identification function
    """
    Y = sorted(Y)
    Z = sorted(Z)
    parents = sorted(parents)
    reachable_y = sorted(reachable_y)

    def identifier(Sigma: NDArray) -> IdentifierResult:
        m = Sigma.shape[0]
        identified_params = id_func(Sigma)
        Lambda = identified_params.Lambda.copy()

        targets = parents + Z
        SigmaMinus = Sigma.copy()

        for y in Y:
            if y in reachable_y:
                SigmaMinus[y, :] = Sigma[y, :] - Lambda[:, y] @ Sigma

            SigmaMinus[y, Z] = SigmaMinus[y, Z] - SigmaMinus[y, :] @ Lambda[:, Z]

        submatrix = SigmaMinus[np.ix_(Y, targets)]
        try:
            solutions = np.linalg.solve(submatrix, SigmaMinus[Y, v])
            Lambda[parents, v] = solutions[: len(parents)]
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "In identification, found near-singular system. Is the input matrix generic?"
            ) from e

        # If all Lambda are identified, compute Omega
        if not np.any(np.isnan(Lambda)):
            I_minus_Lambda = np.eye(m) - Lambda
            Omega = I_minus_Lambda.T @ Sigma @ I_minus_Lambda
            return IdentifierResult(Lambda=Lambda, Omega=Omega)

        return IdentifierResult(Lambda=Lambda, Omega=identified_params.Omega)

    return identifier


def _find_lf_htc_system(
    graph: LatentDigraph,
    node: int,
    observed_parents: list[int],
    solved_nodes: list[int],
    observed_nodes: list[int],
    latent_nodes: list[int],
    latents_to_control: list[int],
    edges_between_observed: NDArray,
    subset_size_control: int | None,
) -> tuple[list[int], list[int], list[int], list[int]] | None:
    """
    Search for valid L, Z sets for latent-factor HTC identification.

    Returns:
        Tuple of (L, Z, Y, reachable_y) if found, None otherwise
    """
    max_k = (
        len(latents_to_control)
        if subset_size_control is None
        else min(subset_size_control, len(latents_to_control))
    )

    for k in range(max_k + 1):
        for L in subsets_of_size(latents_to_control, k):
            # Allowed nodes for Z
            children_of_L = list(
                {child for latent in L for child in graph.children([latent])}
            )

            allowed_for_Z = [
                n
                for n in children_of_L
                if n in solved_nodes and n != node and n not in observed_parents
            ]

            for Z in subsets_of_size(allowed_for_Z, k):
                # Determine allowed set for Y
                latent_parents_of_Z_and_i = [
                    p for p in graph.parents([node] + Z) if p in latent_nodes
                ]
                latent_parents_not_in_L = [
                    p for p in latent_parents_of_Z_and_i if p not in L
                ]

                htr_from_Z_and_i_avoiding_L = set(graph.descendants([node] + Z)) | set(
                    graph.tr_from(latent_parents_not_in_L)
                )

                # Allowed nodes: exclude nodes that are half-trek reachable AND unsolved
                htr_and_unsolved = htr_from_Z_and_i_avoiding_L - set(solved_nodes)
                allowed = [n for n in observed_nodes if n not in htr_and_unsolved]

                # Remove nodes that are children of latent parents not in L
                children_of_latent_parents_not_in_L = [
                    child
                    for parent in latent_parents_not_in_L
                    for child in graph.children([parent])
                ]

                allowed = [
                    n
                    for n in allowed
                    if n not in children_of_latent_parents_not_in_L
                    and n not in Z
                    and n != node
                ]

                # Check if there is a half-trek system
                avoid_right_edges = [
                    (int(edge[0]), int(edge[1]))
                    for edge in edges_between_observed
                    if edge[1] in Z
                ]

                try:
                    trek_system_result = graph.get_trek_system(
                        from_nodes=allowed,
                        to_nodes=observed_parents + Z,
                        avoid_left_edges=[
                            (int(e[0]), int(e[1])) for e in edges_between_observed
                        ],
                        avoid_right_edges=avoid_right_edges,
                    )

                    if trek_system_result.system_exists:
                        Y = trek_system_result.active_from
                        return (L, Z, Y, list(htr_from_Z_and_i_avoiding_L))

                except ValueError:
                    # Trek system failed, continue searching
                    continue

    return None


def lf_htc_identify_step(
    graph: LatentDigraph,
    unsolved_parents: list[list[int]],
    solved_parents: list[list[int]],
    active_froms: list[list[int]],
    Zs: list[list[int]],
    Ls: list[list[int]],
    identifier: Callable[[NDArray], IdentifierResult],
    subset_size_control: int | None = None,
) -> LfhtcIdentifyStepResult:
    """
    Perform one iteration of latent-factor HTC identification.

    A function that does one step through all the nodes in a latent-factor graph
    and tries to identify new edge coefficients using the existence of
    latent-factor half-trek systems.

    Args:
        `graph`: A LatentDigraph object representing the latent-factor graph.
               All latent nodes should be source nodes (i.e., have no parents).
        `unsolved_parents`: List whose ith index is a vector of all the parents
                         j of i in the graph for which the edge j->i is not yet
                         known to be generically identifiable.
        `solved_parents`: Complement of unsolved_parents, a list whose ith index
                       is a vector of all parents j of i for which the edge j->i
                       is known to be generically identifiable.
        `identifier`: An identification function that must produce the identifications
                   corresponding to those in solved_parents.
        `active_froms`: If node i is solved then the ith index is a vector
                     containing the nodes Y, otherwise it is empty.
        `Zs`: If node i is solved then the ith index is a vector
           containing the nodes Z, otherwise it is empty.
        `Ls`: If node i is solved then the ith index is a vector
           containing the latent nodes L, otherwise it is empty.
        `subset_size_control`: The largest subset of latent nodes to consider
                            (default: None, meaning no limit).

    Returns:
        dict with:
            - `identified_edges`: List of newly identified edges as (parent, child) tuples
            - `unsolved_parents`: Updated unsolved parents
            - `solved_parents`: Updated solved parents
            - `identifier`: Updated identifier function
            - `active_froms`: Updated active_froms
            - `Zs`: Updated Zs
            - `Ls`: Updated Ls
    """
    # Sanity check
    validate_latent_nodes_are_sources(graph)

    # Variable to store all newly identified edges
    identified_edges = []

    # Collect basic info from graph
    observed_nodes = graph.observed_nodes()
    latent_nodes = graph.latent_nodes()
    num_observed = len(observed_nodes)
    edge_mat = graph.adj
    edges_between_observed = np.argwhere(edge_mat[:num_observed, :num_observed] == 1)
    solved_nodes = [i for i in range(num_observed) if len(unsolved_parents[i]) == 0]

    # Only latent nodes with >= 4 children may be possibly in L
    children_of_latent_nodes = [graph.children([x]) for x in latent_nodes]
    latent_node_has_geq4_children = [
        len(children) >= 4 for children in children_of_latent_nodes
    ]
    latents_to_control = [
        latent
        for latent, has_geq4 in zip(latent_nodes, latent_node_has_geq4_children)
        if has_geq4
    ]

    # Loop over all unsolved nodes
    for i in [node for node in observed_nodes if node not in solved_nodes]:
        # Collect basic info of unsolved node i
        all_parents = graph.parents([i])

        # TODO: latent_parents unused
        # latent_parents = [p for p in all_parents if p in latent_nodes]

        observed_parents = [p for p in all_parents if p in observed_nodes]

        # Try to find valid L, Z sets using helper function
        result = _find_lf_htc_system(
            graph,
            i,
            observed_parents,
            solved_nodes,
            observed_nodes,
            latent_nodes,
            latents_to_control,
            edges_between_observed,
            subset_size_control,
        )

        if result is not None:
            L, Z, Y, reachable_y = result

            for parent in observed_parents:
                identified_edges.append((parent, i))

            identifier = create_lf_htc_identifier(
                identifier,
                v=i,
                Y=Y,
                Z=Z,
                parents=observed_parents,
                reachable_y=reachable_y,
            )

            solved_parents[i] = observed_parents
            unsolved_parents[i] = []
            active_froms[i] = Y
            Zs[i] = Z
            Ls[i] = L
            solved_nodes.append(i)

    return LfhtcIdentifyStepResult(
        identified_edges=identified_edges,
        unsolved_parents=unsolved_parents,
        solved_parents=solved_parents,
        identifier=identifier,
        active_froms=active_froms,
        Zs=Zs,
        Ls=Ls,
    )


def lf_htc_id(
    graph: LatentDigraph, subset_size_control: int | None = None
) -> LfhtcIDResult:
    """
    Determine which edges in a latent digraph are LF-HTC-identifiable.

    Uses the latent-factor half-trek criterion to determine which edges in a
    latent digraph are generically identifiable.

    Args:
        `graph`: A LatentDigraph object representing the latent-factor graph.
               All latent nodes should be source nodes (i.e., have no parents).
        `subset_size_control`: Maximum size of latent node subsets to consider
                           (default: None, meaning no limit)

    Returns:
        LfhtcIDResult with identification results
    """
    # Check the graph
    latent_digraph_has_simple_numbering(graph)

    observed_nodes = graph.observed_nodes()
    num_observed = len(observed_nodes)

    unsolved_parents = [graph.observed_parents(i) for i in observed_nodes]
    solved_parents = [[] for _ in range(num_observed)]
    active_froms = [[] for _ in range(num_observed)]
    Zs = [[] for _ in range(num_observed)]
    Ls = [[] for _ in range(num_observed)]
    identifier = create_lf_identifier_base_case(graph)

    while True:
        id_result = lf_htc_identify_step(
            graph,
            unsolved_parents,
            solved_parents,
            active_froms,
            Zs,
            Ls,
            identifier,
            subset_size_control,
        )
        unsolved_parents = id_result.unsolved_parents
        solved_parents = id_result.solved_parents
        active_froms = id_result.active_froms
        Zs = id_result.Zs
        Ls = id_result.Ls
        identifier = id_result.identifier

        if not id_result.identified_edges:
            break

    return LfhtcIDResult(
        solved_parents=solved_parents,
        unsolved_parents=unsolved_parents,
        identifier=identifier,
        graph=graph,
        active_froms=active_froms,
        Zs=Zs,
        Ls=Ls,
    )
