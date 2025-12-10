"""Identification algorithms and infrastructure for SEMID."""
# TODO: Log errors

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from .latent_digraph import LatentDigraph
from .mixed_graph import MixedGraph
from .utils import validate_matrices


@dataclass
class GenericIDResult:
    """
    Result from generic identification algorithms.

    Attributes:
        `solved_parents`: List of solved parent nodes for each node
        `unsolved_parents`: List of unsolved parent nodes for each node
        `solved_siblings`: List of solved sibling nodes for each node
        `unsolved_siblings`: List of unsolved sibling nodes for each node
        `identifier`: Function that takes covariance matrix and returns identified parameters
        `mixed_graph`: The input mixed graph
        `tian_decompose`: Whether Tian decomposition was used
    """

    solved_parents: list[list[int]]
    unsolved_parents: list[list[int]]
    solved_siblings: list[list[int]]
    unsolved_siblings: list[list[int]]
    identifier: Optional[Callable[[NDArray], dict[str, NDArray]]]
    mixed_graph: MixedGraph
    tian_decompose: bool = False

    def __str__(self) -> str:
        """Pretty print the result."""
        n = self.mixed_graph.num_nodes
        num_dir_edges = np.sum(self.mixed_graph.d_adj)
        num_bi_edges = np.sum(self.mixed_graph.b_adj) // 2

        num_solved_dir = sum(len(parents) for parents in self.solved_parents)
        num_solved_bi = sum(len(siblings) for siblings in self.solved_siblings) // 2

        lines = [
            "Generic Identifiability Result",
            "=" * 40,
            f"Mixed Graph: {n} nodes, {num_dir_edges} directed edges, {num_bi_edges} bidirected edges",
            f"Tian decomposition: {self.tian_decompose}",
            "",
            "Identification Summary:",
            f"  Directed edges identified: {num_solved_dir}/{num_dir_edges}",
            f"  Bidirected edges identified: {num_solved_bi}/{num_bi_edges}",
            "",
        ]

        # Show some identified directed edges
        if num_solved_dir > 0:
            lines.append("Identified directed edges:")
            count = 0
            for i in range(n):
                for parent in self.solved_parents[i]:
                    if count < 10:
                        lines.append(f"  {parent} -> {i}")
                        count += 1
                    else:
                        lines.append("  ...")
                        break
                if count >= 10:
                    break
            lines.append("")

        # Show unidentified edges if any
        if num_solved_dir < num_dir_edges:
            num_unsolved = num_dir_edges - num_solved_dir
            lines.append(f"Unidentified directed edges: {num_unsolved}")

        return "\n".join(lines)


@dataclass
class SEMIDResult:
    """
    Complete SEMID result including global and generic identifiability.

    Attributes:
        `is_global_id`: Whether the graph is globally identifiable (None if not tested)
        `is_generic_non_id`: Whether generically non-identifiable (None if not tested)
        `generic_id_result`: GenericIDResult from identification algorithms
        `mixed_graph`: The input mixed graph
        `tian_decompose`: Whether Tian decomposition was used
    """

    is_global_id: Optional[bool]
    is_generic_non_id: Optional[bool]
    generic_id_result: Optional[GenericIDResult]
    mixed_graph: MixedGraph
    tian_decompose: bool = False

    def __str__(self) -> str:
        """Pretty print the result."""
        lines = [
            "SEMID Result",
            "=" * 40,
            f"Tian decomposition: {self.tian_decompose}",
            "",
        ]

        if self.is_global_id is not None:
            lines.append(f"Globally identifiable: {self.is_global_id}")
            lines.append("")

        if self.is_generic_non_id is not None:
            if self.is_generic_non_id:
                status = "TRUE (infinite-to-one parameterization exists)"
            elif (
                self.generic_id_result
                and sum(len(p) for p in self.generic_id_result.unsolved_parents) == 0
            ):
                status = "FALSE (all parameters identified)"
            else:
                status = "INCONCLUSIVE"
            lines.append(f"Generically non-identifiable: {status}")
            lines.append("")

        if self.generic_id_result:
            lines.append(str(self.generic_id_result))

        return "\n".join(lines)


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
    if len(incoming) == 0:
        # No incoming nodes, just reorder by topological order
        return Sigma[np.ix_(top_order, top_order)]

    n = len(top_order)

    # The transformation is: condition on incoming nodes
    # Sigma_new = Sigma_internal - Sigma_internal,incoming @ Sigma_incoming^{-1} @ Sigma_incoming,internal
    internal_indices_in_toporder = [
        i for i, node in enumerate(top_order) if node in internal
    ]
    incoming_indices_in_toporder = [
        i for i, node in enumerate(top_order) if node in incoming
    ]

    if len(incoming) > 0:
        # Extract relevant submatrices from original Sigma
        Sigma_ordered = Sigma[np.ix_(top_order, top_order)]

        Sigma_internal_internal = Sigma_ordered[
            np.ix_(internal_indices_in_toporder, internal_indices_in_toporder)
        ]
        Sigma_internal_incoming = Sigma_ordered[
            np.ix_(internal_indices_in_toporder, incoming_indices_in_toporder)
        ]
        Sigma_incoming_incoming = Sigma_ordered[
            np.ix_(incoming_indices_in_toporder, incoming_indices_in_toporder)
        ]

        try:
            Sigma_incoming_inv = np.linalg.inv(Sigma_incoming_incoming)
            conditional_cov = (
                Sigma_internal_internal
                - Sigma_internal_incoming
                @ Sigma_incoming_inv
                @ Sigma_internal_incoming.T
            )

            # Build the new Sigma with internals having conditional covariance
            # and incoming having identity covariance (treated as exogenous)
            new_sigma = np.eye(n)
            new_sigma[
                np.ix_(internal_indices_in_toporder, internal_indices_in_toporder)
            ] = conditional_cov
        except np.linalg.LinAlgError:
            # If singular, just return ordered Sigma
            new_sigma = Sigma_ordered
    else:
        new_sigma = Sigma[np.ix_(top_order, top_order)]

    return new_sigma


def create_identifier_base_case(L: NDArray, O: NDArray) -> Callable[[NDArray], dict]:  # noqa: E741
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

    def identifier(Sigma: NDArray) -> dict[str, NDArray]:
        Lambda = np.full((n, n), np.nan)
        Lambda[L == 0] = 0

        Omega = np.full((n, n), np.nan)
        Omega[(O == 0) & ~np.eye(n, dtype=bool)] = 0

        return {"Lambda": Lambda, "Omega": Omega}

    return identifier


def create_htc_identifier(
    id_func: Callable[[NDArray], dict],
    sources: list[int],
    targets: list[int],
    node: int,
    htr_sources: list[int],
) -> Callable[[NDArray], dict]:
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

    def identifier(Sigma: NDArray) -> dict[str, NDArray]:
        identified_params = id_func(Sigma)
        Lambda = identified_params["Lambda"].copy()
        Omega = identified_params["Omega"].copy()

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

        return {"Lambda": Lambda, "Omega": Omega}

    return identifier


def create_edgewise_identifier(
    id_func: Callable[[NDArray], dict],
    sources: list[int],
    targets: list[int],
    node: int,
    solved_node_parents: list[int],
    source_parents_to_remove: list[list[int]],
) -> Callable[[NDArray], dict]:
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

    def identifier(Sigma: NDArray) -> dict[str, NDArray]:
        identified = id_func(Sigma)
        Lambda = identified["Lambda"].copy()
        Omega = identified["Omega"].copy()

        SigmaMinus = Sigma.copy()

        for source_idx, source in enumerate(sources):
            parents_to_remove = source_parents_to_remove[source_idx]
            if len(parents_to_remove) > 0:
                SigmaMinus[source, :] = (
                    Sigma[source, :]
                    - Lambda[parents_to_remove, source] @ Sigma[parents_to_remove, :]
                )

        if len(solved_node_parents) > 0:
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

        return {"Lambda": Lambda, "Omega": Omega}

    return identifier


def create_simple_bidir_identifier(
    base_identifier: Callable[[NDArray], dict],
) -> Callable[[NDArray], dict]:
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

    def identifier(Sigma: NDArray) -> dict[str, NDArray]:
        identified_params = base_identifier(Sigma)
        Lambda = identified_params["Lambda"].copy()
        Omega = identified_params["Omega"].copy()

        if not np.any(np.isnan(Lambda)):
            I_minus_Lambda = np.eye(Lambda.shape[0]) - Lambda
            Omega = I_minus_Lambda.T @ Sigma @ I_minus_Lambda

        return {"Lambda": Lambda, "Omega": Omega}

    return identifier


def tian_identifier(
    id_funcs: list[Callable[[NDArray], dict]],
    c_components: list[dict],
) -> Callable[[NDArray], dict]:
    """
    Identifies components in a tian decomposition.

    Creates an identification function which combines the identification
    functions created on a collection of c-components into a identification
    for the full mixed graph.

    Args:
        `id_funcs`: List of identifier functions, one per c-component
        `c_components`: List of c-component dicts with internal, incoming, topOrder
                      as returned by `tian_decompose`.

    Returns:
        Combined identifier function for the full graph
    """

    def identifier(Sigma: NDArray) -> dict[str, NDArray]:
        m = Sigma.shape[0]
        Lambda = np.full((m, m), np.nan)
        Omega = np.full((m, m), np.nan)

        for i, c_comp in enumerate(c_components):
            internal = c_comp["internal"]
            incoming = c_comp["incoming"]
            top_order = c_comp["topOrder"]

            new_sigma = tian_sigma_for_component(Sigma, internal, incoming, top_order)

            identified_params = id_funcs[i](new_sigma)
            Lambda_comp = identified_params["Lambda"]
            Omega_comp = identified_params["Omega"]

            # Map internal node indices in topOrder to positions in Lambda_comp
            internal_indices_in_top = [top_order.index(node) for node in internal]
            Lambda[np.ix_(top_order, internal)] = Lambda_comp[
                :, internal_indices_in_top
            ]
            Omega[np.ix_(internal, internal)] = Omega_comp[
                np.ix_(internal_indices_in_top, internal_indices_in_top)
            ]

        return {"Lambda": Lambda, "Omega": Omega}

    return identifier


def create_trek_separation_identifier(
    id_func: Callable[[NDArray], dict],
    sources: list[int],
    targets: list[int],
    node: int,
    parent: int,
    solved_parents: list[int],
) -> Callable[[NDArray], dict]:
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

    def identifier(Sigma: NDArray) -> dict[str, NDArray]:
        identified_params = id_func(Sigma)
        Lambda = identified_params["Lambda"].copy()
        Omega = identified_params["Omega"].copy()

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

        return {"Lambda": Lambda, "Omega": Omega}

    return identifier


def create_ancestral_identifier(
    id_func: Callable[[NDArray], dict],
    sources: list[int],
    targets: list[int],
    node: int,
    htr_sources: list[int],
    ancestral_subset: list[int],
    c_component: dict,
) -> Callable[[NDArray], dict]:
    """
    Create an ancestral identification function.

    A helper function for `ancestralIdentifyStep`, creates an identifier function
    based on its given parameters. This created identifier function will
    identify the directed edges from 'targets' to 'node.'

    Args:
        `id_func`: Previous identifier function to compose with
        `sources`: Source nodes for the identification (active_from)
        targets: Target nodes (parents being identified)
        `node`: The node whose incoming edges we're identifying
        `htr_sources`: Sources that are half-trek reachable from node
        `ancestral_subset`: The ancestral set containing node
        `c_component`: The c-component dict with internal, incoming, topOrder, L, O

    Returns:
        An identifier function
    """

    def identifier(Sigma: NDArray) -> dict[str, NDArray]:
        # Get previously identified parameters
        identified_params = id_func(Sigma)
        Lambda = identified_params["Lambda"].copy()
        Omega = identified_params["Omega"].copy()

        top_order = c_component["topOrder"]
        internal = c_component["internal"]
        incoming = c_component["incoming"]

        # Create mapping from original indices to ancestral indices
        def to_anc_idx(x):
            return ancestral_subset.index(x)

        # Map everything to ancestral indices
        top_order_anc = [to_anc_idx(x) for x in top_order]
        sources_anc = [to_anc_idx(x) for x in sources]
        targets_anc = [to_anc_idx(x) for x in targets]
        htr_sources_anc = [to_anc_idx(x) for x in htr_sources]
        internal_anc = [to_anc_idx(x) for x in internal]
        incoming_anc = [to_anc_idx(x) for x in incoming]
        node_anc = to_anc_idx(node)

        Sigma_anc = Sigma[np.ix_(ancestral_subset, ancestral_subset)]
        Sigma_anc_tian = tian_sigma_for_component(
            Sigma_anc, internal_anc, incoming_anc, top_order_anc
        )

        def to_tian_idx(x):
            return top_order_anc.index(x)

        sources_tian = [to_tian_idx(x) for x in sources_anc]
        targets_tian = [to_tian_idx(x) for x in targets_anc]
        htr_sources_tian = [to_tian_idx(x) for x in htr_sources_anc]
        node_tian = to_tian_idx(node_anc)

        Lambda_anc_tian = Lambda[np.ix_(ancestral_subset, ancestral_subset)]
        Lambda_anc_tian = Lambda_anc_tian[np.ix_(top_order_anc, top_order_anc)]

        SigmaMinus = Sigma_anc_tian.copy()
        for source in htr_sources_tian:
            SigmaMinus[source, :] = (
                Sigma_anc_tian[source, :] - Lambda_anc_tian[:, source] @ Sigma_anc_tian
            )

        try:
            solutions = np.linalg.solve(
                SigmaMinus[np.ix_(sources_tian, targets_tian)],
                SigmaMinus[sources_tian, node_tian],
            )

            # Map back to original indices
            for i, target_orig_idx in enumerate(targets):
                Lambda[target_orig_idx, node] = solutions[i]

        except np.linalg.LinAlgError as e:
            raise ValueError(
                "In identification, found near-singular system. Is the input matrix generic?"
            ) from e

        return {"Lambda": Lambda, "Omega": Omega}

    return identifier


# Latent-Factor Half-Trek Criterion (lfhtcID) implementation


def subsets_of_size(items: list, size: int):
    """Generate all subsets of a given size."""
    if size == 0:
        yield []
    else:
        for subset in combinations(items, size):
            yield list(subset)


def validate_latent_nodes_are_sources(graph: LatentDigraph) -> None:
    """
    Validate that latent nodes in a LatentDigraph are sources.

    Raises:
        ValueError: If any latent node has a parent
    """
    for node in graph.latent_nodes():
        if len(graph.parents([node])) != 0:
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


def create_lf_identifier_base_case(graph: LatentDigraph) -> Callable[[NDArray], dict]:
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

    def identifier(Sigma: NDArray) -> dict[str, NDArray]:
        Lambda = np.full((num_observed, num_observed), np.nan)
        Lambda[L[observed_nodes, :][:, observed_nodes] == 0] = 0

        Omega = np.zeros((num_observed, num_observed))
        for i, j in omega_indices:
            Omega[i, j] = np.nan
            Omega[j, i] = np.nan
        np.fill_diagonal(Omega, np.nan)

        return {"Lambda": Lambda, "Omega": Omega}

    return identifier


def create_lf_htc_identifier(
    id_func: Callable[[NDArray], dict],
    v: int,
    Y: list[int],
    Z: list[int],
    parents: list[int],
    reachable_y: list[int],
) -> Callable[[NDArray], dict]:
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

    def identifier(Sigma: NDArray) -> dict[str, NDArray]:
        m = Sigma.shape[0]
        identified_params = id_func(Sigma)
        Lambda = identified_params["Lambda"].copy()

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
            return {"Lambda": Lambda, "Omega": Omega}

        return {"Lambda": Lambda, "Omega": identified_params["Omega"]}

    return identifier


def lfhtc_identify_step(
    graph: LatentDigraph,
    unsolved_parents: list[list[int]],
    solved_parents: list[list[int]],
    active_froms: list[list[int]],
    Zs: list[list[int]],
    Ls: list[list[int]],
    identifier: Callable[[NDArray], dict],
    subset_size_control: int = float("inf"),
) -> dict:
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
        subset_size_control: The largest subset of latent nodes to consider.

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

        # Loop over possible cardinalities of L
        max_k = min(subset_size_control, len(latents_to_control))
        for k in range(int(max_k) + 1):
            # Loop over all subsets L in latents_to_control with cardinality k
            for L in subsets_of_size(latents_to_control, k):
                # Allowed nodes for Z
                children_of_L = list(
                    {child for latent in L for child in graph.children([latent])}
                )

                allowed_for_Z = [
                    node
                    for node in children_of_L
                    if node in solved_nodes
                    and node != i
                    and node not in observed_parents
                ]

                for Z in subsets_of_size(allowed_for_Z, k):
                    # Determine allowed set for Y
                    latent_parents_of_Z_and_i = [
                        p for p in graph.parents([i] + Z) if p in latent_nodes
                    ]
                    latent_parents_not_in_L = [
                        p for p in latent_parents_of_Z_and_i if p not in L
                    ]

                    htr_from_Z_and_i_avoiding_L = set(graph.descendants([i] + Z)) | set(
                        graph.tr_from(latent_parents_not_in_L)
                    )

                    # Allowed nodes: exclude nodes that are half-trek reachable AND unsolved
                    htr_and_unsolved = htr_from_Z_and_i_avoiding_L - set(solved_nodes)
                    allowed = [
                        node for node in observed_nodes if node not in htr_and_unsolved
                    ]

                    # Remove nodes that are children of latent parents not in L
                    children_of_latent_parents_not_in_L = [
                        child
                        for parent in latent_parents_not_in_L
                        for child in graph.children([parent])
                    ]

                    allowed = [
                        node
                        for node in allowed
                        if node not in children_of_latent_parents_not_in_L
                        and node not in Z
                        and node != i
                    ]

                    # Check if there is a half-trek system
                    # Avoid starting with edge between two observed nodes
                    # Avoid ending a half-trek in Z with an observed edge
                    avoid_right_edges = [
                        tuple(edge) for edge in edges_between_observed if edge[1] in Z
                    ]

                    try:
                        trek_system_result = graph.get_trek_system(
                            from_nodes=allowed,
                            to_nodes=observed_parents + Z,
                            avoid_left_edges=[tuple(e) for e in edges_between_observed],
                            avoid_right_edges=avoid_right_edges,
                        )

                        # If half-trek system exists, all edges to i are identified
                        if trek_system_result["system_exists"]:
                            for parent in observed_parents:
                                identified_edges.append((parent, i))

                            Y = trek_system_result["active_from"]
                            identifier = create_lf_htc_identifier(
                                identifier,
                                v=i,
                                Y=Y,
                                Z=Z,
                                parents=observed_parents,
                                reachable_y=list(htr_from_Z_and_i_avoiding_L),
                            )

                            solved_parents[i] = observed_parents
                            unsolved_parents[i] = []
                            active_froms[i] = Y
                            Zs[i] = Z
                            Ls[i] = L
                            solved_nodes.append(i)
                            break  # Found solution for this node
                    except ValueError:
                        # Trek system failed, continue searching
                        continue

                if i in solved_nodes:
                    break  # Found solution for this node

            if i in solved_nodes:
                break  # Found solution for this node

    return {
        "identified_edges": identified_edges,
        "unsolved_parents": unsolved_parents,
        "solved_parents": solved_parents,
        "active_froms": active_froms,
        "Zs": Zs,
        "Ls": Ls,
        "identifier": identifier,
    }


@dataclass
class LfhtcIDResult:
    """
    Result from latent-factor HTC identification.

    Attributes:
        solved_parents: List of solved parent nodes for each observed node
        unsolved_parents: List of unsolved parent nodes for each observed node
        identifier: Function that takes covariance matrix and returns identified parameters
        graph: The input LatentDigraph
        active_froms: List of Y nodes for each solved node
        Zs: List of Z nodes for each solved node
        Ls: List of L nodes for each solved node
    """

    solved_parents: list[list[int]]
    unsolved_parents: list[list[int]]
    identifier: Callable[[NDArray], dict[str, NDArray]]
    graph: LatentDigraph
    active_froms: list[list[int]]
    Zs: list[list[int]]
    Ls: list[list[int]]

    def __str__(self) -> str:
        """Pretty print the result."""
        observed_nodes = self.graph.observed_nodes()
        n_observed = len(observed_nodes)
        latent_nodes = self.graph.latent_nodes()
        n_latent = len(latent_nodes)

        num_edges = np.sum(self.graph.adj[observed_nodes, :][:, observed_nodes])
        num_identified = sum(len(parents) for parents in self.solved_parents)

        lines = [
            "Latent-Factor HTC Identification Result",
            "=" * 40,
            f"Latent Digraph: {n_observed} observed nodes, {n_latent} latent nodes",
            f"Total edges between observed nodes: {num_edges}",
            "",
            "Generic Identifiability Summary:",
            f"  Identified edges: {num_identified}/{num_edges}",
            "",
        ]

        # Show some identified edges
        if num_identified > 0:
            lines.append("Identified edges:")
            count = 0
            for i in range(n_observed):
                for parent in self.solved_parents[i]:
                    if count < 10:
                        lines.append(f"  {parent} -> {i}")
                        count += 1
                    else:
                        lines.append("  ...")
                        break
                if count >= 10:
                    break
        else:
            lines.append("No edges identified")

        return "\n".join(lines)


def lf_htc_id(
    graph: LatentDigraph, subset_size_control: int = float("inf")
) -> LfhtcIDResult:
    """
    Determine which edges in a latent digraph are LF-HTC-identifiable.

    Uses the latent-factor half-trek criterion to determine which edges in a
    latent digraph are generically identifiable.

    Args:
        `graph`: A LatentDigraph object representing the latent-factor graph.
               All latent nodes should be source nodes (i.e., have no parents).
        `subset_size_control`: Maximum size of latent node subsets to consider
                           (default: infinity, meaning no limit)

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

    change_flag = True
    while change_flag:
        id_result = lfhtc_identify_step(
            graph,
            unsolved_parents,
            solved_parents,
            active_froms,
            Zs,
            Ls,
            identifier,
            subset_size_control,
        )
        change_flag = len(id_result["identified_edges"]) != 0
        unsolved_parents = id_result["unsolved_parents"]
        solved_parents = id_result["solved_parents"]
        active_froms = id_result["active_froms"]
        Zs = id_result["Zs"]
        Ls = id_result["Ls"]
        identifier = id_result["identifier"]

    return LfhtcIDResult(
        solved_parents=solved_parents,
        unsolved_parents=unsolved_parents,
        identifier=identifier,
        graph=graph,
        active_froms=active_froms,
        Zs=Zs,
        Ls=Ls,
    )


# Main identification algorithm
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
        `tian_decompose`: Whether to use Tian decomposition. In general,
                        enabling this will make the (default: False) will make
                        the algorithm faster and more powerful

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

        change_flag = True
        while change_flag:
            change_flag = False
            for id_step_function in id_step_functions:
                id_result = id_step_function(
                    mixed_graph, unsolved_parents, solved_parents, identifier
                )

                if len(id_result["identified_edges"]) > 0:
                    change_flag = True
                    unsolved_parents = id_result["unsolved_parents"]
                    solved_parents = id_result["solved_parents"]
                    identifier = id_result["identifier"]
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
            comp_graph = MixedGraph(c_comp["L"], c_comp["O"])
            comp_result = general_generic_id(
                comp_graph, id_step_functions, tian_decompose=False
            )

            comp_results.append(comp_result)
            identifiers.append(comp_result.identifier)

            # Map results back to original indices
            top_order = c_comp["topOrder"]

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
        solved_parents=solved_parents,
        unsolved_parents=unsolved_parents,
        solved_siblings=solved_siblings,
        unsolved_siblings=unsolved_siblings,
        identifier=identifier,
        mixed_graph=mixed_graph,
        tian_decompose=tian_decompose,
    )


# Identification step functions


def htc_identify_step(
    mixed_graph: MixedGraph,
    unsolved_parents: list[list[int]],
    solved_parents: list[list[int]],
    identifier: Callable[[NDArray], dict],
) -> dict:
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

        if half_trek_result["system_exists"]:
            # All parents of i are now identified
            for parent in node_parents:
                if parent in unsolved_parents[i]:
                    identified_edges.append((parent, i))

            # Update solved/unsolved
            solved_parents[i] = node_parents
            unsolved_parents[i] = []

            # Create HTC identifier for this node
            active_from = half_trek_result["active_from"]
            # htr_sources are nodes that are both in active_from AND half-trek-reachable from i
            htr_sources = [node for node in active_from if node in htr_from_node]

            identifier = create_htc_identifier(
                id_func=identifier,
                sources=active_from,
                targets=node_parents,
                node=i,
                htr_sources=htr_sources,
            )

    return {
        "identified_edges": identified_edges,
        "unsolved_parents": unsolved_parents,
        "solved_parents": solved_parents,
        "identifier": identifier,
    }


def edgewise_identify_step(
    mixed_graph: MixedGraph,
    unsolved_parents: list[list[int]],
    solved_parents: list[list[int]],
    identifier: Callable[[NDArray], dict],
    subset_size_control: int = 3,
) -> dict:
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
        if len(unsolved) == 0:
            continue

        subset_found = False
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

        if len(allowed_nodes) == 0:
            continue

        # Compute half-trek reachable nodes for each allowed node (for filtering later)
        # Also includes trek-reachable from unsolved parents of each allowed node
        htr_from_allowed_or_tr_from_unsolved = []
        for a in allowed_nodes:
            reachable = set(mixed_graph.htr_from([a]))
            for unsolved_parent in unsolved_parents[a]:
                reachable |= set(mixed_graph.tr_from([unsolved_parent]))
            htr_from_allowed_or_tr_from_unsolved.append(reachable)

        # Determine subset sizes to search
        n_unsolved = len(unsolved)
        # Search both small subsets (1 to subset_size_control) and large subsets (n down to n-subset_size_control+1)
        small_sizes = list(range(1, min(subset_size_control, n_unsolved) + 1))
        large_sizes = list(
            range(
                n_unsolved,
                max(n_unsolved - subset_size_control + 1, 1) - 1,
                -1,
            )
        )
        subset_sizes = sorted(set(small_sizes + large_sizes))

        # Try different subset sizes
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
                    continue  # Not enough allowed nodes

                half_trek_result = mixed_graph.get_half_trek_system(
                    from_nodes=allowed_for_subset, to_nodes=subset
                )

                if half_trek_result["system_exists"]:
                    for parent in subset:
                        identified_edges.append((parent, i))

                    subset_found = True
                    active_from = half_trek_result["active_from"]

                    source_parents_to_remove = [
                        solved_parents[source] for source in active_from
                    ]

                    identifier = create_edgewise_identifier(
                        id_func=identifier,
                        sources=active_from,
                        targets=subset,
                        node=i,
                        solved_node_parents=solved_parents[i],
                        source_parents_to_remove=source_parents_to_remove,
                    )

                    solved_parents[i] = sorted(set(solved_parents[i]) | set(subset))
                    unsolved_parents[i] = [
                        p for p in unsolved_parents[i] if p not in subset
                    ]

                    break

            if subset_found:
                break

    return {
        "identified_edges": identified_edges,
        "unsolved_parents": unsolved_parents,
        "solved_parents": solved_parents,
        "identifier": identifier,
    }


def trek_separation_identify_step(
    mixed_graph: MixedGraph,
    unsolved_parents: list[list[int]],
    solved_parents: list[list[int]],
    identifier: Callable[[NDArray], dict],
    max_subset_size: int = 3,
) -> dict:
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
        if len(unsolved_before) == 0 or len(component) != 1:
            continue

        i_descendants = mixed_graph.descendants([i])
        non_i_descendants = [n for n in all_nodes if n not in i_descendants]

        for j in unsolved_before:
            edge_identified = False

            # Try different subset sizes
            for k in range(1, min(max_subset_size, m) + 1):
                # Generate all k-subsets of sources
                source_sets = list(combinations(all_nodes, k))

                # Generate all (k-1)-subsets of non-descendants excluding j
                target_candidates = [n for n in non_i_descendants if n != j]
                target_sets = list(combinations(target_candidates, k - 1))

                for sources in source_sets:
                    sources = list(sources)
                    for targets in target_sets:
                        targets = list(targets)

                        system_with_j = mixed_graph.get_trek_system(
                            from_nodes=sources, to_nodes=targets + [j]
                        )

                        if system_with_j["system_exists"]:
                            # Build avoid_right_edges: edges from (j, solved_parents) to i
                            to_remove_on_right = []
                            for parent in [j] + solved_parents[i]:
                                to_remove_on_right.append((parent, i))

                            # Check if trek system exists from sources to (targets + i)
                            # avoiding edges from j and solved parents to i
                            system_without_edges = mixed_graph.get_trek_system(
                                from_nodes=sources,
                                to_nodes=targets + [i],
                                avoid_right_edges=to_remove_on_right,
                            )

                            if not system_without_edges["system_exists"]:
                                identified_edges.append((j, i))
                                edge_identified = True

                                identifier = create_trek_separation_identifier(
                                    id_func=identifier,
                                    sources=sources,
                                    targets=targets,
                                    node=i,
                                    parent=j,
                                    solved_parents=solved_parents[i],
                                )

                                solved_parents[i] = sorted(solved_parents[i] + [j])
                                unsolved_parents[i] = [
                                    p for p in unsolved_parents[i] if p != j
                                ]
                                break

                    if edge_identified:
                        break

                if edge_identified:
                    break

    return {
        "identified_edges": identified_edges,
        "unsolved_parents": unsolved_parents,
        "solved_parents": solved_parents,
        "identifier": identifier,
    }


def ancestral_identify_step(
    mixed_graph: MixedGraph,
    unsolved_parents: list[list[int]],
    solved_parents: list[list[int]],
    identifier: Callable[[NDArray], dict],
) -> dict:
    """
    Perform one iteration of ancestral identification.

    A function that does one step through all the nodes in a mixed graph
    and tries to determine if directed edge coefficients are generically
    identifiable by leveraging decomposition by ancestral subsets. See
    Algorithm 1 of Drton and Weihs (2015); this version of the algorithm
    is somewhat different from Drton and Weihs (2015) in that it also works
    on cyclic graphs.

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
    all_nodes = mixed_graph.nodes

    # Find nodes that are already solved (no unsolved parents)
    solved_nodes = [i for i in all_nodes if len(unsolved_parents[i]) == 0]

    # Cache ancestral components for each node
    ancestral_comps = {}

    for i in all_nodes:
        unsolved = unsolved_parents[i]
        if len(unsolved) == 0:
            continue

        if i not in ancestral_comps:
            node_ancestors = mixed_graph.ancestors(i)
            anc_graph = mixed_graph.induced_subgraph(node_ancestors)
            try:
                tian_comp = anc_graph.tian_component(i)
                tian_graph = MixedGraph(
                    tian_comp["L"],
                    tian_comp["O"],
                    vertex_nums=tian_comp["topOrder"],
                )
                ancestral_comps[i] = {
                    "ancestors": node_ancestors,
                    "component": tian_comp,
                    "anc_graph": anc_graph,
                    "tian_graph": tian_graph,
                }
            except ValueError:
                # If Tian component fails, skip this node
                continue

        # Try first ancestral graph
        anc_info = ancestral_comps[i]
        tian_graph = anc_info["tian_graph"]

        node_parents = tian_graph.parents(i)
        htr_from_node = set(tian_graph.htr_from([i]))

        # Filter solved nodes that are in this Tian graph
        solved_in_tian = [s for s in solved_nodes if s in tian_graph.nodes]
        siblings_in_tian = tian_graph.siblings(i)

        allowed_nodes = []
        for node in tian_graph.nodes:
            if node in solved_in_tian and node not in siblings_in_tian:
                allowed_nodes.append(node)
            elif node not in htr_from_node:
                allowed_nodes.append(node)

        if len(allowed_nodes) >= len(node_parents):
            half_trek_result = tian_graph.get_half_trek_system(
                from_nodes=allowed_nodes, to_nodes=node_parents
            )

            if half_trek_result["system_exists"]:
                for parent in node_parents:
                    if parent in unsolved_parents[i]:
                        identified_edges.append((parent, i))

                solved_parents[i] = node_parents
                unsolved_parents[i] = []

                active_from = half_trek_result["active_from"]
                htr_sources = [node for node in active_from if node in htr_from_node]

                identifier = create_ancestral_identifier(
                    id_func=identifier,
                    sources=active_from,
                    targets=node_parents,
                    node=i,
                    htr_sources=htr_sources,
                    ancestral_subset=anc_info["ancestors"],
                    c_component=anc_info["component"],
                )

                solved_nodes.append(i)
                continue

        # Try second ancestor graph
        node_ancestors = mixed_graph.ancestors(i)
        anc_siblings = set(node_ancestors) | set(mixed_graph.siblings(node_ancestors))

        htr_from_i = set(mixed_graph.htr_from([i]))
        all_nodes_set = set(mixed_graph.nodes)
        siblings_of_i = set(mixed_graph.siblings(i))

        allowed_anc_sibs = (
            anc_siblings
            & (set(solved_nodes) | (all_nodes_set - htr_from_i)) - {i} - siblings_of_i
        )

        anc_set = sorted(set(mixed_graph.ancestors([i] + list(allowed_anc_sibs))))
        anc_graph = mixed_graph.induced_subgraph(anc_set)

        try:
            tian_comp = anc_graph.tian_component(i)
            tian_graph = MixedGraph(
                tian_comp["L"],
                tian_comp["O"],
                vertex_nums=tian_comp["topOrder"],
            )

            htr_from = set(tian_graph.htr_from([i]))
            siblings = set(tian_graph.siblings(i))
            solved_in = [s for s in solved_nodes if s in tian_graph.nodes]

            allowed_nodes = set(solved_in) - siblings
            allowed_nodes |= set(tian_graph.nodes) - htr_from
            allowed_nodes_larger = sorted(allowed_nodes & set(tian_graph.nodes))

            if len(allowed_nodes_larger) >= len(node_parents):
                half_trek_result = tian_graph.get_half_trek_system(
                    from_nodes=allowed_nodes_larger, to_nodes=node_parents
                )

                if half_trek_result["system_exists"]:
                    # All parents of i are now identified
                    for parent in node_parents:
                        if parent in unsolved_parents[i]:
                            identified_edges.append((parent, i))

                    solved_parents[i] = node_parents
                    unsolved_parents[i] = []

                    active_from = half_trek_result["active_from"]
                    htr_sources = [node for node in active_from if node in htr_from]

                    identifier = create_ancestral_identifier(
                        id_func=identifier,
                        sources=active_from,
                        targets=node_parents,
                        node=i,
                        htr_sources=htr_sources,
                        ancestral_subset=anc_set,
                        c_component=tian_comp,
                    )

                    solved_nodes.append(i)

        except ValueError:
            pass

    return {
        "identified_edges": identified_edges,
        "unsolved_parents": unsolved_parents,
        "solved_parents": solved_parents,
        "identifier": identifier,
    }


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
        >>> from semid_py import MixedGraph, semid
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
                comp_graph = MixedGraph(c_comp["L"], c_comp["O"])
                if comp_graph.non_htc_id():
                    is_generic_non_id = True
                    break
        else:
            is_generic_non_id = mixed_graph.non_htc_id()

    # Run generic identification
    generic_id_result = None
    if len(id_step_functions) > 0:
        generic_id_result = general_generic_id(
            mixed_graph, id_step_functions, tian_decompose
        )

    return SEMIDResult(
        is_global_id=is_global_id,
        is_generic_non_id=is_generic_non_id,
        generic_id_result=generic_id_result,
        mixed_graph=mixed_graph,
        tian_decompose=tian_decompose,
    )


def htc_id(
    mixed_graph: MixedGraph,
    tian_decompose: bool = False,
) -> GenericIDResult:
    return general_generic_id(mixed_graph, [htc_identify_step], tian_decompose)


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
        identifier: Callable[[NDArray], dict],
    ) -> dict:
        return edgewise_identify_step(
            mixed_graph,
            unsolved_parents,
            solved_parents,
            identifier,
            subset_size_control=subset_size_control,
        )

    return general_generic_id(mixed_graph, [eid_step], tian_decompose)


def ancestral_id(
    mixed_graph: MixedGraph,
    tian_decompose: bool = True,
) -> GenericIDResult:
    """
    Determines which edges in a mixed graph are ancestralID-identifiable

    Uses the an identification criterion of Drton and Weihs (2015); this version
    of the algorithm is somewhat different from Drton and Weihs (2015) in that it
    also works on cyclic graphs.

    Args:
        `mixed_graph`: The mixed graph to analyze
        `tian_decompose`: Whether to use Tian decomposition (default True)

    Returns:
        GenericIDResult with identified edges and identifier function
    """
    return general_generic_id(mixed_graph, [ancestral_identify_step], tian_decompose)


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
