from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Optional

import igraph as ig
import numpy as np
from numpy.typing import NDArray

from semid import utils
from semid.utils import IdentifierResult, LfhtcIdentifyStepResult, TrekSystem


class LatentDigraph:
    adj: NDArray[np.int32]
    """Adjacency matrix L"""

    digraph: ig.Graph

    num_observed: int
    """Number of observed nodes (first rows of adj matrix)"""

    num_latents: int
    """Number of latent nodes (remaining rows of adj matrix)"""

    _tr_graph: Optional[ig.Graph] = None
    """Cached trek graph"""

    def __init__(
        self,
        adj: NDArray[np.int32] | list[int],
        num_observed: Optional[int] = None,
        nodes: Optional[int] = None,
    ):
        """
        Create a LatentDigraph object.

        Args:
            adj: Adjacency matrix where first num_observed rows are observed nodes
            num_observed: Number of observed nodes. If None, all nodes are observed.
        """
        self.adj = utils.reshape_arr(adj, nodes)
        utils.validate_matrix(self.adj)
        self.digraph = ig.Graph.Adjacency(matrix=adj, mode="directed")

        if num_observed is None:
            num_observed = self.adj.shape[0]

        if not (0 <= num_observed <= self.adj.shape[0]):
            raise ValueError("num_observed must be between 0 and number of nodes")

        self.num_observed = num_observed
        self.num_latents = self.adj.shape[0] - num_observed

    @property
    def num_nodes(self) -> int:
        return self.adj.shape[0]

    def observed_nodes(self) -> list[int]:
        """Return list of observed node indices."""
        return list(range(self.num_observed))

    def latent_nodes(self) -> list[int]:
        """Return list of latent node indices."""
        return list(range(self.num_observed, self.num_nodes))

    def observed_parents(self, nodes: int | list[int]) -> list[int]:
        """
        Get observed parents of the given nodes.

        Args:
            nodes: Node or list of nodes to find observed parents of

        Returns:
            List of observed parent node indices
        """
        if isinstance(nodes, int):
            nodes = [nodes]
        return self.parents(nodes, include_observed=True, include_latents=False)

    def parents(
        self,
        nodes: list[int],
        include_observed: bool = True,
        include_latents: bool = True,
    ) -> list[int]:
        """
        Get all parents of the given nodes.

        Args:
            nodes: Nodes to find parents of
            include_observed: Whether to include observed nodes in result
            include_latents: Whether to include latent nodes in result

        Returns:
            List of parent node indices
        """
        included_nodes = []
        if include_observed:
            included_nodes.extend(range(self.num_observed))
        if include_latents:
            included_nodes.extend(range(self.num_observed, self.num_nodes))

        # Parents are nodes with edges TO the target nodes
        parent_mask = self.adj[included_nodes, :][:, nodes].sum(axis=1) > 0
        return [included_nodes[i] for i in np.flatnonzero(parent_mask)]

    def children(
        self,
        nodes: list[int],
        include_observed: bool = True,
        include_latents: bool = True,
    ) -> list[int]:
        """
        Get all children of the given nodes.

        Args:
            nodes: Nodes to find children of
            include_observed: Whether to include observed nodes in result
            include_latents: Whether to include latent nodes in result

        Returns:
            List of child node indices
        """
        included_nodes = []
        if include_observed:
            included_nodes.extend(range(self.num_observed))
        if include_latents:
            included_nodes.extend(range(self.num_observed, self.num_nodes))

        # Children are nodes with edges FROM the source nodes
        child_mask = self.adj[nodes, :][:, included_nodes].sum(axis=0) > 0
        return [included_nodes[i] for i in np.flatnonzero(child_mask)]

    def ancestors(
        self,
        nodes: list[int],
        include_observed: bool = True,
        include_latents: bool = True,
    ) -> list[int]:
        """
        Get all ancestors of the given nodes (includes the nodes themselves).

        Args:
            nodes: Nodes to find ancestors of
            include_observed: Whether to include observed nodes in result
            include_latents: Whether to include latent nodes in result

        Returns:
            List of ancestor node indices
        """

        ancestors = set()
        for node in nodes:
            ancestors.update(
                self.digraph.neighborhood(
                    vertices=node, order=self.num_nodes, mode="in"
                )
            )

        result = []
        for node in ancestors:
            if include_observed and node < self.num_observed:
                result.append(node)
            elif include_latents and node >= self.num_observed:
                result.append(node)

        return sorted(result)

    def descendants(
        self,
        nodes: list[int],
        include_observed: bool = True,
        include_latents: bool = True,
    ) -> list[int]:
        """
        Get all descendants of the given nodes (includes the nodes themselves).

        Args:
            nodes: Nodes to find descendants of
            include_observed: Whether to include observed nodes in result
            include_latents: Whether to include latent nodes in result

        Returns:
            List of descendant node indices
        """

        descendants = set()
        for node in nodes:
            descendants.update(
                self.digraph.neighborhood(
                    vertices=node, order=self.num_nodes, mode="out"
                )
            )

        result = []
        for node in descendants:
            if include_observed and node < self.num_observed:
                result.append(node)
            elif include_latents and node >= self.num_observed:
                result.append(node)

        return sorted(result)

    def _create_tr_graph(self) -> ig.Graph:
        """
        Create a trek graph for this latent digraph.

        The trek graph encodes trek reachability. For a graph with m nodes,
        the trek graph has 2m nodes where:
        - Nodes 0..m-1 are "left" copies
        - Nodes m..2m-1 are "right" copies

        Returns:
            igraph Graph representing trek reachability
        """
        m = self.num_nodes
        adj_mat = np.zeros((2 * m, 2 * m), dtype=np.int32)

        for i in range(m):
            adj_mat[i, m + i] = 1

        adj_mat[0:m, 0:m] = self.adj.T
        adj_mat[m : 2 * m, m : 2 * m] = self.adj

        return ig.Graph.Adjacency(adj_mat.tolist(), mode="directed")

    def tr_from(
        self,
        nodes: list[int],
        avoid_left_nodes: list[int] = [],
        avoid_right_nodes: list[int] = [],
        include_observed: bool = True,
        include_latents: bool = True,
    ) -> list[int]:
        """
        Get all nodes that are trek-reachable from the given nodes.

        A trek is a path that goes backwards along directed edges, then forwards.

        Args:
            `nodes`: Nodes to start from
            `avoid_left_nodes`: Nodes to avoid on the left (backward) side
            `avoid_right_nodes`: Nodes to avoid on the right (forward) side
            `include_observed`: Whether to include observed nodes in result
            `include_latents`: Whether to include latent nodes in result

        Returns:
            List of trek-reachable node indices
        """
        if self._tr_graph is None:
            self._tr_graph = self._create_tr_graph()

        m = self.num_nodes

        # Build set of avoided nodes in trek graph (2m nodes total)
        avoid_set = set(avoid_left_nodes) | set(n + m for n in avoid_right_nodes)

        # Manual BFS that respects avoid constraints (python-igraph lacks 'restricted' param)
        # We cannot traverse THROUGH avoided nodes
        reachable = set()
        for start_node in nodes:
            if start_node in avoid_set:
                continue

            queue = [start_node]
            visited = {start_node}
            reachable.add(start_node)

            while queue:
                current = queue.pop(0)

                # Get outgoing neighbors
                neighbors = self._tr_graph.neighbors(current, mode="out")
                for neighbor in neighbors:
                    if neighbor not in visited and neighbor not in avoid_set:
                        visited.add(neighbor)
                        reachable.add(neighbor)
                        queue.append(neighbor)

        # Convert back from trek graph node indices to original graph indices
        # Nodes 0..m-1 map to themselves, nodes m..2m-1 map to 0..m-1
        original_nodes = set()
        for tr_node in reachable:
            if tr_node >= m:
                original_nodes.add(tr_node - m)
            else:
                original_nodes.add(tr_node)

        result = []
        for node in sorted(original_nodes):
            if include_observed and node < self.num_observed:
                result.append(node)
            elif include_latents and node >= self.num_observed:
                result.append(node)

        return result

    def htr_from(
        self,
        nodes: list[int],
        avoid_left_nodes: list[int] = [],
        avoid_right_nodes: list[int] = [],
        include_observed: bool = True,
        include_latents: bool = True,
    ) -> list[int]:
        """
        Get all nodes that are half-trek-reachable from the given nodes.

        A half-trek is a trek where you can only go backwards through the starting nodes,
        then forwards. This is equivalent to trek reachability where we avoid going
        backwards through any node except the starting nodes.

        Args:
            nodes: Nodes to start from
            avoid_left_nodes: Additional nodes to avoid on the left (backward) side
            avoid_right_nodes: Nodes to avoid on the right (forward) side
            include_observed: Whether to include observed nodes in result
            include_latents: Whether to include latent nodes in result

        Returns:
            List of half-trek-reachable node indices
        """
        # Half-trek: avoid all nodes on the left except the starting nodes
        all_nodes = list(range(self.num_nodes))
        additional_avoid = [n for n in all_nodes if n not in nodes]
        combined_avoid_left = list(set(avoid_left_nodes) | set(additional_avoid))

        return self.tr_from(
            nodes=nodes,
            avoid_left_nodes=combined_avoid_left,
            avoid_right_nodes=avoid_right_nodes,
            include_observed=include_observed,
            include_latents=include_latents,
        )

    def _create_trek_flow_graph(self) -> tuple[NDArray[np.int32], int, int]:
        """
        Create a flow graph for computing trek systems.

        The flow graph encodes vertex-disjoint trek paths using vertex capacities.
        Each vertex in the trek graph is split into "in" and "out" parts with
        capacity 1 between them to enforce vertex disjointness.

        Structure for m total nodes:
        - Trek graph has M = 2m nodes (left and right copies)
        - Flow graph has 2M + 2 nodes total:
          - Nodes 0..M-1: "in" parts of trek graph vertices
          - Nodes M..2M-1: "out" parts of trek graph vertices
          - Node 2M: source
          - Node 2M+1: sink

        Returns:
            Tuple of (capacity_matrix, source_idx, sink_idx)
        """
        if self._tr_graph is None:
            self._tr_graph = self._create_tr_graph()

        m = self.num_nodes
        M = 2 * m
        N = 2 * M + 2
        SOURCE = 2 * M
        SINK = 2 * M + 1

        cap = np.zeros((N, N), dtype=np.int32)

        tr_adj = np.array(self._tr_graph.get_adjacency().data, dtype=np.int32)
        cap[M : 2 * M, 0:M] = tr_adj
        for i in range(M):
            cap[i, M + i] = 1

        return cap, SOURCE, SINK

    def get_trek_system(
        self,
        from_nodes: list[int],
        to_nodes: list[int],
        avoid_left_nodes: list[int] = [],
        avoid_right_nodes: list[int] = [],
        avoid_left_edges: list[tuple[int, int]] = [],
        avoid_right_edges: list[tuple[int, int]] = [],
    ) -> TrekSystem:
        """
        Determines if a trek system exists in the latent digraph.

        A trek system is a set of node-disjoint treks from from_nodes to to_nodes.
        A trek is a path that goes backwards along directed edges from a source,
        potentially through a latent node, then forwards to a target.

        This uses max-flow to find the maximum number of vertex-disjoint treks.

        Args:
            `from_nodes`: Source nodes for treks (left side)
            `to_nodes`: Target nodes for treks (right side)
            `avoid_left_nodes`: Nodes that cannot appear on left (backward) side
            `avoid_right_nodes`: Nodes that cannot appear on right (forward) side
            `avoid_left_edges`: Edges (i,j) that cannot be traversed backward (j->i)
            `avoid_right_edges`: Edges (i,j) that cannot be traversed forward (i->j)

        Returns:
            TrekSystem with system_exists and active_from fields
        """
        m = self.num_nodes
        cap, SOURCE, SINK = self._create_trek_flow_graph()

        # Connect source to left copies of from_nodes (these are the "in" parts)
        # Left copies are at indices 0..m-1 in the trek graph
        for node in from_nodes:
            cap[SOURCE, node] = 1

        # Connect right copies of to_nodes to sink (from the "out" parts)
        # Right copies are at indices m..2m-1 in trek graph
        # Out-parts are shifted by M in the flow graph
        M = 2 * m
        for node in to_nodes:
            right_out = M + m + node  # Out-part of right copy
            cap[right_out, SINK] = 1

        # Apply vertex avoidances by setting vertex capacity to 0
        # This blocks the edge from in-part to out-part
        for node in avoid_left_nodes:
            cap[node, M + node] = 0  # Block left copy of node

        for node in avoid_right_nodes:
            right_in = m + node  # Right copy in-part
            cap[right_in, M + right_in] = 0  # Block right copy of node

        # Apply edge avoidances
        # Left edges are reversed in the trek graph (we go backward)
        # If we want to avoid edge i->j on the left, we block j's out-part to i's in-part
        for i, j in avoid_left_edges:
            left_j_out = M + j  # Left copy of j, out-part
            left_i_in = i  # Left copy of i, in-part
            cap[left_j_out, left_i_in] = 0

        # Right edges follow the original direction
        # If we want to avoid edge i->j on the right, we block i's out-part to j's in-part
        for i, j in avoid_right_edges:
            right_i_out = M + m + i  # Right copy of i, out-part
            right_j_in = m + j  # Right copy of j, in-part
            cap[right_i_out, right_j_in] = 0

        flow_graph = ig.Graph.Adjacency(cap, mode="directed")
        flow_result = flow_graph.maxflow(SOURCE, SINK)

        # Determine which from_nodes are active by checking flow from SOURCE to each
        active_from = []
        for node in from_nodes:
            # Get the edge ID from SOURCE to this from_node's in-part
            # from_nodes connect to left copies which are at indices 0..m-1
            edge_id = flow_graph.get_eid(SOURCE, node, directed=True, error=False)
            if edge_id != -1 and flow_result.flow[edge_id] > 0:
                active_from.append(node)

        return TrekSystem(
            system_exists=flow_result.value == len(to_nodes),
            active_from=active_from,
        )

    def induced_subgraph(self, nodes: list[int]) -> LatentDigraph:
        new_adj = self.adj[nodes, :][:, nodes]
        # Determine how many of the selected nodes are observed
        num_obs_in_subgraph = sum(1 for n in nodes if n < self.num_observed)
        return LatentDigraph(new_adj, num_obs_in_subgraph)

    def strongly_connected_component(self, node: int) -> list[int]:
        components = self.digraph.components(mode="strong")
        node_component = components.membership[node]
        return [
            i for i, comp in enumerate(components.membership) if comp == node_component
        ]


# Latent-Factor Half-Trek Criterion (lfhtcID) implementation
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
    identifier: Callable[[NDArray], IdentifierResult]
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


def _find_lfhtc_system(
    graph: LatentDigraph,
    node: int,
    observed_parents: list[int],
    solved_nodes: list[int],
    observed_nodes: list[int],
    latent_nodes: list[int],
    latents_to_control: list[int],
    edges_between_observed: NDArray,
    subset_size_control: Optional[int],
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


def lfhtc_identify_step(
    graph: LatentDigraph,
    unsolved_parents: list[list[int]],
    solved_parents: list[list[int]],
    active_froms: list[list[int]],
    Zs: list[list[int]],
    Ls: list[list[int]],
    identifier: Callable[[NDArray], IdentifierResult],
    subset_size_control: Optional[int] = None,
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
        result = _find_lfhtc_system(
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
    graph: LatentDigraph, subset_size_control: Optional[int] = None
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
