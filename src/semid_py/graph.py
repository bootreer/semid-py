from typing import Optional, Self

import igraph as ig
import numpy as np
from numpy.typing import NDArray

import semid_py.utils as utils


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

    def __init__(self, adj: NDArray[np.int32], num_observed: Optional[int] = None):
        """
        Create a LatentDigraph object.

        Args:
            adj: Adjacency matrix where first num_observed rows are observed nodes
            num_observed: Number of observed nodes. If None, all nodes are observed.
        """
        utils.validate_matrix(adj)
        self.adj = adj
        self.digraph = ig.Graph.Adjacency(matrix=adj, mode="directed")

        if num_observed is None:
            num_observed = adj.shape[0]

        assert 0 <= num_observed <= adj.shape[0], (
            "num_observed must be between 0 and number of nodes"
        )

        self.num_observed = num_observed
        self.num_latents = adj.shape[0] - num_observed

    @property
    def nodes(self) -> int:
        return self.adj.shape[0]

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
            included_nodes.extend(range(self.num_observed, self.nodes))

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
            included_nodes.extend(range(self.num_observed, self.nodes))

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
                self.digraph.neighborhood(vertices=node, order=self.nodes, mode="in")
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
                self.digraph.neighborhood(vertices=node, order=self.nodes, mode="out")
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
        m = self.nodes
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
            nodes: Nodes to start from
            avoid_left_nodes: Nodes to avoid on the left (backward) side
            avoid_right_nodes: Nodes to avoid on the right (forward) side
            include_observed: Whether to include observed nodes in result
            include_latents: Whether to include latent nodes in result

        Returns:
            List of trek-reachable node indices
        """
        if self._tr_graph is None:
            self._tr_graph = self._create_tr_graph()

        m = self.nodes

        # Build set of allowed nodes in trek graph (2m nodes total)
        avoid_set = set(avoid_left_nodes) | set(n + m for n in avoid_right_nodes)
        allowed = [i for i in range(2 * m) if i not in avoid_set]

        # Use BFS to find reachable nodes
        reachable = set()
        for node in nodes:
            result = self._tr_graph.bfs(
                vid=node, mode="out", unreachable=False, restricted=allowed
            )
            reachable.update(result[0])

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

        m = self.nodes
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

    # TODO: review and test
    def get_trek_system(
        self,
        from_nodes: list[int],
        to_nodes: list[int],
        avoid_left_nodes: list[int] = [],
        avoid_right_nodes: list[int] = [],
        avoid_left_edges: list[tuple[int, int]] = [],
        avoid_right_edges: list[tuple[int, int]] = [],
    ) -> dict:
        """
        Determines if a trek system exists in the latent digraph.

        A trek system is a set of node-disjoint treks from from_nodes to to_nodes.
        A trek is a path that goes backwards along directed edges from a source,
        potentially through a latent node, then forwards to a target.

        This uses max-flow to find the maximum number of vertex-disjoint treks.

        Args:
            from_nodes: Source nodes for treks (left side)
            to_nodes: Target nodes for treks (right side)
            avoid_left_nodes: Nodes that cannot appear on left (backward) side
            avoid_right_nodes: Nodes that cannot appear on right (forward) side
            avoid_left_edges: Edges (i,j) that cannot be traversed backward (j->i)
            avoid_right_edges: Edges (i,j) that cannot be traversed forward (i->j)

        Returns:
            dict with:
                - system_exists: True if a trek system of size len(to_nodes) exists
                - active_from: Subset of from_nodes used in the maximal trek system
        """
        m = self.nodes
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
        active_from = [node for node in from_nodes if flow_result.flow[node] > 0]

        return {
            "system_exists": flow_result.value == len(to_nodes),
            "active_from": active_from,
        }

    def induced_subgraph(self, nodes: list[int]) -> Self:
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


# NOTE: 0-indexed unlike in R
class MixedGraph:
    nodes: int

    d_adj: NDArray[np.int32]
    """Directed adjacency matrix (aka L)"""

    b_adj: NDArray[np.int32]
    """Bidirected adjacency matrix (aka O)"""

    directed: ig.Graph
    bidirected: ig.Graph

    # TODO:
    internal: LatentDigraph

    def _reshape(
        self, matrix: NDArray | list[int], nodes: Optional[int] = None
    ) -> NDArray[np.int32]:
        if isinstance(matrix, np.ndarray) and matrix.ndim > 1:
            assert matrix.ndim == 2, "Matrix must be 2-dimensional"
            return matrix.astype(np.int32)
        else:
            assert nodes is not None, "node count required for 1D array or list"
            return np.array(matrix, dtype=np.int32).reshape((nodes, nodes))

    # TODO: named nodes?
    def __init__(
        self,
        d_adj: NDArray[np.int32] | list[int],
        b_adj: NDArray[np.int32] | list[int],
        nodes: Optional[int] = None,
    ):
        self.d_adj = self._reshape(d_adj, nodes)
        self.b_adj = self._reshape(b_adj, nodes)
        utils.validate_matrices(self.d_adj, self.b_adj)

        self.nodes = d_adj.shape[0]
        if nodes is not None:
            assert nodes == self.nodes

        self.directed = ig.Graph.Adjacency(matrix=d_adj, mode="directed")
        self.bidirected = ig.Graph.Adjacency(matrix=b_adj, mode="undirected")

    def is_sibling(self, a: int, b: int) -> bool:
        return self.b_adj[a, b] != 0

    def induced_subgraph(self, nodes: list[int]) -> Self:
        new_L = self.d_adj[nodes, :][:, nodes]
        new_O = self.b_adj[nodes, :][:, nodes]
        return MixedGraph(new_L, new_O)

    def get_trek_system(
        self,
        from_nodes: list[int],
        to_nodes: list[int],
        avoid_left_nodes: list[int] = [],
        avoid_right_nodes: list[int] = [],
        avoid_left_edges: list[tuple[int, int]] = [],
        avoid_right_edges: list[tuple[int, int]] = [],
    ) -> dict:
        """
        Determines if a trek system exists in the mixed graph.

        A trek system is a set of node-disjoint treks from from_nodes to to_nodes.
        A trek is a path that goes backwards along directed edges, then forwards.

        Args:
            from_nodes: The start nodes
            to_nodes: The end nodes
            avoid_left_nodes: Nodes to avoid on the left (backward) side of treks
            avoid_right_nodes: Nodes to avoid on the right (forward) side of treks
            avoid_left_edges: Directed edges to avoid on left side (as list of (i,j) tuples)
            avoid_right_edges: Directed edges to avoid on right side (as list of (i,j) tuples)

        Returns:
            dict with:
                - system_exists: True if a trek system of size len(to_nodes) exists
                - active_from: The subset of from_nodes used in the maximal trek system
        """
        self.internal.get_trek_system(
            from_nodes,
            to_nodes,
            avoid_left_nodes,
            avoid_left_nodes,
            avoid_left_edges,
            avoid_right_edges,
        )

    def get_half_trek_system(
        self,
        from_nodes: list[int],
        to_nodes: list[int],
        avoid_left_nodes: list[int] = [],
        avoid_right_nodes: list[int] = [],
        avoid_right_edges: list[tuple[int, int]] = [],
    ) -> dict:
        """
        Determines if a half-trek system exists in the mixed graph.

        A half-trek system is a trek system where directed edges cannot be used
        on the left (backward) side - only on the right (forward) side.

        Args:
            from_nodes: The start nodes
            to_nodes: The end nodes
            avoid_left_nodes: Nodes to avoid on the left side
            avoid_right_nodes: Nodes to avoid on the right side
            avoid_right_edges: Directed edges to avoid on right side

        Returns:
            dict with:
                - system_exists: True if a half-trek system exists
                - active_from: The subset of from_nodes used
        """
        # Get all directed edges to avoid on the left side
        avoid_left_edges = [
            (i, j)
            for i in range(self.nodes)
            for j in range(self.nodes)
            if self.d_adj[i, j] != 0
        ]

        return self.get_trek_system(
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            avoid_left_nodes=avoid_left_nodes,
            avoid_right_nodes=avoid_right_nodes,
            avoid_left_edges=avoid_left_edges,
            avoid_right_edges=avoid_right_edges,
        )

    #############################################
    # Half-trek Identifiability

    def htc_id(self) -> Optional[list[int]]:
        """
        Determines if a mixed graph is HTC-identifiable

        Uses the half-trek criterion of Foygel, Draisma, and Drton (2012) to check if an input
        mixed graph is generically identifiable.

        Returns:
            List | None of HTC-identifiable nodes. (0-indexed)
        """
        siblings = [np.flatnonzero(self.b_adj[i, :]) for i in range(self.nodes)]
        children = [np.flatnonzero(self.d_adj[i, :]) for i in range(self.nodes)]
        parents = [np.flatnonzero(self.d_adj[:, i]) for i in range(self.nodes)]

        L = 2  # L(i) nodes offset
        R_IN = 2 + self.nodes  # R(i)-in node offset
        R_OUT = 2 + 2 * self.nodes  # R(i)-out node offset
        NODES = 2 + 3 * self.nodes
        cap_init = np.zeros((NODES, NODES), dtype=int)

        for i in range(self.nodes):
            # edge from L(i) to R(i)-in, and to R(j)-in for all siblings j of i
            cols = np.r_[i, siblings[i]]
            cap_init[L + i, R_IN + cols] = 1

            # edge from R(i)-in to R(i)-out
            cap_init[R_IN + i, R_OUT + i] = 1

            # edge from R(i)-out to R(j)-in for all directed edges i->j
            cap_init[R_OUT + i, R_IN + children[i]] = 1

        # when testing if a set A satisfies the HTC with respect to a node i, need to add
        # (1) edge from source to L(j) for all j in A and (2) edge from R(j)-out to
        # target for all parents j of i
        dep = self.b_adj + np.eye(self.nodes, dtype=int)
        for _ in range(self.nodes):
            dep = dep + (dep @ self.d_adj) > 0

        solved = (self.d_adj.sum(axis=0) == 0).astype(int)
        count: int = solved.sum()

        change = True
        while change:
            change = False

            for i in np.flatnonzero(solved == 0):
                a = np.union1d(
                    np.flatnonzero(solved > 0), np.flatnonzero(dep[i, :] == 0)
                )
                # exclude i and its siblings
                b = np.r_[i, siblings[i]]
                A = np.setdiff1d(a, b)

                cap = cap_init.copy()

                cap[0, L + A] = 1
                cap[R_OUT + parents[i], 1] = 1

                graph = ig.Graph.Adjacency(cap)
                flow = ig.Graph.maxflow(graph, source=0, target=1).value

                if flow == parents[i].size:
                    change = True
                    count += 1
                    solved[i] = count

        if count == 0:
            return None

        num_unsolved = np.sum(solved == 0)
        return np.argsort(solved)[num_unsolved:]

    def non_htc_id(self) -> bool:
        """
        Check for generic infinite-to-one via the half-trek criterion.

        Checks if a mixed graph is infinite-to-one using the half-trek criterion presented
        by Foygel, Draisma, and Drton (2012).

        Returns:
            True if the graph could be determined to be generically non-identifiable,
            False if this test was inconclusive.
        """
        if self.nodes == 1:
            return False

        nonsibs: list[tuple[int, int]] = [
            (idx[0], idx[1]) for idx in np.argwhere(self.b_adj == 0) if idx[0] < idx[1]
        ]

        siblings = [np.flatnonzero(self.b_adj[i, :]) for i in range(self.nodes)]
        parents = [np.flatnonzero(self.d_adj[:, i]) for i in range(self.nodes)]
        children = [np.flatnonzero(self.d_adj[i, :]) for i in range(self.nodes)]

        N = len(nonsibs)
        m = self.nodes

        R_IN = 2 + N
        R_OUT = 2 + N + m**2

        cap = np.zeros((2 * m**2 + N + 2, 2 * m**2 + N + 2))

        if N != 0:
            # Edges from source to L{i,j} for each nonsibling pair
            cap[0, 2 + np.arange(N)] = 1

            # edges from source to L{i,j} for each nonsibling pair {i,j} = nonsibs[n,1:2]
            for n, (i, j) in enumerate(nonsibs):
                # Edge from L{i,j} to R_i(j)-in and R_i(k)-in for all siblings k of j
                cap[
                    2 + n,
                    R_IN + i * m + np.r_[j, siblings[j]],
                ] = 1

                # Edge from L{i,j} to R_j(i)-in and R_j(k)-in for all siblings k of i
                cap[
                    2 + n,
                    R_IN + j * m + np.r_[i, siblings[i]],
                ] = 1

        for i in range(self.nodes):
            # edge from R_i(j)-out to target when j is a parent of i
            cap[R_OUT + i * m + parents[i], 1] = 1
            for j in range(m):
                offset = i * m + j

                # edge from R_i(j)-in to R_i(j)-out
                cap[R_IN + offset, R_OUT + offset] = 1
                # edge from R_i(j)-out to R_i(k)-in where j->k is a directed edge
                cap[
                    R_OUT + offset,
                    R_IN + i * m + children[j],
                ] = 1

        graph = ig.Graph.Adjacency(cap)
        result = ig.Graph.maxflow(graph, source=0, target=1).value
        return result < np.sum(self.d_adj)

    #############################################
    # Ancestral Identifiability
    # TODO:

    #############################################
    # Global Identifiability

    def bidirected_components(self) -> list[Self]:
        comps: ig.VertexClustering = self.bidirected.components()
        return [self.induced_subgraph(cluster) for cluster in comps if len(cluster) > 1]

    def global_id(self) -> bool:
        if not self.directed.is_dag():
            return False

        comps = self.bidirected_components()
        while len(comps) > 0:
            comp, *comps = comps
            sinks = [
                i for i, deg in enumerate(comp.directed.degree(mode="out")) if deg == 0
            ]

            if len(sinks) == 1:
                return False
            else:
                for s in sinks:
                    ancestors = comp.directed.neighborhood(
                        vertices=s, order=comp.nodes, mode="in"
                    )
                    if len(ancestors) > 1:
                        ag = comp.induced_subgraph(ancestors)
                        comps.extend(ag.bidirected_components())

        return True
