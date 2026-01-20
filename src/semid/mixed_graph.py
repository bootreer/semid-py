from typing import Optional, overload

import igraph as ig
import numpy as np
from numpy.typing import NDArray

from semid import utils
from semid.latent_digraph import LatentDigraph
from semid.utils import BiNodesResult, CComponent, TrekSystem


# NOTE: 0-indexed unlike in R
class MixedGraph:
    d_adj: NDArray[np.int32]
    """Directed adjacency matrix (aka L)"""

    b_adj: NDArray[np.int32]
    """Bidirected adjacency matrix (aka O)"""

    directed: ig.Graph
    bidirected: ig.Graph

    internal: LatentDigraph
    """Internal representation as LatentDigraph with latent confounders"""

    c_components: list[CComponent] | None = None
    _tian_node_map: dict[int, CComponent] | None = None

    _vertex_nums: list[int]

    def __init__(
        self,
        d_adj: NDArray[np.int32] | list[int],
        b_adj: NDArray[np.int32] | list[int],
        nodes: Optional[int] = None,
        vertex_nums: Optional[list[int]] = None,
    ):
        """
        Create a mixed graph.

        Args:
            d_adj: Directed adjacency matrix (L)
            b_adj: Bidirected adjacency matrix (O)
            nodes: Optional hint for reshaping arrays
            vertex_nums: Optional list of original vertex IDs. If None, uses [0, 1, 2, ...]
        """
        self.d_adj = utils.reshape_arr(d_adj, nodes)
        self.b_adj = utils.reshape_arr(b_adj, nodes)
        utils.validate_matrices(self.d_adj, self.b_adj)

        if not (self.b_adj == self.b_adj.T).all():
            self.b_adj = ((self.b_adj + self.b_adj.T) > 0).astype(np.int32)

        # Track original vertex numbers (external IDs)
        n = self.d_adj.shape[0]
        if vertex_nums is None:
            self._vertex_nums = list(range(n))
        else:
            if len(vertex_nums) != n:
                raise ValueError(
                    f"vertex_nums length {len(vertex_nums)} != matrix size {n}"
                )
            self._vertex_nums = list(vertex_nums)

        # Create fast lookup: external_id -> internal_index
        self._vertex_to_idx = {v: i for i, v in enumerate(self._vertex_nums)}

        self.directed = ig.Graph.Adjacency(matrix=self.d_adj, mode="directed")
        self.bidirected = ig.Graph.Adjacency(matrix=self.b_adj, mode="undirected")

        # Create internal LatentDigraph representation
        # Each bidirected edge gets its own latent confounder
        num_observed = self.d_adj.shape[0]
        # Count bidirected edges (upper triangle only, since matrix is symmetric)
        bidirected_edges = np.argwhere(np.triu(self.b_adj, k=1) == 1)
        num_latents = len(bidirected_edges)
        num_total = num_observed + num_latents

        # Build adjacency matrix with latents
        L_with_latents = np.zeros((num_total, num_total), dtype=np.int32)
        # Copy directed edges among observed nodes
        L_with_latents[:num_observed, :num_observed] = self.d_adj

        # Add latent confounders: for each bidirected edge (i,j), add latent L with L->i and L->j
        for latent_idx, (i, j) in enumerate(bidirected_edges):
            latent_node = num_observed + latent_idx
            L_with_latents[latent_node, i] = 1
            L_with_latents[latent_node, j] = 1

        self.internal = LatentDigraph(L_with_latents, num_observed=num_observed)

    @overload
    def to_internal(self, nodes: int) -> int: ...

    @overload
    def to_internal(self, nodes: list[int]) -> list[int]: ...

    def to_internal(self, nodes: int | list[int]) -> int | list[int]:
        """
        Convert external vertex IDs to internal 0-based indices.

        Args:
            nodes: External vertex ID(s)

        Returns:
            Internal index/indices
        """
        if isinstance(nodes, int):
            return self._vertex_to_idx[nodes]
        return [self._vertex_to_idx[n] for n in nodes]

    @overload
    def to_external(self, indices: int) -> int: ...

    @overload
    def to_external(self, indices: list[int]) -> list[int]: ...

    def to_external(self, indices: int | list[int]) -> int | list[int]:
        """
        Convert internal 0-based indices to external vertex IDs.

        Args:
            indices: Internal index/indices

        Returns:
            External vertex ID(s)
        """
        if isinstance(indices, int):
            return self._vertex_nums[indices]
        return [self._vertex_nums[i] for i in indices]

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self.d_adj.shape[0]

    @property
    def nodes(self) -> list[int]:
        """List of vertex IDs in the graph."""
        return self._vertex_nums.copy()

    def is_sibling(self, a: int, b: int) -> bool:
        return self.b_adj[a, b] != 0

    def tr_from(
        self,
        nodes: list[int],
        avoid_left_nodes: list[int] = [],
        avoid_right_nodes: list[int] = [],
    ) -> list[int]:
        """
        Get all nodes that are trek-reachable from the given nodes.

        A trek is a path that goes backwards along directed edges, then forwards.
        In the mixed graph, this uses the internal LatentDigraph representation
        but only returns observed nodes.

        Args:
            `nodes`: Nodes to start from (external vertex IDs)
            `avoid_left_nodes`: Nodes to avoid on the left (external vertex IDs)
            `avoid_right_nodes`: Nodes to avoid on the right (external vertex IDs)

        Returns:
            List of trek-reachable node indices (external vertex IDs)
        """
        # Convert external IDs to internal indices
        internal_nodes = self.to_internal(nodes)
        internal_avoid_left = (
            self.to_internal(avoid_left_nodes) if avoid_left_nodes else []
        )
        internal_avoid_right = (
            self.to_internal(avoid_right_nodes) if avoid_right_nodes else []
        )

        result_internal = self.internal.tr_from(
            nodes=internal_nodes,
            avoid_left_nodes=internal_avoid_left,
            avoid_right_nodes=internal_avoid_right,
            include_observed=True,
            include_latents=False,
        )

        # Convert results back to external IDs
        return self.to_external(result_internal)

    def htr_from(
        self,
        nodes: list[int],
        avoid_left_nodes: list[int] = [],
        avoid_right_nodes: list[int] = [],
    ) -> list[int]:
        """
        Get all nodes that are half-trek-reachable from the given nodes.

        A half-trek is a trek where you can only go backwards through the starting
        nodes, then forwards. Equivalent to avoiding all non-starting nodes on the left.

        Args:
            `nodes`: Nodes to start from (external vertex IDs)
            `avoid_left_nodes`: Additional nodes to avoid on the left (external vertex IDs)
            `avoid_right_nodes`: Nodes to avoid on the right (external vertex IDs)

        Returns:
            List of half-trek-reachable node indices (external vertex IDs)
        """
        # Half-trek: avoid all nodes on the left except the starting nodes
        all_nodes = self.nodes
        additional_avoid = [n for n in all_nodes if n not in nodes]
        combined_avoid_left = list(set(avoid_left_nodes) | set(additional_avoid))

        return self.tr_from(
            nodes=nodes,
            avoid_left_nodes=combined_avoid_left,
            avoid_right_nodes=avoid_right_nodes,
        )

    def induced_subgraph(self, nodes: list[int]) -> MixedGraph:
        """
        Create induced subgraph on the given nodes.

        Args:
            nodes: List of vertex IDs (external) to include in subgraph

        Returns:
            New MixedGraph with preserved vertex numbering
        """
        internal_indices = [self._vertex_to_idx[n] for n in nodes]

        new_L = self.d_adj[np.ix_(internal_indices, internal_indices)]
        new_O = self.b_adj[np.ix_(internal_indices, internal_indices)]

        return MixedGraph(new_L, new_O, vertex_nums=nodes)

    def ancestors(self, nodes: int | list[int]) -> list[int]:
        """
        Get all ancestor nodes (nodes reachable by following directed edges backward).

        Ancestors include the input nodes themselves.

        Args:
            `nodes`: Single node or list of nodes (external vertex IDs)

        Returns:
            Sorted list of all ancestor nodes including the input nodes (external IDs)
        """
        if isinstance(nodes, int):
            nodes = [nodes]
        if not nodes:
            return []

        internal_nodes = [self._vertex_to_idx[n] for n in nodes]

        ancestors_set = set()
        for idx in internal_nodes:
            reachable = self.directed.neighborhood(
                vertices=idx, order=self.num_nodes, mode="in"
            )
            ancestors_set.update(reachable)

        return sorted([self._vertex_nums[i] for i in ancestors_set])

    def parents(self, node: int) -> list[int]:
        """Get parents of a node (using external vertex IDs)."""
        idx = self._vertex_to_idx[node]
        parent_indices = np.flatnonzero(self.d_adj[:, idx])
        return [self._vertex_nums[i] for i in parent_indices]

    def children(self, node: int) -> list[int]:
        """Get children of a node (using external vertex IDs)."""
        idx = self._vertex_to_idx[node]
        child_indices = np.flatnonzero(self.d_adj[idx, :])
        return [self._vertex_nums[i] for i in child_indices]

    def siblings(self, nodes: int | list[int]) -> list[int]:
        """
        Get all siblings (nodes connected by bidirected edges).

        Siblings include the input nodes themselves.

        Args:
            `nodes`: Single node or list of nodes (external vertex IDs)

        Returns:
            Sorted list of all sibling nodes including the input nodes (external IDs)
        """
        if isinstance(nodes, int):
            nodes = [nodes]
        if not nodes:
            return []

        internal_nodes = [self._vertex_to_idx[n] for n in nodes]
        siblings_set = set(internal_nodes)

        for idx in internal_nodes:
            sibs = self.bidirected.neighborhood(vertices=idx, order=1, mode="all")
            siblings_set.update(sibs)

        return sorted([self._vertex_nums[i] for i in siblings_set])

    def descendants(self, nodes: int | list[int]) -> list[int]:
        """
        Get all descendant nodes (nodes reachable by following directed edges forward).

        Descendants include the input nodes themselves.

        Args:
            `nodes`: Single node or list of nodes (external vertex IDs)

        Returns:
            Sorted list of all descendant nodes including the input nodes (external IDs)
        """
        if isinstance(nodes, int):
            nodes = [nodes]
        if not nodes:
            return []

        internal_nodes = [self._vertex_to_idx[n] for n in nodes]

        descendants_set = set()
        for idx in internal_nodes:
            reachable = self.directed.neighborhood(
                vertices=idx, order=self.num_nodes, mode="out"
            )
            descendants_set.update(reachable)

        return sorted([self._vertex_nums[i] for i in descendants_set])

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
        Determines if a trek system exists in the mixed graph.

        A trek system is a set of node-disjoint treks from from_nodes to to_nodes.
        A trek is a path that goes backwards along directed edges, then forwards.

        Args:
            `from_nodes`: The start nodes (external vertex IDs)
            `to_nodes`: The end nodes (external vertex IDs)
            `avoid_left_nodes`: Nodes to avoid on the left (external vertex IDs)
            `avoid_right_nodes`: Nodes to avoid on the right (external vertex IDs)
            `avoid_left_edges`: Directed edges to avoid on left side (external IDs)
            `avoid_right_edges`: Directed edges to avoid on right side (external IDs)

        Returns:
            TrekSystem (with external vertex IDs)
        """
        # Convert all external IDs to internal indices
        internal_from = self.to_internal(from_nodes)
        internal_to = self.to_internal(to_nodes)
        internal_avoid_left = (
            self.to_internal(avoid_left_nodes) if avoid_left_nodes else []
        )
        internal_avoid_right = (
            self.to_internal(avoid_right_nodes) if avoid_right_nodes else []
        )
        internal_avoid_left_edges = (
            [
                (self._vertex_to_idx[i], self._vertex_to_idx[j])
                for i, j in avoid_left_edges
            ]
            if avoid_left_edges
            else []
        )
        internal_avoid_right_edges = (
            [
                (self._vertex_to_idx[i], self._vertex_to_idx[j])
                for i, j in avoid_right_edges
            ]
            if avoid_right_edges
            else []
        )

        result = self.internal.get_trek_system(
            internal_from,
            internal_to,
            internal_avoid_left,
            internal_avoid_right,
            internal_avoid_left_edges,
            internal_avoid_right_edges,
        )

        # Convert active_from back to external IDs
        return TrekSystem(
            system_exists=result.system_exists,
            active_from=(
                self.to_external(result.active_from) if result.system_exists else []
            ),
        )

    def get_half_trek_system(
        self,
        from_nodes: list[int],
        to_nodes: list[int],
        avoid_left_nodes: list[int] = [],
        avoid_right_nodes: list[int] = [],
        avoid_right_edges: list[tuple[int, int]] = [],
    ) -> TrekSystem:
        """
        Determines if a half-trek system exists in the mixed graph.

        A half-trek system is a trek system where directed edges cannot be used
        on the left (backward) side - only on the right (forward) side.

        Args:
            `from_nodes`: The start nodes (external vertex IDs)
            `to_nodes`: The end nodes (external vertex IDs)
            `avoid_left_nodes`: Nodes to avoid on the left side (external IDs)
            `avoid_right_nodes`: Nodes to avoid on the right side (external IDs)
            `avoid_right_edges`: Directed edges to avoid on right side (external IDs)

        Returns:
            TrekSystem with system_exists and active_from fields (external IDs)
        """
        # Get all directed edges to avoid on the left side (in external IDs)
        avoid_left_edges = [
            (self._vertex_nums[i], self._vertex_nums[j])
            for i in range(self.num_nodes)
            for j in range(self.num_nodes)
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
        siblings = [np.flatnonzero(self.b_adj[i, :]) for i in range(self.num_nodes)]
        children = [np.flatnonzero(self.d_adj[i, :]) for i in range(self.num_nodes)]
        parents = [np.flatnonzero(self.d_adj[:, i]) for i in range(self.num_nodes)]

        L = 2  # L(i) nodes offset
        R_IN = 2 + self.num_nodes  # R(i)-in node offset
        R_OUT = 2 + 2 * self.num_nodes  # R(i)-out node offset
        NODES = 2 + 3 * self.num_nodes
        cap_init = np.zeros((NODES, NODES), dtype=int)

        for i in range(self.num_nodes):
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
        dep = self.b_adj + np.eye(self.num_nodes, dtype=int)
        for _ in range(self.num_nodes):
            dep = dep + (dep @ self.d_adj) > 0

        solved = (self.d_adj.sum(axis=0) == 0).astype(int)
        count: int = solved.sum()

        while True:
            changed = False

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
                    changed = True
                    count += 1
                    solved[i] = count

            if not changed:
                break

        if count == 0:
            return None

        num_unsolved = np.sum(solved == 0)
        return list(np.argsort(solved)[num_unsolved:])

    def non_htc_id(self) -> bool:
        """
        Check for generic infinite-to-one via the half-trek criterion.

        Checks if a mixed graph is infinite-to-one using the half-trek criterion presented
        by Foygel, Draisma, and Drton (2012).

        Returns:
            True if the graph could be determined to be generically non-identifiable,
            False if this test was inconclusive.
        """
        if self.num_nodes == 1:
            return False

        nonsibs: list[tuple[int, int]] = [
            (idx[0], idx[1]) for idx in np.argwhere(self.b_adj == 0) if idx[0] < idx[1]
        ]

        siblings = [np.flatnonzero(self.b_adj[i, :]) for i in range(self.num_nodes)]
        parents = [np.flatnonzero(self.d_adj[:, i]) for i in range(self.num_nodes)]
        children = [np.flatnonzero(self.d_adj[i, :]) for i in range(self.num_nodes)]

        N = len(nonsibs)
        m = self.num_nodes

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

        for i in range(self.num_nodes):
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
        return bool(result < np.sum(self.d_adj))

    def strongly_connected_component(self, node: int) -> list[int]:
        """
        Get the strongly connected component containing a node.

        Args:
            `node`: The node (external vertex ID)

        Returns:
            List of nodes in the strongly connected component (external IDs)
        """
        internal_node = self._vertex_to_idx[node]

        scc = self.directed.components(mode="strong")

        for component in scc:
            if internal_node in component:
                return sorted([self._vertex_nums[i] for i in component])

        return [node]

    def tian_decompose(self) -> list[CComponent]:
        """
        Performs Tian decomposition on the mixed graph.

        Returns:
            List of CComponent objects
        """
        if self.c_components is not None:
            return self.c_components

        all_nodes = self.nodes
        scc_components = []
        remaining_nodes = set(all_nodes)

        while remaining_nodes:
            node = min(remaining_nodes)
            component = self.strongly_connected_component(node)  # Returns external IDs
            scc_components.append(component)
            remaining_nodes -= set(component)

        num_components = len(scc_components)

        shrunk_L = np.zeros((num_components, num_components), dtype=np.int32)
        shrunk_O = np.zeros((num_components, num_components), dtype=np.int32)

        for i in range(num_components - 1):
            for j in range(i + 1, num_components):
                comp_i = [self._vertex_to_idx[n] for n in scc_components[i]]
                comp_j = [self._vertex_to_idx[n] for n in scc_components[j]]

                if np.any(self.d_adj[np.ix_(comp_i, comp_j)]):
                    shrunk_L[i, j] = 1
                if np.any(self.d_adj[np.ix_(comp_j, comp_i)]):
                    shrunk_L[j, i] = 1

                if np.any(self.b_adj[np.ix_(comp_i, comp_j)]):
                    shrunk_O[i, j] = 1
                    shrunk_O[j, i] = 1

        bi_graph = ig.Graph.Adjacency(shrunk_O, mode="undirected")
        bi_components = bi_graph.components().membership

        shrunk_graph = ig.Graph.Adjacency(shrunk_L, mode="directed")
        try:
            shrunk_top_order = shrunk_graph.topological_sorting()
        except Exception:
            # If cyclic, just use existing order
            shrunk_top_order = list(range(num_components))

        global_top_order = []
        for comp_idx in shrunk_top_order:
            global_top_order.extend(scc_components[comp_idx])

        num_bi_components = max(bi_components) + 1
        c_components = []

        for bi_comp_idx in range(num_bi_components):
            # Get all SCCs in this bidirected component
            super_nodes = [
                i
                for i, membership in enumerate(bi_components)
                if membership == bi_comp_idx
            ]

            # Internal nodes: all nodes in SCCs belonging to this bidirected component
            # Maintain global topological order
            internal_nodes_set = set()
            for scc_idx in super_nodes:
                internal_nodes_set.update(scc_components[scc_idx])

            internal = [n for n in global_top_order if n in internal_nodes_set]

            # Incoming nodes: parents of internal nodes that are not themselves internal
            # Only include nodes that appear in global_top_order (maintains global order)
            parents_of_internal = set()
            for node in internal:
                node_internal_idx = self._vertex_to_idx[node]
                parent_indices = np.flatnonzero(self.d_adj[:, node_internal_idx])
                parents_of_internal.update(
                    [self._vertex_nums[i] for i in parent_indices]
                )

            incoming_set = parents_of_internal - internal_nodes_set
            incoming = [n for n in global_top_order if n in incoming_set]

            all_ordered_set = internal_nodes_set | incoming_set
            all_ordered = [n for n in global_top_order if n in all_ordered_set]

            internal_indices = [i for i, n in enumerate(all_ordered) if n in internal]
            incoming_indices = [i for i, n in enumerate(all_ordered) if n in incoming]

            comp_size = len(all_ordered)
            new_L = np.zeros((comp_size, comp_size), dtype=np.int32)
            new_O = np.zeros((comp_size, comp_size), dtype=np.int32)

            # Extract submatrices from original graph
            # newL[c(indsInt, indsInc), indsInt] <- L[c(internal, incoming), internal]
            all_rows = internal + incoming
            all_rows_internal = [self._vertex_to_idx[n] for n in all_rows]
            internal_cols_internal = [self._vertex_to_idx[n] for n in internal]

            new_L[np.ix_(internal_indices + incoming_indices, internal_indices)] = (
                self.d_adj[np.ix_(all_rows_internal, internal_cols_internal)]
            )

            # newO[indsInt, indsInt] <- O[internal, internal]
            internal_internal = [self._vertex_to_idx[n] for n in internal]
            new_O[np.ix_(internal_indices, internal_indices)] = self.b_adj[
                np.ix_(internal_internal, internal_internal)
            ]

            c_components.append(
                CComponent(
                    internal,
                    incoming,
                    all_ordered,
                    new_L,
                    new_O,
                )
            )

        self.c_components = c_components
        # Cache node to component mapping for O(1) lookup
        self._tian_node_map = {}
        for comp in c_components:
            for node in comp.internal:
                self._tian_node_map[node] = comp

        return c_components

    def tian_component(self, node: int) -> CComponent:
        """
        Returns the Tian c-component containing a specific node.

        Args:
            node: The node to find the c-component for (external vertex ID)

        Returns:
            CComponent

        Raises:
            ValueError: If node is not found in any component
        """
        if self.c_components is None or self._tian_node_map is None:
            self.tian_decompose()

        # pyrefly: ignore
        if self._tian_node_map and node in self._tian_node_map:
            return self._tian_node_map[node]

        raise ValueError(
            f"No Tian component found for node {node}, was the node mispecified?"
        )

    #############################################
    # Global Identifiability

    def bidirected_components(self) -> list[MixedGraph]:
        comps: ig.VertexClustering = self.bidirected.components()
        # igraph returns internal indices, convert to external IDs
        return [
            self.induced_subgraph(self.to_external(cluster))
            for cluster in comps
            if len(cluster) > 1
        ]

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
                    # igraph.neighborhood returns internal indices
                    ancestors_internal = comp.directed.neighborhood(
                        vertices=s, order=comp.num_nodes, mode="in"
                    )
                    if len(ancestors_internal) > 1:
                        ancestors_external = comp.to_external(ancestors_internal)
                        ag = comp.induced_subgraph(ancestors_external)
                        comps.extend(ag.bidirected_components())

        return True

    def get_mixed_comp(self, sub_nodes: list[int], node: int) -> BiNodesResult:
        if node in sub_nodes:
            raise ValueError(
                f"Node {node} cannot be in sub_nodes for mixed component computation"
            )

        reachable = {node}
        avoid_set = set(sub_nodes)

        queue = [node]
        visited = {node}

        while queue:
            current = queue.pop(0)

            neighbors = self.bidirected.neighbors(current, mode="out")
            for neighbor in neighbors:
                if neighbor not in visited and neighbor not in avoid_set:
                    visited.add(neighbor)
                    reachable.add(neighbor)
                    queue.append(neighbor)

        parents = [parent for n in reachable for parent in self.parents(n)]
        incoming = (set(parents) - reachable) & avoid_set

        return BiNodesResult(parents, incoming)

    def plot(self, **kwargs):
        """
        Plot this mixed graph.

        This is a convenience method that calls plot_mixed_graph(self, **kwargs).
        See plot_mixed_graph for all available parameters.

        Returns:
            Tuple of (matplotlib Figure, matplotlib Axes)

        Examples:
            >>> graph = MixedGraph(L, O)
            >>> fig, ax = graph.plot(show=False)
            >>> fig, ax = graph.plot(layout="circle", node_color="lightblue")
        """
        from semid.visualization import plot_mixed_graph

        return plot_mixed_graph(self, **kwargs)
