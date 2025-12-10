from typing import Optional

import igraph as ig
import numpy as np
from numpy.typing import NDArray

from semid_py import utils


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

        assert 0 <= num_observed <= self.adj.shape[0], (
            "num_observed must be between 0 and number of nodes"
        )

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
                self.digraph.neighborhood(vertices=node, order=self.num_nodes, mode="in")
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
                self.digraph.neighborhood(vertices=node, order=self.num_nodes, mode="out")
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
            `from_nodes`: Source nodes for treks (left side)
            `to_nodes`: Target nodes for treks (right side)
            `avoid_left_nodes`: Nodes that cannot appear on left (backward) side
            `avoid_right_nodes`: Nodes that cannot appear on right (forward) side
            `avoid_left_edges`: Edges (i,j) that cannot be traversed backward (j->i)
            `avoid_right_edges`: Edges (i,j) that cannot be traversed forward (i->j)

        Returns:
            dict with:
                - `system_exists`: True if a trek system of size len(to_nodes) exists
                - `active_from`: Subset of from_nodes used in the maximal trek system
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

        return {
            "system_exists": flow_result.value == len(to_nodes),
            "active_from": active_from,
        }

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
