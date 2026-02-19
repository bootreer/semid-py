from __future__ import annotations

from typing import Literal

import igraph as ig
import numpy as np
from numpy.typing import NDArray

from semid import utils
from semid.utils import TrekSystem
from collections import deque


class LatentDigraph:
    adj: NDArray[np.int32]
    """Adjacency matrix L"""

    digraph: ig.Graph

    num_observed: int
    """Number of observed nodes (first rows of adj matrix)"""

    num_latents: int
    """Number of latent nodes (remaining rows of adj matrix)"""

    _tr_graph: ig.Graph | None
    """Cached trek graph"""

    def __init__(
        self,
        adj: NDArray[np.int32] | list[int],
        num_observed: int | None = None,
        nodes: int | None = None,
    ):
        """
        Create a LatentDigraph object.

        Args:
            adj: Adjacency matrix where first num_observed rows are observed nodes
            num_observed: Number of observed nodes. If None, all nodes are observed.
        """
        self.adj = utils.reshape_arr(adj, nodes)
        utils.validate_matrix(self.adj)
        self.digraph = ig.Graph.Adjacency(matrix=self.adj, mode="directed")

        if num_observed is None:
            num_observed = self.adj.shape[0]

        if not (0 <= num_observed <= self.adj.shape[0]):
            raise ValueError("num_observed must be between 0 and number of nodes")

        self.num_observed = num_observed
        self.num_latents = self.adj.shape[0] - num_observed
        self._tr_graph = None

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
        nodes: int | list[int],
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
        if isinstance(nodes, int):
            nodes = [nodes]

        res = set()
        for n in nodes:
            res.update(self.digraph.predecessors(n))

        # Filter based on node index ranges
        if not include_observed and not include_latents:
            return []
        if not include_observed:
            res = {n for n in res if n >= self.num_observed}
        elif not include_latents:
            res = {n for n in res if n < self.num_observed}

        return sorted(res)

    def children(
        self,
        nodes: int | list[int],
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
        if isinstance(nodes, int):
            nodes = [nodes]

        res = set()
        for n in nodes:
            res.update(self.digraph.successors(n))

        if not include_observed and not include_latents:
            return []
        if not include_observed:
            res = {n for n in res if n >= self.num_observed}
        elif not include_latents:
            res = {n for n in res if n < self.num_observed}

        return sorted(res)

    def ancestors(
        self,
        nodes: int | list[int],
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
        if isinstance(nodes, int):
            nodes = [nodes]

        # neighborhood with order=num_nodes is essentially a reachability query
        reachable = self.digraph.neighborhood(
            vertices=nodes, order=self.num_nodes, mode="in"
        )
        res = set().union(*reachable)

        if not include_observed and not include_latents:
            return []
        if not include_observed:
            return sorted(n for n in res if n >= self.num_observed)
        if not include_latents:
            return sorted(n for n in res if n < self.num_observed)
        return sorted(res)

    def descendants(
        self,
        nodes: int | list[int],
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
        if isinstance(nodes, int):
            nodes = [nodes]

        reachable = self.digraph.neighborhood(
            vertices=nodes, order=self.num_nodes, mode="out"
        )
        res = set().union(*reachable)

        if not include_observed and not include_latents:
            return []
        if not include_observed:
            return sorted(n for n in res if n >= self.num_observed)
        if not include_latents:
            return sorted(n for n in res if n < self.num_observed)
        return sorted(res)

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

        return ig.Graph.Adjacency(adj_mat, mode="directed")

    def tr_from(
        self,
        nodes: list[int],
        avoid_left_nodes: list[int] | None = None,
        avoid_right_nodes: list[int] | None = None,
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
        if avoid_left_nodes is None:
            avoid_left_nodes = []
        if avoid_right_nodes is None:
            avoid_right_nodes = []

        if self._tr_graph is None:
            self._tr_graph = self._create_tr_graph()

        m = self.num_nodes

        # Build set of avoided nodes in trek graph (2m nodes total)
        avoid_set = set(avoid_left_nodes) | set(n + m for n in avoid_right_nodes)

        # Use a single BFS traversal for efficiency
        # Filter valid start nodes that are not in the avoid set
        valid_starts = [n for n in nodes if n not in avoid_set]

        queue: deque[int] = deque(valid_starts)
        visited: set[int] = set(valid_starts)

        while queue:
            current = queue.popleft()

            # Get outgoing neighbors
            neighbors = self._tr_graph.neighbors(current, mode="out")
            for neighbor in neighbors:
                if neighbor not in visited and neighbor not in avoid_set:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Convert back from trek graph node indices to original graph indices
        # Nodes 0..m-1 map to themselves, nodes m..2m-1 map to 0..m-1
        original_nodes = set()
        for tr_node in visited:
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
        avoid_left_nodes: list[int] | None = None,
        avoid_right_nodes: list[int] | None = None,
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
        if avoid_left_nodes is None:
            avoid_left_nodes = []
        if avoid_right_nodes is None:
            avoid_right_nodes = []

        # Half-trek: avoid all nodes on the left except the starting nodes
        all_nodes = list(range(self.num_nodes))
        nodes_set = set(nodes)
        additional_avoid = [n for n in all_nodes if n not in nodes_set]
        combined_avoid_left = list(set(avoid_left_nodes) | set(additional_avoid))

        return self.tr_from(
            nodes=nodes,
            avoid_left_nodes=combined_avoid_left,
            avoid_right_nodes=avoid_right_nodes,
            include_observed=include_observed,
            include_latents=include_latents,
        )

    def _build_flow_graph(
        self,
        trek_edges: list[tuple[int, int]] | set[tuple[int, int]],
        from_nodes: list[int],
        to_nodes: list[int],
        avoid_left_nodes: list[int],
        avoid_right_nodes: list[int],
        trek_avoid_edges: set[tuple[int, int]],
    ) -> tuple[ig.Graph, int, int]:
        """
        Build a sparse flow graph for computing trek systems via max-flow.

        The flow graph encodes vertex-disjoint trek paths using vertex splitting.
        Each trek graph vertex is split into "in" and "out" parts with a single
        edge between them to enforce vertex disjointness.

        Node layout (N = 2*M + 2 total):
          - 0..M-1     : in-parts of trek graph vertices
          - M..2*M-1   : out-parts of trek graph vertices
          - SOURCE=2*M, SINK=2*M+1

        Args:
            trek_edges: Trek graph edges to include (src, dst)
            from_nodes: Source nodes connected to SOURCE
            to_nodes: Target nodes connected to SINK
            avoid_left_nodes: Left-copy nodes to block (vertex capacity = 0)
            avoid_right_nodes: Right-copy nodes to block (vertex capacity = 0)
            trek_avoid_edges: Trek graph edges to exclude

        Returns:
            Tuple of (flow_graph, SOURCE, SINK)
        """
        m = self.num_nodes
        M = 2 * m
        N = 2 * M + 2
        SOURCE = 2 * M
        SINK = 2 * M + 1

        avoid_left_set = set(avoid_left_nodes)
        avoid_right_set = set(avoid_right_nodes)

        edges = []

        # Vertex-splitting: in-part i -> out-part M+i (capacity 1 each)
        for i in range(M):
            if i < m and i in avoid_left_set:
                continue
            if i >= m and (i - m) in avoid_right_set:
                continue
            edges.append((i, M + i))

        # Trek edges: out-part of src -> in-part of dst
        for src, dst in trek_edges:
            if (src, dst) not in trek_avoid_edges:
                edges.append((M + src, dst))

        # Source -> left in-parts of from_nodes
        for node in from_nodes:
            edges.append((SOURCE, node))

        # Right out-parts of to_nodes -> sink
        for node in to_nodes:
            edges.append((M + m + node, SINK))

        return ig.Graph(n=N, edges=edges, directed=True), SOURCE, SINK

    def _run_maxflow(
        self,
        flow_graph: ig.Graph,
        from_nodes: list[int],
        to_nodes: list[int],
        SOURCE: int,
        SINK: int,
    ) -> TrekSystem:
        """Run max-flow and extract active_from nodes."""
        flow_result = flow_graph.maxflow(SOURCE, SINK)

        active_from = []
        for node in from_nodes:
            edge_id = flow_graph.get_eid(SOURCE, node, directed=True, error=False)
            if edge_id != -1 and flow_result.flow[edge_id] > 0:
                active_from.append(node)

        return TrekSystem(
            system_exists=flow_result.value == len(to_nodes),
            active_from=active_from,
        )

    def get_trek_system(
        self,
        from_nodes: list[int],
        to_nodes: list[int],
        avoid_left_nodes: list[int] | None = None,
        avoid_right_nodes: list[int] | None = None,
        avoid_left_edges: list[tuple[int, int]] | None = None,
        avoid_right_edges: list[tuple[int, int]] | None = None,
    ) -> TrekSystem:
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
            TrekSystem with system_exists and active_from fields
        """
        if avoid_left_nodes is None:
            avoid_left_nodes = []
        if avoid_right_nodes is None:
            avoid_right_nodes = []
        if avoid_left_edges is None:
            avoid_left_edges = []
        if avoid_right_edges is None:
            avoid_right_edges = []

        if self._tr_graph is None:
            self._tr_graph = self._create_tr_graph()

        m = self.num_nodes

        # Build avoided edges in trek graph coordinates
        trek_avoid_edges: set[tuple[int, int]] = set()
        for i, j in avoid_left_edges:
            trek_avoid_edges.add((j, i))
        for i, j in avoid_right_edges:
            trek_avoid_edges.add((m + i, m + j))

        flow_graph, SOURCE, SINK = self._build_flow_graph(
            trek_edges=self._tr_graph.get_edgelist(),
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            avoid_left_nodes=avoid_left_nodes,
            avoid_right_nodes=avoid_right_nodes,
            trek_avoid_edges=trek_avoid_edges,
        )

        return self._run_maxflow(flow_graph, from_nodes, to_nodes, SOURCE, SINK)

    def _bfs_reachable_trek_edges(
        self,
        from_nodes: list[int],
        max_depth: int,
        avoid_set: set[int],
        avoid_edges: set[tuple[int, int]] | None = None,
    ) -> set[tuple[int, int]]:
        """
        BFS on the trek graph to find edges reachable within max_depth steps.

        Args:
            from_nodes: Start nodes (original graph indices, mapped to left copies)
            max_depth: Maximum BFS depth in the trek graph
            avoid_set: Set of trek graph node indices to avoid
            avoid_edges: Set of (src, dst) trek graph edges to avoid

        Returns:
            Set of (src, dst) edges in the trek graph that are reachable
        """
        if self._tr_graph is None:
            self._tr_graph = self._create_tr_graph()

        if avoid_edges is None:
            avoid_edges = set()

        # Start BFS from left copies of from_nodes
        valid_starts = [n for n in from_nodes if n not in avoid_set]
        queue: deque[tuple[int, int]] = deque([(n, 0) for n in valid_starts])
        visited: set[int] = set(valid_starts)
        reachable_edges: set[tuple[int, int]] = set()

        while queue:
            current, depth = queue.popleft()

            if depth >= max_depth:
                continue

            neighbors = self._tr_graph.neighbors(current, mode="out")
            for neighbor in neighbors:
                if neighbor in avoid_set:
                    continue
                if (current, neighbor) in avoid_edges:
                    continue
                reachable_edges.add((current, neighbor))
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return reachable_edges

    def get_trek_system_local(
        self,
        from_nodes: list[int],
        to_nodes: list[int],
        max_hops: int,
        avoid_left_nodes: list[int] | None = None,
        avoid_right_nodes: list[int] | None = None,
        avoid_left_edges: list[tuple[int, int]] | None = None,
        avoid_right_edges: list[tuple[int, int]] | None = None,
    ) -> TrekSystem:
        """
        Depth-limited trek system using max-flow on a truncated flow graph.

        Like get_trek_system, but only considers treks of bounded length.
        max_hops defines the number of edges in the original graph sense:
        max_hops=1 means direct children/siblings only.

        Internally, max_hops is converted to trek graph depth as max_hops + 1
        (accounting for the bridge edge from left to right copy).

        Args:
            from_nodes: Source nodes for treks (left side)
            to_nodes: Target nodes for treks (right side)
            max_hops: Maximum number of hops in the original graph.
                      1 = direct children/siblings, 2 = two edges, etc.
            avoid_left_nodes: Nodes that cannot appear on left (backward) side
            avoid_right_nodes: Nodes that cannot appear on right (forward) side
            avoid_left_edges: Edges (i,j) that cannot be traversed backward (j->i)
            avoid_right_edges: Edges (i,j) that cannot be traversed forward (i->j)

        Returns:
            TrekSystem with system_exists and active_from fields
        """
        if avoid_left_nodes is None:
            avoid_left_nodes = []
        if avoid_right_nodes is None:
            avoid_right_nodes = []
        if avoid_left_edges is None:
            avoid_left_edges = []
        if avoid_right_edges is None:
            avoid_right_edges = []

        m = self.num_nodes

        # Convert max_hops to trek graph depth: bridge + forward edges
        max_depth = max_hops + 1

        # Build avoid set for BFS
        avoid_set = set(avoid_left_nodes) | set(n + m for n in avoid_right_nodes)

        # Build avoided edges in trek graph coordinates
        trek_avoid_edges: set[tuple[int, int]] = set()
        for i, j in avoid_left_edges:
            trek_avoid_edges.add((j, i))
        for i, j in avoid_right_edges:
            trek_avoid_edges.add((m + i, m + j))

        # BFS to find reachable edges within max_depth
        reachable_edges = self._bfs_reachable_trek_edges(
            from_nodes, max_depth, avoid_set, trek_avoid_edges
        )

        flow_graph, SOURCE, SINK = self._build_flow_graph(
            trek_edges=reachable_edges,
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            avoid_left_nodes=avoid_left_nodes,
            avoid_right_nodes=avoid_right_nodes,
            trek_avoid_edges=trek_avoid_edges,
        )

        return self._run_maxflow(flow_graph, from_nodes, to_nodes, SOURCE, SINK)

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

    def plot(
        self,
        layout: Literal["auto", "circle", "fr", "kk", "grid", "tree"] = "auto",
        node_labels: list[str] | None = None,
        **kwargs,
    ):
        """
        Plot this latent digraph using matplotlib.

        Args:
            layout: Layout algorithm - "auto", "circle", "fr" (Fruchterman-Reingold),
                "kk" (Kamada-Kawai), "grid", or "tree"
            node_labels: Custom node labels. If None, uses numeric indices
            **kwargs: Additional arguments passed to plot_latent_digraph()

        Returns:
            Tuple of (matplotlib figure, matplotlib axes)
        """
        from .visualization import plot_latent_digraph

        # pyrefly: ignore
        return plot_latent_digraph(
            self, layout=layout, node_labels=node_labels, **kwargs
        )
