from typing import Optional

import igraph as ig
import numpy as np
from numpy.typing import NDArray

from semid_py import LatentDigraph, utils


# NOTE: 0-indexed unlike in R
class MixedGraph:
    d_adj: NDArray[np.int32]
    """Directed adjacency matrix (aka L)"""

    b_adj: NDArray[np.int32]
    """Bidirected adjacency matrix (aka O)"""

    directed: ig.Graph
    bidirected: ig.Graph

    # TODO: init
    internal: LatentDigraph

    # TODO: named nodes?
    def __init__(
        self,
        d_adj: NDArray[np.int32] | list[int],
        b_adj: NDArray[np.int32] | list[int],
        nodes: Optional[int] = None,
    ):
        self.d_adj = utils.reshape_arr(d_adj, nodes)
        self.b_adj = utils.reshape_arr(b_adj, nodes)
        utils.validate_matrices(self.d_adj, self.b_adj)

        self.directed = ig.Graph.Adjacency(matrix=d_adj, mode="directed")
        self.bidirected = ig.Graph.Adjacency(matrix=b_adj, mode="undirected")

    @property
    def nodes(self) -> int:
        return self.d_adj.shape[0]

    def is_sibling(self, a: int, b: int) -> bool:
        return self.b_adj[a, b] != 0

    def induced_subgraph(self, nodes: list[int]) -> MixedGraph:
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
        return self.internal.get_trek_system(
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

    def bidirected_components(self) -> list[MixedGraph]:
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
