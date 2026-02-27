"""
Functions for identifying the causal effect vector in a causal graph.

Implements identifiability criteria based on vertex-disjoint max-flow
in the directed part of the graph.
"""

from __future__ import annotations

import igraph as ig
import numpy as np

from semid.mixed_graph import MixedGraph


def _removable_ancestors(g: MixedGraph, v: int) -> list[int]:
    """Return ancestors of v whose sibling set is not a subset of sib(v) ∪ {v}."""
    sv = set(g.siblings(v)) | {v}
    return [u for u in g.ancestors(v) if u != v and (set(g.siblings(u)) | {u}) - sv]


def _build_flow_graph(
    adj: np.ndarray,
    ancestors: list[int],
    rv: list[int],
    v: int,
    q_edges: set[int],
    p: int,
) -> tuple[ig.Graph, int, int]:
    """
    Build an igraph flow graph with vertex splitting for vertex-disjoint paths.

    The graph is restricted to ancestors of v plus a super-source node.
    Only edges from q_edges are allowed to enter v. Vertex splitting
    enforces capacity 1 per internal node, with a higher capacity for
    the super-source and sink so they never bottleneck the flow.

    Args:
        adj: Directed adjacency matrix of the full graph.
        ancestors: Ancestor nodes of v (not including v itself).
        rv: Removable ancestors of v.
        q_edges: Set of nodes whose edges into v should be kept.
        v: Target node.
        p: Total number of nodes in the original graph.

    Returns:
        (flow_graph, source_idx, sink_idx)
    """
    # Restrict incoming edges of v to q_edges only
    adj = adj.copy()
    new_col = np.zeros(p, dtype=int)
    new_col[list(q_edges)] = 1
    adj[:, v] = new_col

    # Super-source row: connects to all removable ancestors
    super_src_row = np.zeros(p, dtype=int)
    super_src_row[list(rv)] = 1

    # Prepend super-source as node 0 (inserts a new row/column)
    a = np.insert(adj, 0, super_src_row, axis=0)
    a = np.insert(a, 0, np.zeros(p + 1, dtype=int), axis=1)

    # Shift node indices by 1 to account for the super-source at index 0
    av_shifted = [x + 1 for x in ancestors]
    v_shifted = v + 1

    # Flow subgraph: super-source (0), ancestors of v, and v itself
    mf_vert = [0] + av_shifted + [v_shifted]

    # Scale edge capacities so inter-node edges never bottleneck the flow
    # (node capacities are 1; edge capacities must be >= max possible flow)
    high_cap = len(q_edges) + 1
    a = a[np.ix_(mf_vert, mf_vert)] * high_cap

    n_sub = len(mf_vert)

    # Vertex splitting: node i is split into i_out (= i) and i_in (= i + n_sub).
    #   Original edge i -> j  becomes:  i_out -> j_in  (capacity from a)
    #   Internal capacity edge:         i_in  -> i_out (capacity = node capacity)
    aug = np.zeros((2 * n_sub, 2 * n_sub), dtype=int)

    rows, cols = np.nonzero(a)
    aug[rows, cols + n_sub] = a[rows, cols]

    # Node capacities: 1 for all internal nodes, high_cap for super-source and v
    mf_cap = np.ones(n_sub, dtype=int)
    mf_cap[0] = high_cap  # super-source
    mf_cap[n_sub - 1] = high_cap  # v (sink)
    aug[np.arange(n_sub) + n_sub, np.arange(n_sub)] = mf_cap

    # Build igraph from the augmented matrix
    edge_rows, edge_cols = np.nonzero(aug)
    edges = list(zip(edge_rows.tolist(), edge_cols.tolist()))
    capacities = aug[edge_rows, edge_cols].tolist()

    flow_graph = ig.Graph(n=2 * n_sub, edges=edges, directed=True)
    flow_graph.es["capacity"] = capacities

    # source = out-node of super-source (index 0)
    # sink   = in-node of v (index n_sub - 1 + n_sub = 2*n_sub - 1)
    return flow_graph, 0, 2 * n_sub - 1


def _maxflow_q(g: MixedGraph, v: int, q: set[int]) -> int:
    """
    Compute vertex-disjoint max-flow from removable ancestors to v,
    with only edges from q allowed to enter v.

    Args:
        g: The mixed graph.
        v: Target node.
        q: Set of nodes whose edges into v are kept.

    Returns:
        Max-flow value.
    """
    if not q:
        return 0

    rv = _removable_ancestors(g, v)
    ancestors = [u for u in g.ancestors(v) if u != v]

    flow_graph, source, sink = _build_flow_graph(
        np.array(g.d_adj, dtype=int), ancestors, rv, v, q, g.num_nodes
    )
    return int(
        flow_graph.maxflow(source, sink, capacity=flow_graph.es["capacity"]).value
    )


def check_criterion(g: MixedGraph, v: int, q: set[int]) -> bool:
    """
    Check identifiability of the causal effect vector from node set q to node v.

    The causal effect is identifiable if the vertex-disjoint max-flow satisfies:
        maxflow(q) - maxflow(pa(v) \\ q) == |q ∩ pa(v)|

    Args:
        g: A MixedGraph representing the causal model.
        v: Target node.
        q: Set of source nodes to check.

    Returns:
        True if the causal effect from q to v is identifiable.
    """
    pav = set(g.parents(v))
    q = q & pav
    if not q:
        return True

    return (_maxflow_q(g, v, q) - _maxflow_q(g, v, pav - q)) == len(q)


def wholematrix_criterion(g: MixedGraph) -> bool:
    """
    Check identifiability of the entire causal effect matrix.

    Iterates over all nodes and checks whether the full parent set
    satisfies the criterion for each node.

    Args:
        g: A MixedGraph representing the causal model.

    Returns:
        True if all causal effects in the model are identifiable.
    """
    return all(check_criterion(g, v, set(g.parents(v))) for v in range(g.num_nodes))
