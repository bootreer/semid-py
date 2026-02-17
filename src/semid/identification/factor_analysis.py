"""Factor analysis sign-identifiability algorithms.

Implements the matching criterion from Sturma et al. for identifiability in sparse factor analysis models.

References
----------
Sturma, N., Kranzlmüller, M., Portakal, I., and Drton, M. (2025)
Matching Criterion for Identifiability in Sparse Factor Analysis.
arXiv:2502.02986
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import comb

import igraph as ig
import numpy as np
from numpy.typing import NDArray

from semid.latent_digraph import LatentDigraph


@dataclass
class ZUTAResult:
    """Result of the Zero Upper Triangular Assumption check."""

    zuta: bool
    latent_nodes: list[int]
    observed_nodes: list[int]


@dataclass
class MatchingResult:
    """Result of matching criterion check for one latent node."""

    found: bool
    h: int
    v: int | None = None
    W: list[int] | None = None
    U: list[int] | None = None


@dataclass
class LocalBBResult:
    """Result of local BB-criterion check."""

    found: bool
    new_nodes_in_S: list[int] | None = None
    U: list[int] | None = None


@dataclass
class MIDResult:
    """Result of M-identifiability check."""

    identifiable: bool
    tuple_list: list[dict] = field(default_factory=list)
    latent_nodes: list[int] = field(default_factory=list)
    observed_nodes: list[int] = field(default_factory=list)


@dataclass
class ExtMIDResult:
    """Result of extended M-identifiability check."""

    identifiable: bool
    tuple_list: list[dict] = field(default_factory=list)
    latent_nodes: list[int] = field(default_factory=list)
    observed_nodes: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _transform_lambda(
    lam: NDArray,
) -> tuple[NDArray, list[int], list[int]]:
    """Convert loading matrix to adjacency matrix + node lists.

    Lambda is (observed x latent). The adjacency matrix places latent nodes
    first (indices 0..k-1) and observed nodes next (indices k..k+p-1).
    adj[i, j] == 1 means i -> j.
    """
    p, k = lam.shape  # p observed, k latent
    n = k + p
    adj = np.zeros((n, n), dtype=int)
    adj[:k, k:] = lam.T  # latent -> observed
    latent_nodes = list(range(k))
    observed_nodes = list(range(k, n))
    return adj, latent_nodes, observed_nodes


def _transform_latent_digraph(
    graph: LatentDigraph,
) -> tuple[NDArray, list[int], list[int]]:
    """Extract adjacency matrix and node lists from a LatentDigraph."""
    return graph.adj.copy(), graph.latent_nodes(), graph.observed_nodes()


def _joint_parents(
    adj: NDArray, nodes: list[int], latent_nodes: list[int]
) -> list[int]:
    """Latent nodes that are parents of at least 2 nodes in `nodes`."""
    return [p for p in latent_nodes if adj[p, nodes].sum() >= 2]


def _children_of_nodes(
    adj: NDArray, nodes: list[int] | int, possible_children: list[int]
) -> list[int]:
    """Children of `nodes` that are in `possible_children`."""
    if isinstance(nodes, int):
        nodes = [nodes]
    return [c for c in possible_children if adj[nodes, c].any()]


def _parents_of_nodes(
    adj: NDArray, nodes: list[int], possible_parents: list[int]
) -> list[int]:
    """Parents of `nodes` that are in `possible_parents`."""
    return [p for p in possible_parents if adj[p, nodes].any()]


def _flow_graph_matrix(
    adj: NDArray, latent_nodes: list[int], observed_nodes: list[int]
) -> NDArray:
    """Build the base flow graph adjacency matrix.

    The flow graph has 2m + 2 nodes where m = adj.shape[0].
    Node indices: original 0..m-1, copies m..2m-1, s=2m, t=2m+1.
    """
    m = adj.shape[0]
    flow = np.zeros((2 * m + 2, 2 * m + 2), dtype=int)
    s, t = 2 * m, 2 * m + 1

    obs = np.array(observed_nodes)
    flow[s, obs] = 1            # s -> observed
    flow[m + obs, t] = 1        # copy of observed -> t

    lat = np.array(latent_nodes)
    flow[lat, m + lat] = 1      # latent -> copy of latent

    return flow


def _max_flow_st_graph(
    flow_graph: NDArray,
    adj: NDArray,
    latent_nodes: list[int],
    W: list[int],
    U: list[int],
) -> int:
    """Compute max s-t flow using igraph."""
    flow_adj = flow_graph.copy()
    m = adj.shape[0]

    # For each latent node y: if y -> x exists, add edge x -> y (for W)
    # and copy(y) -> copy(x) (for U) in the flow graph
    for y in latent_nodes:
        for x in W:
            if adj[y, x]:
                flow_adj[x, y] = 1
        for x in U:
            if adj[y, x]:
                flow_adj[m + y, m + x] = 1

    g = ig.Graph.Adjacency(flow_adj.tolist(), mode="directed")
    return g.maxflow_value(2 * m, 2 * m + 1, capacity=None)


def _matching_criterion(
    flow_graph_adj: NDArray,
    adj: NDArray,
    v: int,
    W: list[int],
    U: list[int],
    latent_nodes: list[int],
) -> bool:
    """Check the 4 conditions of the matching criterion for a given tuple."""
    # (i) v not in W or U
    if v in W or v in U:
        return False

    # (ii) W and U disjoint, same size, non-empty
    if not W or len(W) != len(U) or set(W) & set(U):
        return False

    # (iii) max flow for (W, U) equals |W|
    if _max_flow_st_graph(flow_graph_adj, adj, latent_nodes, W, U) != len(W):
        return False

    # (iv) max flow for (W ∪ {v}, U ∪ {v}) < |W| + 1
    if _max_flow_st_graph(flow_graph_adj, adj, latent_nodes, W + [v], U + [v]) == len(W) + 1:
        return False

    return True


def _power_set(elements: list[int], max_size: int) -> list[list[int]]:
    """Generate all subsets of *elements* up to *max_size*, including empty set."""
    result: list[list[int]] = [[]]
    for k in range(1, max_size + 1):
        for combo in combinations(elements, k):
            result.append(list(combo))
    return result


def _find_columns_with_sum_one(matrix: NDArray) -> bool:
    """Recursive helper for ZUTA: find column with sum 1, remove its row, recurse."""
    cols_with_one = np.where(matrix.sum(axis=0) == 1)[0]
    if not len(cols_with_one):
        return False
    if matrix.shape[0] <= 2:
        return True
    return any(
        _find_columns_with_sum_one(np.delete(matrix, matrix[:, col].argmax(), axis=0))
        for col in cols_with_one
    )


def _full_factor_criterion(
    adj: NDArray,
    U: list[int],
    joint_parents_U: list[int],
    latent_nodes: list[int],
    observed_nodes: list[int],
) -> bool:
    """Check whether the local BB sub-criteria are fulfilled for a tuple."""
    # criterion (i)
    induced = _adj_matrix_induced_subgraph(
        adj, U, joint_parents_U, latent_nodes, observed_nodes
    )
    zuta_result = _check_full_factor_zuta(induced, joint_parents_U, U)
    if not zuta_result["zuta"]:
        return False

    # criterion (ii)
    ordering: list[int] = zuta_result["ordering"]
    for h in joint_parents_U:
        set_of_v = [c for c in _children_of_nodes(adj, h, observed_nodes) if c not in U]
        set_of_l = ordering[: ordering.index(h) + 1]

        remaining_v = set(set_of_v)
        checked = list(U)
        while remaining_v:
            for v in list(remaining_v):
                if any(
                    all(p in set_of_l for p in _joint_parents(adj, [u, v], latent_nodes))
                    for u in checked
                ):
                    remaining_v.discard(v)
                    checked.append(v)
                    break
            else:
                return False
    return True


def _check_full_factor_zuta(
    adj: NDArray, latent_nodes: list[int], observed_nodes: list[int]
) -> dict:
    """Check full-factor ZUTA and return ordering if satisfied."""
    p = len(observed_nodes)
    m = len(latent_nodes)

    # count children per latent node and sort by decreasing count
    num_children = [adj[parent, observed_nodes].sum() for parent in latent_nodes]
    ordered_latent = sorted(latent_nodes, key=lambda n: num_children[latent_nodes.index(n)], reverse=True)
    sorted_counts = sorted(num_children, reverse=True)

    # check if node with most children has all children
    if sorted_counts[0] < p:
        return {"zuta": False}

    # check required counts (p-m+1)..p are all present, with no duplicates
    if set(sorted_counts) != set(range(p - m + 1, p + 1)):
        return {"zuta": False}

    # check consecutive pairs in the ordering
    for i in range(m - 1):
        node, next_node = ordered_latent[i], ordered_latent[i + 1]

        # observed children of node but not of next_node
        only_children = [
            obs for obs in observed_nodes
            if adj[node, obs] == 1 and adj[next_node, obs] == 0
        ]
        if len(only_children) != 1:
            return {"zuta": False}

        # check no other later latent node is also a parent of this child
        child = only_children[0]
        if any(adj[other, child] == 1 for other in ordered_latent[i + 2:]):
            return {"zuta": False}

    return {"zuta": True, "ordering": ordered_latent}


def _adj_matrix_induced_subgraph(
    adj: NDArray,
    U: list[int],
    joint_parents_U: list[int],
    latent_nodes: list[int],
    observed_nodes: list[int],
) -> NDArray:
    """Restrict adjacency matrix to the induced subgraph of U and joint parents."""
    result = adj.copy()
    non_joint = [lat for lat in latent_nodes if lat not in joint_parents_U]
    non_U_obs = [obs for obs in observed_nodes if obs not in U]
    result[np.ix_(non_joint, observed_nodes)] = 0
    result[np.ix_(joint_parents_U, non_U_obs)] = 0
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def zuta(lam: NDArray | LatentDigraph) -> ZUTAResult:
    """Check the Zero Upper Triangular Assumption (ZUTA).

    Parameters
    ----------
    lam : NDArray or LatentDigraph
        Binary loading matrix of shape (observed, latent), or a LatentDigraph.

    Returns
    -------
    ZUTAResult
        Whether ZUTA holds, plus the latent/observed node lists.
    """
    if isinstance(lam, LatentDigraph):
        adj, latent_nodes, observed_nodes = _transform_latent_digraph(lam)
    else:
        adj, latent_nodes, observed_nodes = _transform_lambda(lam)

    clean = adj[np.ix_(latent_nodes, observed_nodes)].copy()
    row_mask = clean.sum(axis=1) > 0
    col_mask = clean.sum(axis=0) > 0

    if row_mask.sum() == 1:
        return ZUTAResult(
            zuta=True, latent_nodes=latent_nodes, observed_nodes=observed_nodes
        )

    if col_mask.sum() == 1:
        is_zuta = bool(clean.sum() == 1)
        return ZUTAResult(
            zuta=is_zuta, latent_nodes=latent_nodes, observed_nodes=observed_nodes
        )

    clean = clean[np.ix_(row_mask, col_mask)]

    if clean.shape[0] <= 1:
        return ZUTAResult(
            zuta=True, latent_nodes=latent_nodes, observed_nodes=observed_nodes
        )

    is_zuta = _find_columns_with_sum_one(clean)
    return ZUTAResult(
        zuta=is_zuta, latent_nodes=latent_nodes, observed_nodes=observed_nodes
    )


def check_matching_criterion(
    flow_graph_adj: NDArray,
    adj: NDArray,
    h: int,
    latent_nodes: list[int],
    observed_nodes: list[int],
    max_card: int | None = None,
) -> MatchingResult:
    """Check the matching criterion for a single latent node *h*.

    Parameters
    ----------
    flow_graph_adj : NDArray
        Base flow graph adjacency matrix.
    adj : NDArray
        Graph adjacency matrix.
    h : int
        Latent node index.
    latent_nodes : list[int]
        All latent node indices.
    observed_nodes : list[int]
        All observed node indices.
    max_card : int or None
        Maximum cardinality for set W. Defaults to len(observed_nodes).

    Returns
    -------
    MatchingResult
    """
    if max_card is None:
        max_card = len(observed_nodes)

    for v in observed_nodes:
        # v must have h as its only latent parent
        v_parents = [node for node in latent_nodes if adj[node, v] == 1]
        if v_parents != [h]:
            continue

        obs_without_v = [o for o in observed_nodes if o != v]
        max_size_w = min(len(obs_without_v) // 2, len(latent_nodes), max_card)
        for W in _power_set(obs_without_v, max_size_w):
            if not W or not adj[h, W].any():
                continue
            obs_without_w = [o for o in obs_without_v if o not in W]
            possible_u = _children_of_nodes(
                adj, _parents_of_nodes(adj, W, latent_nodes), obs_without_w,
            )
            if len(possible_u) < len(W):
                continue
            for U_combo in combinations(possible_u, len(W)):
                U = list(U_combo)
                if not adj[h, U].any():
                    continue
                if _matching_criterion(flow_graph_adj, adj, v, W, U, latent_nodes):
                    return MatchingResult(found=True, h=h, v=v, W=W, U=U)

    return MatchingResult(found=False, h=h)


def check_local_bb_criterion(
    adj: NDArray, latent_nodes: list[int], observed_nodes: list[int]
) -> LocalBBResult:
    """Check the local BB-criterion.

    Parameters
    ----------
    adj : NDArray
        Graph adjacency matrix.
    latent_nodes : list[int]
        Latent node indices.
    observed_nodes : list[int]
        Observed node indices.

    Returns
    -------
    LocalBBResult
    """
    for h in latent_nodes:
        children_h = _children_of_nodes(adj, h, observed_nodes)
        for U in _power_set(children_h, len(children_h)):
            if len(U) <= 2:
                continue
            jp_U = _joint_parents(adj, U, latent_nodes)
            p = len(U)
            m = len(jp_U)
            # cardinality inequality
            if p * (m + 1) - comb(m, 2) >= comb(p + 1, 2):
                continue
            if _full_factor_criterion(adj, U, jp_U, latent_nodes, observed_nodes):
                return LocalBBResult(found=True, new_nodes_in_S=jp_U, U=U)
    return LocalBBResult(found=False)


def m_id(lam: NDArray | LatentDigraph, max_card: int | None = None) -> MIDResult:
    """Check M-identifiability via the matching criterion.

    Parameters
    ----------
    lam : NDArray or LatentDigraph
        Binary loading matrix of shape (observed, latent), or a LatentDigraph.
    max_card : int or None
        Maximum cardinality for set W. Defaults to len(observed_nodes).

    Returns
    -------
    MIDResult
    """
    if isinstance(lam, LatentDigraph):
        adj, latent_nodes_orig, observed_nodes = _transform_latent_digraph(lam)
    else:
        adj, latent_nodes_orig, observed_nodes = _transform_lambda(lam)

    if max_card is None:
        max_card = len(observed_nodes)

    S: list[int] = []

    # childless latent nodes are trivially identifiable
    latent_nodes = list(latent_nodes_orig)
    for lat in latent_nodes_orig:
        has_children = any(adj[lat, obs] == 1 for obs in observed_nodes)
        if not has_children:
            S.append(lat)

    latent_nodes = [n for n in latent_nodes if n not in S]
    not_identified = list(latent_nodes)

    flow_graph_adj = _flow_graph_matrix(adj, latent_nodes, observed_nodes)
    tuple_list: list[dict] = []

    while latent_nodes:
        found = False
        for h in list(not_identified):
            result = check_matching_criterion(
                flow_graph_adj, adj, h, latent_nodes, observed_nodes, max_card
            )
            if result.found:
                found = True
                latent_nodes = [n for n in latent_nodes if n != h]
                not_identified = [n for n in not_identified if n != h]
                tuple_list.append(
                    {
                        "h": result.h,
                        "S": list(S),
                        "v": result.v,
                        "W": result.W,
                        "U": result.U,
                    }
                )
                S.append(h)

        if not found:
            return MIDResult(
                identifiable=False,
                tuple_list=tuple_list,
                latent_nodes=latent_nodes_orig,
                observed_nodes=observed_nodes,
            )

    return MIDResult(
        identifiable=True,
        tuple_list=tuple_list,
        latent_nodes=latent_nodes_orig,
        observed_nodes=observed_nodes,
    )


def ext_m_id(lam: NDArray | LatentDigraph, max_card: int | None = None) -> ExtMIDResult:
    """Check extended M-identifiability (local BB + matching criterion).

    Parameters
    ----------
    lam : NDArray or LatentDigraph
        Binary loading matrix of shape (observed, latent), or a LatentDigraph.
    max_card : int or None
        Maximum cardinality for set W. Defaults to len(observed_nodes).

    Returns
    -------
    ExtMIDResult
    """
    if isinstance(lam, LatentDigraph):
        adj, latent_nodes_orig, observed_nodes_orig = _transform_latent_digraph(lam)
    else:
        adj, latent_nodes_orig, observed_nodes_orig = _transform_lambda(lam)

    observed_nodes = list(observed_nodes_orig)
    if max_card is None:
        max_card = len(observed_nodes)

    S: list[int] = []

    # childless latent nodes are trivially identifiable
    latent_nodes = list(latent_nodes_orig)
    for lat in list(latent_nodes):
        has_children = any(adj[lat, obs] == 1 for obs in observed_nodes)
        if not has_children:
            S.append(lat)

    # zero out rows for childless latent nodes
    adj[S, :] = 0

    # remove observed nodes without parents
    col_sums = adj.sum(axis=0)
    obs_without_parents = [o for o in observed_nodes if col_sums[o] == 0]
    observed_nodes = [o for o in observed_nodes if o not in obs_without_parents]

    latent_nodes = [n for n in latent_nodes if n not in S]
    not_identified = list(latent_nodes)
    flow_graph_adj = _flow_graph_matrix(adj, latent_nodes, observed_nodes)
    tuple_list: list[dict] = []

    while latent_nodes:
        # first try local BB criterion
        bb_result = check_local_bb_criterion(adj, latent_nodes, observed_nodes)
        if bb_result.found:
            assert bb_result.new_nodes_in_S is not None

            tuple_list.append(
                {
                    "criterion": "localBB",
                    "S": list(S),
                    "new_nodes_in_S": bb_result.new_nodes_in_S,
                    "U": bb_result.U,
                }
            )
            latent_nodes = [
                n for n in latent_nodes if n not in bb_result.new_nodes_in_S
            ]
            S.extend(bb_result.new_nodes_in_S)

            adj[bb_result.new_nodes_in_S, :] = 0

            col_sums = adj.sum(axis=0)
            obs_without_parents = [o for o in observed_nodes if col_sums[o] == 0]
            observed_nodes = [o for o in observed_nodes if o not in obs_without_parents]
            continue

        # try matching criterion
        found_any = False
        for h in list(not_identified):
            mc_result = check_matching_criterion(
                flow_graph_adj, adj, h, latent_nodes, observed_nodes, max_card
            )
            if mc_result.found:
                found_any = True
                latent_nodes = [n for n in latent_nodes if n != h]
                not_identified = [n for n in not_identified if n != h]
                tuple_list.append(
                    {
                        "criterion": "matching",
                        "h": mc_result.h,
                        "S": list(S),
                        "v": mc_result.v,
                        "W": mc_result.W,
                        "U": mc_result.U,
                    }
                )
                S.append(h)

                adj[h, :] = 0
                col_sums = adj.sum(axis=0)
                obs_without_parents = [
                    o for o in observed_nodes if col_sums[o] == 0
                ]
                observed_nodes = [
                    o for o in observed_nodes if o not in obs_without_parents
                ]

        if not found_any:
            return ExtMIDResult(
                identifiable=False,
                tuple_list=tuple_list,
                latent_nodes=latent_nodes_orig,
                observed_nodes=observed_nodes_orig,
            )

    return ExtMIDResult(
        identifiable=True,
        tuple_list=tuple_list,
        latent_nodes=latent_nodes_orig,
        observed_nodes=observed_nodes_orig,
    )
