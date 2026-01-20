"""Latent Subgraph Criterion (LSC)"""

import igraph as ig
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog

from semid.latent_digraph import LatentDigraph

from .lfhtc import subsets_of_size


class LatentSubgraph:
    cov: LatentDigraph
    semidirect: LatentDigraph
    lp_graph: ig.Graph
    lp_sub_graph: ig.Graph

    def __init__(self, g: LatentDigraph) -> None:
        """
        Initialize latent subgraph structures for LSC algorithm.

        Args:
            `g`: The latent digraph to analyze
        """
        self.cov = LatentSubgraph._latent_cov_graph(g)
        self.semidirect = LatentSubgraph._semidirect_effect_graph(g)
        self.lp_graph, self.lp_sub_graph = LatentSubgraph._get_lp_graphs(g)

    @staticmethod
    def _semidirect_effect_graph(g: LatentDigraph) -> LatentDigraph:
        n_obs = g.num_observed
        n_lat = g.num_latents
        n_tot = g.num_nodes

        L = g.adj

        # Compute semi-direct effects: L_obs + L_obs_lat * (I - L_lat)^-1 * L_lat_obs
        semidirect_adj: NDArray = L[:n_obs, :n_obs] + L[
            :n_obs, n_obs:n_tot
        ] @ np.linalg.solve(
            np.eye(n_lat) - L[n_obs:n_tot, n_obs:n_tot],
            L[n_obs:n_tot, :n_obs],
        )
        semidirect_adj = (semidirect_adj > 0).astype(np.int32)

        return LatentDigraph(semidirect_adj, num_observed=n_obs)

    @staticmethod
    def _latent_cov_graph(g: LatentDigraph) -> LatentDigraph:
        observed_nodes = g.observed_nodes()

        latent_L = g.adj.copy()
        latent_L[observed_nodes, :] = 0

        return LatentDigraph(latent_L, num_observed=g.num_observed)

    @staticmethod
    def _get_lp_graphs(g: LatentDigraph) -> tuple[ig.Graph, ig.Graph]:
        observed_nodes = g.observed_nodes()
        n_obs = g.num_observed
        n_lat = g.num_latents
        m = n_obs + n_lat

        adj_mat = np.zeros((2 * m, 2 * m), dtype=np.int32)
        adj_mat_sub = np.zeros((2 * m, 2 * m), dtype=np.int32)

        latent_L = g.adj.copy()
        latent_L[observed_nodes, :] = 0

        adj_mat[:m, :m] = latent_L.T
        adj_mat_sub[:m, :m] = latent_L.T

        for i in range(m):
            adj_mat[i, m + i] = 1
            adj_mat_sub[i, m + i] = 1

        adj_mat[m : 2 * m, m : 2 * m] = g.adj
        adj_mat_sub[m : 2 * m, m : 2 * m] = latent_L

        g_lp = ig.Graph.Adjacency(adj_mat.tolist(), mode="directed")
        g_lp_sub = ig.Graph.Adjacency(adj_mat_sub.tolist(), mode="directed")

        return g_lp, g_lp_sub


def _is_subgraph(g: ig.Graph, g_sub: ig.Graph) -> bool:
    adj = np.array(g.get_adjacency().data)
    adj_sub = np.array(g_sub.get_adjacency().data)

    if adj.shape != adj_sub.shape:
        return False

    return bool(np.all((adj - adj_sub) >= 0))


def _get_incidence_matrix(g: ig.Graph) -> NDArray[np.float64]:
    """
    Get the incidence matrix of a graph.

    The incidence matrix B has B[v,e] = -1 if edge e leaves vertex v,
    B[v,e] = 1 if edge e enters vertex v, and 0 otherwise.

    Args:
        `g`: Input graph

    Returns:
        Incidence matrix of shape (num_vertices, num_edges)
    """
    edges = g.get_edgelist()
    num_vertices = g.vcount()
    num_edges = g.ecount()

    inc_mat = np.zeros((num_vertices, num_edges), dtype=np.float64)

    for e, (tail, head) in enumerate(edges):
        inc_mat[tail, e] = -1  # Outgoing edge
        inc_mat[head, e] = 1  # Incoming edge

    return inc_mat


def _get_incoming_matrix(g: ig.Graph) -> NDArray[np.float64]:
    """
    Get the incoming edge matrix of a graph.

    The incoming matrix M has M[v,e] = 1 if edge e enters vertex v, 0 otherwise.

    Args:
        `g`: Input graph

    Returns:
        Incoming matrix of shape (num_vertices, num_edges)
    """
    edges = g.get_edgelist()
    num_vertices = g.vcount()
    num_edges = g.ecount()

    incoming_mat = np.zeros((num_vertices, num_edges), dtype=np.float64)

    for e, (tail, head) in enumerate(edges):
        incoming_mat[head, e] = 1

    return incoming_mat


def joint_flow(
    g: ig.Graph, g_sub: ig.Graph, s: int, t: int, integer: bool = False
) -> dict:
    """
    Solve the joint flow linear program.

    This LP finds a maximum flow from s to t where flows are constrained
    to respect both graph g and its subgraph g_sub.

    Args:
        `g`: Main flow graph
        `g_sub`: Subgraph of g
        `s`: Source vertex
        `t`: Sink vertex
        `integer`: If True, require integer solution (default False)

    Returns:
        Dictionary with 'objval' (maximum flow), 'solution' (edge flow values),
        and 'success' (whether optimization succeeded)
    """
    if not _is_subgraph(g, g_sub):
        raise ValueError("g_sub is not a subgraph of g")

    num_vertices = g.vcount()
    num_edges_g = g.ecount()
    num_edges_g_sub = g_sub.ecount()
    edges_g = g.get_edgelist()
    edges_g_sub = g_sub.get_edgelist()

    # Objective: maximize flow into sink t
    c = np.zeros(num_edges_g + num_edges_g_sub)

    # Create dict for O(1) edge lookup in subgraph
    edges_g_sub_dict = {edge: idx for idx, edge in enumerate(edges_g_sub)}

    for e, (tail, head) in enumerate(edges_g):
        if head == t:
            # Check if this edge is also in subgraph
            if (tail, head) in edges_g_sub_dict:
                idx_sub = edges_g_sub_dict[(tail, head)]
                c[num_edges_g + idx_sub] = 1
            else:
                c[e] = 1

    # We want to maximize, so negate for scipy's minimize
    c = -c

    # Equality constraints: flow conservation at all nodes except s and t
    inc_g = _get_incidence_matrix(g)
    inc_g_sub = _get_incidence_matrix(g_sub)

    # Remove rows for s and t
    vertices_interior = [v for v in range(num_vertices) if v != s and v != t]
    num_eq = 2 * len(vertices_interior)

    A_eq = np.zeros((num_eq, num_edges_g + num_edges_g_sub))
    A_eq[: len(vertices_interior), :num_edges_g] = inc_g[vertices_interior, :]
    A_eq[len(vertices_interior) :, num_edges_g:] = inc_g_sub[vertices_interior, :]

    b_eq = np.zeros(num_eq)

    # Inequality constraints: at most 1 unit of flow into each interior vertex
    incoming_g = _get_incoming_matrix(g)
    incoming_g_sub = _get_incoming_matrix(g_sub)

    A_ub = np.zeros((len(vertices_interior), num_edges_g + num_edges_g_sub))
    A_ub[:, :num_edges_g] = incoming_g[vertices_interior, :]
    A_ub[:, num_edges_g:] = incoming_g_sub[vertices_interior, :]

    b_ub = np.ones(len(vertices_interior))

    # Bounds: all flows non-negative
    bounds = [(0, None) for _ in range(num_edges_g + num_edges_g_sub)]

    # 0 = continuous, 1 = integer
    integrality = None
    if integer:
        integrality = np.ones(num_edges_g + num_edges_g_sub, dtype=int)

    # Solve LP
    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        integrality=integrality,
    )

    return {
        "objval": -result.fun if result.success else 0,
        "solution": (
            result.x if result.success else np.zeros(num_edges_g + num_edges_g_sub)
        ),
        "success": result.success,
    }


def _construct_path_system(
    lp_res: dict, g: ig.Graph, g_sub: ig.Graph, s: int, t: int
) -> list:
    """
    Construct path system from LP solution.

    Args:
        `lp_res`: LP solution from joint_flow
        `g`: Main graph
        `g_sub`: Subgraph
        `s`: Source vertex
        `t`: Sink vertex

    Returns:
        List of paths as edge sequences
    """
    solution = lp_res["solution"]

    # Check if solution is integer
    if not np.allclose(solution, np.round(solution)):
        return []

    solution = np.round(solution).astype(int)

    num_edges_g = g.ecount()
    num_edges_g_sub = g_sub.ecount()
    edges_g = g.get_edgelist()
    edges_g_sub = g_sub.get_edgelist()

    paths = []

    # Paths in subgraph
    active_edges_sub = [
        edges_g_sub[e] for e in range(num_edges_g_sub) if solution[num_edges_g + e] == 1
    ]

    if active_edges_sub:
        starting_edges = [(u, v) for u, v in active_edges_sub if u == s]
        for start_edge in starting_edges:
            path = [start_edge]
            current = start_edge[1]
            while current != t:
                next_edge = next((u, v) for u, v in active_edges_sub if u == current)
                path.append(next_edge)
                current = next_edge[1]
            paths.append(path)

    # Paths in main graph
    active_edges_main = [edges_g[e] for e in range(num_edges_g) if solution[e] == 1]

    if active_edges_main:
        starting_edges = [(u, v) for u, v in active_edges_main if u == s]
        for start_edge in starting_edges:
            path = [start_edge]
            current = start_edge[1]
            while current != t:
                next_edge = next((u, v) for u, v in active_edges_main if u == current)
                path.append(next_edge)
                current = next_edge[1]
            paths.append(path)

    return paths


def _construct_trek_system(
    res: dict, flow_graph: ig.Graph, flow_sub_graph: ig.Graph, s: int, t: int, m: int
) -> dict:
    """
    Construct trek system from flow LP solution.

    Args:
        `res`: LP solution
        `flow_graph`: Flow graph
        `flow_sub_graph`: Flow subgraph
        `s`: Source vertex
        `t`: Sink vertex
        `m`: Number of nodes in original graph (before doubling)

    Returns:
        Dictionary with 'TrekSystem' and 'startNodes' keys
    """
    path_system = _construct_path_system(res, flow_graph, flow_sub_graph, s, t)
    trek_system = []
    start_nodes = []

    for path in path_system:
        if not path:
            continue

        # Extract start node (second node in path, after source)
        start_nodes.append(path[0][1])

        # Remove source and sink from path
        trek_edges = path[1:-1] if len(path) > 2 else []

        # Map back from doubled graph: if node > m, subtract m
        trek = []
        for u, v in trek_edges:
            u_mapped = u - m if u >= m else u
            v_mapped = v - m if v >= m else v
            if u_mapped != v_mapped:  # Skip self-loops from doubling
                trek.append((u_mapped, v_mapped))

        if trek:
            trek_system.append(trek)

    return {"TrekSystem": trek_system, "startNodes": start_nodes}


def allowed_nodes_for_z(
    g: LatentSubgraph,
    v: int,
    S: list[int],
    H1: list[int],
    H2: list[int],
) -> list[int]:
    """
    Compute allowed nodes for Z given constraints.

    Args:
        `g`: Latent subgraph structure
        `v`: Target node
        `S`: Set of identified nodes
        `H1`: First latent node set
        `H2`: Second latent node set

    Returns:
        Allowed nodes for Z
    """
    semi_parents_of_v = g.semidirect.parents([v])

    tr_from_h1 = g.cov.tr_from(H1, include_latents=False)
    des_h2 = g.cov.descendants(H2, include_latents=False)

    allowed_nodes = (set(tr_from_h1) | set(des_h2)) & set(S)
    allowed_nodes = list(allowed_nodes - {v} - set(semi_parents_of_v))

    return allowed_nodes


def allowed_nodes_for_y(
    g_orig: LatentDigraph,
    g: LatentSubgraph,
    v: int,
    S: list[int],
    Z: list[int],
    H1: list[int],
    H2: list[int],
) -> list[int]:
    """
    Compute allowed nodes for Y given constraints.

    Args:
        `g_orig`: Original latent digraph
        `g`: Latent subgraph structure
        `v`: Target node
        `S`: Set of identified nodes
        `Z`: Z node set
        `H1`: First latent node set
        `H2`: Second latent node set

    Returns:
        Allowed nodes for Y
    """
    latent_tr_from_z_and_v = g.cov.tr_from(
        Z + [v],
        include_latents=False,
        avoid_left_nodes=H2,
        avoid_right_nodes=H1,
    )

    observed_set = set(g_orig.observed_nodes())
    ext_latent_tr = observed_set & set(g_orig.descendants(latent_tr_from_z_and_v))
    not_allowed = (ext_latent_tr - set(S)) | set(latent_tr_from_z_and_v)
    allowed = list(observed_set - not_allowed)

    return allowed


def _check_trek_system(
    g_orig: LatentDigraph, g: LatentSubgraph, Z: list[int], v: int, Ya: list[int]
) -> dict:
    """
    Check if a trek system exists for given parameters.

    Args:
        `g_orig`: Original latent digraph
        `g`: Latent subgraph structure
        `Z`: Z node set
        `v`: Target node
        `Ya`: Allowed Y nodes

    Returns:
        Dictionary with 'objval' and optionally 'trekSystem' and 'Y'
    """
    semi_parents_of_v = g.semidirect.parents([v])
    # observed_nodes = g_orig.observed_nodes()
    # latent_nodes = g_orig.latent_nodes()
    n_obs = g_orig.num_observed
    n_lat = g_orig.num_latents
    m = n_obs + n_lat

    # Create flow graph with source and sink
    s = 2 * m
    t = 2 * m + 1

    # Get LP graph adjacency matrices
    lp_adj = np.array(g.lp_graph.get_adjacency().data)
    lp_sub_adj = np.array(g.lp_sub_graph.get_adjacency().data)

    # Create flow graphs with source/sink
    flow_adj = np.zeros((2 * m + 2, 2 * m + 2), dtype=np.int32)
    flow_sub_adj = np.zeros((2 * m + 2, 2 * m + 2), dtype=np.int32)

    flow_adj[: 2 * m, : 2 * m] = lp_adj
    flow_adj[s, Ya] = 1
    for node in semi_parents_of_v + Z:
        flow_adj[m + node, t] = 1

    flow_sub_adj[: 2 * m, : 2 * m] = lp_sub_adj
    flow_sub_adj[s, Ya] = 1
    for node in Z:
        flow_sub_adj[m + node, t] = 1

    # Create igraph objects
    flow_graph = ig.Graph.Adjacency(flow_adj.tolist(), mode="directed")
    flow_sub_graph = ig.Graph.Adjacency(flow_sub_adj.tolist(), mode="directed")

    # Solve LP
    res = joint_flow(flow_graph, flow_sub_graph, s, t, integer=False)
    objval = res["objval"]

    # Check if we need integer solution
    if not np.allclose(res["solution"], np.round(res["solution"])):
        res_int = joint_flow(flow_graph, flow_sub_graph, s, t, integer=True)
        objval = res_int["objval"]
        res = res_int

    target_flow = len(Z) + len(semi_parents_of_v)

    if np.isclose(objval, target_flow):
        trek_result = _construct_trek_system(res, flow_graph, flow_sub_graph, s, t, m)
        return {
            "objval": objval,
            "trekSystem": trek_result["TrekSystem"],
            "Y": trek_result["startNodes"],
        }
    else:
        return {"objval": objval}


def _generate_h1_h2_combinations(latent_nodes: list[int], max_k: int):
    """
    Generate all (k, H1, H2) combinations for LSC algorithm.

    Yields tuples of (k, H1, H2) where k = |H1| + |H2|.
    """
    for i in range(max_k + 1):
        for j in range(i + 1):
            for H1 in subsets_of_size(latent_nodes, j):
                for H2 in subsets_of_size(latent_nodes, i - j):
                    yield i, H1, H2


def lsc_id(g: LatentDigraph, subset_size_control: int | None = None) -> dict:
    """
    Determine which edges in a latent digraph are LSC-identifiable.

    Uses the Latent Subgraph Criterion to determine which edges in a latent
    digraph are generically identifiable.

    Args:
        `g`: The latent digraph to analyze
        `subset_size_control`: Maximum size of latent node subsets to consider
                              (default None, meaning no limit)

    Returns:
        Dictionary with 'S' (identified nodes), 'Ys', 'Zs', 'H1s', 'H2s',
        'trekSystems', and 'id' (whether all nodes are identified)
    """
    # Build auxiliary graphs
    lg = LatentSubgraph(g)

    observed_nodes = g.observed_nodes()
    latent_nodes = g.latent_nodes()
    n_obs = g.num_observed
    n_lat = g.num_latents

    # Find source nodes (nodes with no semi-direct parents)
    S = [n for n in observed_nodes if not lg.semidirect.parents([n])]

    Ys = [[] for _ in range(n_obs)]
    Zs = [[] for _ in range(n_obs)]
    H1s = [[] for _ in range(n_obs)]
    H2s = [[] for _ in range(n_obs)]
    trek_systems = [[] for _ in range(n_obs)]

    change_flag = len(S) != len(observed_nodes)

    while change_flag:
        change_flag = False

        # Loop over unsolved nodes
        for v in [n for n in observed_nodes if n not in S]:
            semi_parents_of_v = lg.semidirect.parents([v])

            # Loop over possible cardinalities of |H1| + |H2|
            max_k = (
                n_lat
                if subset_size_control is None
                else min(subset_size_control, n_lat)
            )
            found = False

            for k, H1, H2 in _generate_h1_h2_combinations(latent_nodes, max_k):
                if found:
                    break

                # Compute allowed nodes for Z
                Za = allowed_nodes_for_z(lg, v, S, H1, H2)

                for Z in subsets_of_size(Za, k):
                    # Compute allowed nodes for Y
                    Ya = allowed_nodes_for_y(g, lg, v, S, Z, H1, H2)

                    if len(Ya) >= len(semi_parents_of_v) + len(Z):
                        res = _check_trek_system(g, lg, Z, v, Ya)

                        # Check if trek system exists
                        if np.isclose(
                            res["objval"],
                            len(semi_parents_of_v) + len(Z),
                        ):
                            Ys[v] = res["Y"]
                            Zs[v] = Z
                            H1s[v] = H1
                            H2s[v] = H2
                            trek_systems[v] = res["trekSystem"]
                            S.append(v)
                            change_flag = len(S) != len(observed_nodes)
                            found = True
                            break

    identifiable = len(S) == len(observed_nodes)

    return {
        "S": S,
        "Ys": Ys,
        "Zs": Zs,
        "H1s": H1s,
        "H2s": H2s,
        "trekSystems": trek_systems,
        "id": identifiable,
    }
