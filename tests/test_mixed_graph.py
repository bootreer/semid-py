import numpy as np
import pytest

from semid import MixedGraph
from semid.identification.ancestral import ancestral_identify_step
from semid.identification.algorithm import general_generic_id
from semid.identification.edgewise import edgewise_id, edgewise_identify_step
from semid.identification.htc import htc_id, htc_identify_step
from semid.identification.trek_separation import trek_separation_identify_step

from tests.conftest import GRAPH_EXAMPLES, Graph


@pytest.mark.parametrize("graph", GRAPH_EXAMPLES)
def test_htc_id(graph: Graph):
    m = graph.dim[0]
    mg: MixedGraph = graph.to_mixed_graph()
    result = mg.htc_id()

    if graph.htc_id == 1:
        assert result is not None
        assert (sorted(result) == np.arange(m)).all(), (
            "Identifiable graph should return all nodes as half-trek identifiable"
        )
    else:
        assert result is None or (len(result) < m and np.isin(result, range(m)).all())


@pytest.mark.parametrize("graph", GRAPH_EXAMPLES)
def test_non_htc_id(graph: Graph):
    mg: MixedGraph = graph.to_mixed_graph()
    result = mg.non_htc_id()

    if graph.htc_id == 0:
        assert result
    else:
        assert not result


@pytest.mark.parametrize("graph", GRAPH_EXAMPLES)
def test_global_id(graph: Graph):
    mg: MixedGraph = graph.to_mixed_graph()
    result = mg.global_id()

    if graph.global_id == 1:
        assert result is True
    else:
        assert result is False


# ---------------------------------------------------------------------------
# Regression tests for external vertex ID handling
# ---------------------------------------------------------------------------

# A simple identifiable graph: 0->1->2 with no bidirected edges.
# Used as a baseline; we then test the same graph with shifted vertex_nums.
_CHAIN_L = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int32)
_CHAIN_O = np.zeros((3, 3), dtype=np.int32)

# Verma graph – HTC-identifiable with a bidirected edge.
_VERMA_L = np.array(
    [[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.int32
)
_VERMA_O = np.array(
    [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.int32
)


def test_is_sibling_default_vertex_nums():
    """is_sibling returns correct result when vertex_nums are the default 0-based."""
    L = np.array([[0, 0], [0, 0]], dtype=np.int32)
    O = np.array([[0, 1], [1, 0]], dtype=np.int32)
    g = MixedGraph(L, O)
    assert g.is_sibling(0, 1) is True
    assert g.is_sibling(1, 0) is True


def test_is_sibling_custom_vertex_nums():
    """is_sibling must use external→internal conversion; previously indexed b_adj with raw external IDs."""
    L = np.array([[0, 0], [0, 0]], dtype=np.int32)
    O = np.array([[0, 1], [1, 0]], dtype=np.int32)
    # nodes are 10 and 20 – without the fix, is_sibling(10, 20) would try b_adj[10, 20] → IndexError
    g = MixedGraph(L, O, vertex_nums=[10, 20])
    assert g.is_sibling(10, 20) is True
    assert g.is_sibling(20, 10) is True


def test_is_sibling_no_edge_custom_vertex_nums():
    """is_sibling returns False for nodes without a bidirected edge, with custom vertex_nums."""
    L = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int32)
    O = np.zeros((3, 3), dtype=np.int32)
    g = MixedGraph(L, O, vertex_nums=[5, 10, 15])
    assert g.is_sibling(5, 10) is False
    assert g.is_sibling(10, 15) is False


def test_get_mixed_comp_custom_vertex_nums():
    """get_mixed_comp must BFS in internal index space; previously mixed internal igraph indices with external IDs."""
    # Simple 3-node graph with a bidirected edge between nodes 10 and 20
    L = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int32)
    O = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.int32)
    g = MixedGraph(L, O, vertex_nums=[10, 20, 30])

    # Starting from node 10, avoiding [30]: should reach 10 and 20 via bidirected edge
    result = g.get_mixed_comp(sub_nodes=[30], node=10)
    # parents of {10, 20}: 20's parent is 10 (directed edge 10->20); 10 has no parents
    # in_nodes = parents not in reachable, intersected with sub_nodes
    assert isinstance(result.in_nodes, list)


def test_get_mixed_comp_results_match_default_and_custom_vertex_nums():
    """get_mixed_comp should give equivalent results regardless of vertex_nums."""
    L = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int32)
    O = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.int32)

    g_default = MixedGraph(L, O)
    g_custom = MixedGraph(L, O, vertex_nums=[10, 20, 30])

    result_default = g_default.get_mixed_comp(sub_nodes=[2], node=0)
    result_custom = g_custom.get_mixed_comp(sub_nodes=[30], node=10)

    # in_nodes should have the same count
    assert len(result_default.in_nodes) == len(result_custom.in_nodes)


def test_htc_id_custom_vertex_nums_matches_default():
    """htc_id on a graph with custom vertex_nums should identify the same edges as the 0-based equivalent."""
    g_default = MixedGraph(_VERMA_L, _VERMA_O)
    g_custom = MixedGraph(_VERMA_L, _VERMA_O, vertex_nums=[10, 20, 30, 40])

    result_default = htc_id(g_default)
    result_custom = htc_id(g_custom)

    # Same number of solved/unsolved parents
    solved_default = sum(len(p) for p in result_default.solved_parents)
    solved_custom = sum(len(p) for p in result_custom.solved_parents)
    assert solved_default == solved_custom

    unsolved_default = sum(len(p) for p in result_default.unsolved_parents)
    unsolved_custom = sum(len(p) for p in result_custom.unsolved_parents)
    assert unsolved_default == unsolved_custom


def test_edgewise_id_custom_vertex_nums_matches_default():
    """edgewise_id on a graph with custom vertex_nums should identify the same edges as the 0-based equivalent."""
    g_default = MixedGraph(_VERMA_L, _VERMA_O)
    g_custom = MixedGraph(_VERMA_L, _VERMA_O, vertex_nums=[10, 20, 30, 40])

    result_default = edgewise_id(g_default)
    result_custom = edgewise_id(g_custom)

    solved_default = sum(len(p) for p in result_default.solved_parents)
    solved_custom = sum(len(p) for p in result_custom.solved_parents)
    assert solved_default == solved_custom

    unsolved_default = sum(len(p) for p in result_default.unsolved_parents)
    unsolved_custom = sum(len(p) for p in result_custom.unsolved_parents)
    assert unsolved_default == unsolved_custom


def test_identification_with_custom_vertex_nums_tian_decompose_false():
    """With tian_decompose=False, identification must still work on a graph with custom vertex_nums."""
    # Without the guard in general_generic_id, this would crash with KeyError
    # when htr_from/parents receive 0-based indices as if they were external IDs like [10, 20, 30]
    g_custom = MixedGraph(_VERMA_L, _VERMA_O, vertex_nums=[10, 20, 30, 40])
    g_default = MixedGraph(_VERMA_L, _VERMA_O)

    result_custom = htc_id(g_custom, tian_decompose=False)
    result_default = htc_id(g_default, tian_decompose=False)

    solved_custom = sum(len(p) for p in result_custom.solved_parents)
    solved_default = sum(len(p) for p in result_default.solved_parents)
    assert solved_custom == solved_default


def test_mixed_graph_htc_id_returns_external_ids():
    """MixedGraph.htc_id() must return external vertex IDs, not internal 0-based indices."""
    g = MixedGraph(_VERMA_L, _VERMA_O, vertex_nums=[10, 20, 30, 40])
    result = g.htc_id()
    assert result is not None
    # All returned IDs must be external vertex IDs, not internal indices 0-3
    for node in result:
        assert node in [10, 20, 30, 40], f"Expected external ID, got {node}"


def test_mixed_graph_htc_id_default_vertex_nums_unchanged():
    """MixedGraph.htc_id() result is unchanged for default vertex_nums."""
    g_default = MixedGraph(_VERMA_L, _VERMA_O)
    g_custom = MixedGraph(_VERMA_L, _VERMA_O, vertex_nums=[10, 20, 30, 40])

    result_default = g_default.htc_id()
    result_custom = g_custom.htc_id()

    assert result_default is not None
    assert result_custom is not None
    # Custom result should map each default internal index through vertex_nums=[10,20,30,40]
    vertex_nums = [10, 20, 30, 40]
    expected = sorted(vertex_nums[n] for n in result_default)
    assert sorted(result_custom) == expected


# ---------------------------------------------------------------------------
# Assertions: step functions must not be called with non-normalized graphs
# ---------------------------------------------------------------------------

_NON_NORMALIZED = MixedGraph(_VERMA_L, _VERMA_O, vertex_nums=[10, 20, 30, 40])
_DUMMY_UNSOLVED = [[] for _ in range(4)]
_DUMMY_SOLVED = [[] for _ in range(4)]


def _dummy_identifier(Sigma):
    import numpy as np

    n = Sigma.shape[0]
    from semid.identification.types import IdentifierResult

    return IdentifierResult(np.zeros((n, n)), np.zeros((n, n)))


def test_htc_identify_step_asserts_on_non_normalized_graph():
    with pytest.raises(ValueError, match="0-based vertex_nums"):
        htc_identify_step(
            _NON_NORMALIZED, _DUMMY_UNSOLVED, _DUMMY_SOLVED, _dummy_identifier
        )


def test_edgewise_identify_step_asserts_on_non_normalized_graph():
    with pytest.raises(AssertionError, match="0-based vertex_nums"):
        edgewise_identify_step(
            _NON_NORMALIZED, _DUMMY_UNSOLVED, _DUMMY_SOLVED, _dummy_identifier
        )


def test_trek_separation_identify_step_asserts_on_non_normalized_graph():
    with pytest.raises(AssertionError, match="0-based vertex_nums"):
        trek_separation_identify_step(
            _NON_NORMALIZED, _DUMMY_UNSOLVED, _DUMMY_SOLVED, _dummy_identifier
        )


def test_ancestral_identify_step_asserts_on_non_normalized_graph():
    with pytest.raises(ValueError, match="0-based vertex_nums"):
        ancestral_identify_step(
            _NON_NORMALIZED, _DUMMY_UNSOLVED, _DUMMY_SOLVED, _dummy_identifier
        )


# ---------------------------------------------------------------------------
# general_generic_id: result must use external IDs when custom vertex_nums given
# ---------------------------------------------------------------------------


def test_general_generic_id_result_uses_external_ids():
    """GenericIDResult.solved_parents and .mixed_graph.nodes must use external IDs."""
    vertex_nums = [10, 20, 30, 40]
    g = MixedGraph(_VERMA_L, _VERMA_O, vertex_nums=vertex_nums)

    result = general_generic_id(g, [htc_identify_step])

    # The graph stored in the result must carry the original external IDs
    assert result.mixed_graph.nodes == vertex_nums

    external_id_set = set(vertex_nums)

    # Every parent ID in solved_parents must be an external ID
    for parents in result.solved_parents:
        for p in parents:
            assert p in external_id_set, (
                f"solved_parents contains internal index {p}, expected one of {vertex_nums}"
            )

    # Every parent ID in unsolved_parents must be an external ID
    for parents in result.unsolved_parents:
        for p in parents:
            assert p in external_id_set, (
                f"unsolved_parents contains internal index {p}, expected one of {vertex_nums}"
            )


def test_general_generic_id_result_matches_default_count():
    """With custom vertex_nums the number of identified edges must match the default graph."""
    g_default = MixedGraph(_VERMA_L, _VERMA_O)
    g_custom = MixedGraph(_VERMA_L, _VERMA_O, vertex_nums=[10, 20, 30, 40])

    result_default = general_generic_id(g_default, [htc_identify_step])
    result_custom = general_generic_id(g_custom, [htc_identify_step])

    assert sum(len(p) for p in result_default.solved_parents) == sum(
        len(p) for p in result_custom.solved_parents
    )
    assert sum(len(p) for p in result_default.unsolved_parents) == sum(
        len(p) for p in result_custom.unsolved_parents
    )


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_mixed_graph_repr():
    import numpy as np
    from semid import MixedGraph

    L = np.array([[0, 1], [0, 0]], dtype=np.int32)
    O = np.array([[0, 1], [1, 0]], dtype=np.int32)
    g = MixedGraph(L, O)
    r = repr(g)
    assert "MixedGraph" in r
    assert "n_nodes=2" in r
    assert "n_directed=1" in r
    assert "n_bidirected=1" in r


# ---------------------------------------------------------------------------
# MixedGraph.from_edges
# ---------------------------------------------------------------------------


def test_from_edges_basic():
    """from_edges constructs the same graph as the matrix constructor."""
    import numpy as np
    from semid import MixedGraph

    L = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int32)
    O = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.int32)
    g_mat = MixedGraph(L, O)

    g_edges = MixedGraph.from_edges(
        n_nodes=3,
        directed=[(0, 1), (1, 2)],
        bidirected=[(0, 2)],
    )

    np.testing.assert_array_equal(g_mat.d_adj, g_edges.d_adj)
    np.testing.assert_array_equal(g_mat.b_adj, g_edges.b_adj)


def test_from_edges_with_custom_vertex_nums():
    """from_edges supports custom vertex numbering."""
    from semid import MixedGraph

    g = MixedGraph.from_edges(
        n_nodes=3,
        directed=[(10, 20), (20, 30)],
        bidirected=[(10, 30)],
        vertex_nums=[10, 20, 30],
    )
    assert g.nodes == [10, 20, 30]
    assert 20 in g.parents(30)


def test_from_edges_empty():
    """from_edges with no edges creates an empty graph."""
    from semid import MixedGraph

    g = MixedGraph.from_edges(n_nodes=4)
    assert g.num_nodes == 4
    assert g.parents(0) == []
    assert g.siblings(0) == []


def test_from_edges_invalid_node_raises():
    """from_edges raises ValueError for edges referencing unknown nodes."""
    import pytest
    from semid import MixedGraph

    with pytest.raises(ValueError, match="references a node not in vertex_nums"):
        MixedGraph.from_edges(n_nodes=2, directed=[(0, 5)])


def test_constructor_n_param_reshapes_flat_array():
    """n= parameter reshapes a flat 1D list into an n×n matrix."""
    import numpy as np
    from semid import MixedGraph

    L_flat = [0, 1, 0, 0, 0, 0, 0, 0, 0]
    O_flat = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    g = MixedGraph(L_flat, O_flat, n=3)
    assert g.num_nodes == 3
    assert g.d_adj[0, 1] == 1
