import numpy as np
import pytest

from tests.conftest import LATENT_DIGRAPH_EXAMPLES


# =============================================================================
# Property tests
# =============================================================================


@pytest.mark.parametrize("test_case", LATENT_DIGRAPH_EXAMPLES, ids=lambda tc: tc.name)
def test_num_observed(test_case):
    """Test num_observed property."""
    graph = test_case.to_latent_digraph()
    assert graph.num_observed == test_case.expected_num_observed


@pytest.mark.parametrize("test_case", LATENT_DIGRAPH_EXAMPLES, ids=lambda tc: tc.name)
def test_num_latents(test_case):
    """Test num_latents property."""
    graph = test_case.to_latent_digraph()
    assert graph.num_latents == test_case.expected_num_latents


@pytest.mark.parametrize("test_case", LATENT_DIGRAPH_EXAMPLES, ids=lambda tc: tc.name)
def test_observed_nodes(test_case):
    """Test observed_nodes() method."""
    graph = test_case.to_latent_digraph()
    assert graph.observed_nodes() == test_case.expected_observed_nodes


@pytest.mark.parametrize("test_case", LATENT_DIGRAPH_EXAMPLES, ids=lambda tc: tc.name)
def test_latent_nodes(test_case):
    """Test latent_nodes() method."""
    graph = test_case.to_latent_digraph()
    assert graph.latent_nodes() == test_case.expected_latent_nodes


# =============================================================================
# Method tests
# =============================================================================


@pytest.mark.parametrize("test_case", LATENT_DIGRAPH_EXAMPLES, ids=lambda tc: tc.name)
def test_parents(test_case):
    """Test parents() method."""
    graph = test_case.to_latent_digraph()
    for test in test_case.parents_tests:
        nodes = test["nodes"]
        expected = test["expected"]
        result = graph.parents(nodes)
        assert sorted(result) == sorted(expected), (
            f"parents({nodes}) mismatch: got {result}, expected {expected}"
        )


@pytest.mark.parametrize("test_case", LATENT_DIGRAPH_EXAMPLES, ids=lambda tc: tc.name)
def test_observed_parents(test_case):
    """Test observed_parents() method."""
    graph = test_case.to_latent_digraph()
    for test in test_case.observed_parents_tests:
        nodes = test["nodes"]
        expected = test["expected"]
        result = graph.observed_parents(nodes)
        assert sorted(result) == sorted(expected), (
            f"observed_parents({nodes}) mismatch: got {result}, expected {expected}"
        )


@pytest.mark.parametrize("test_case", LATENT_DIGRAPH_EXAMPLES, ids=lambda tc: tc.name)
def test_ancestors(test_case):
    """Test ancestors() method."""
    graph = test_case.to_latent_digraph()
    for test in test_case.ancestors_tests:
        nodes = test["nodes"]
        expected = test["expected"]
        result = graph.ancestors(nodes)
        assert sorted(result) == sorted(expected), (
            f"ancestors({nodes}) mismatch: got {result}, expected {expected}"
        )


@pytest.mark.parametrize("test_case", LATENT_DIGRAPH_EXAMPLES, ids=lambda tc: tc.name)
def test_descendants(test_case):
    """Test descendants() method."""
    graph = test_case.to_latent_digraph()
    for test in test_case.descendants_tests:
        nodes = test["nodes"]
        expected = test["expected"]
        result = graph.descendants(nodes)
        assert sorted(result) == sorted(expected), (
            f"descendants({nodes}) mismatch: got {result}, expected {expected}"
        )


@pytest.mark.parametrize("test_case", LATENT_DIGRAPH_EXAMPLES, ids=lambda tc: tc.name)
def test_induced_subgraph(test_case):
    """Test induced_subgraph() method."""
    graph = test_case.to_latent_digraph()
    for test in test_case.induced_subgraph_tests:
        nodes = test["nodes"]
        expected_num_observed = test["expected_num_observed"]
        expected_adj = np.array(test["expected_adj"], dtype=np.int32)
        n = int(np.sqrt(len(expected_adj)))
        expected_adj = expected_adj.reshape((n, n))

        subgraph = graph.induced_subgraph(nodes)
        assert subgraph.num_observed == expected_num_observed, (
            f"induced_subgraph({nodes}).num_observed mismatch"
        )
        np.testing.assert_array_equal(
            subgraph.adj, expected_adj,
            err_msg=f"induced_subgraph({nodes}).adj mismatch"
        )


# =============================================================================
# Trek system and tr_from tests
# =============================================================================


@pytest.mark.parametrize("test_case", LATENT_DIGRAPH_EXAMPLES, ids=lambda tc: tc.name)
def test_get_trek_system(test_case):
    """Test get_trek_system with various configurations"""
    graph = test_case.to_latent_digraph()

    for trek_test in test_case.trek_system_tests:
        from_nodes = trek_test["from_nodes"]
        to_nodes = trek_test["to_nodes"]
        avoid_left = trek_test.get("avoid_left_nodes", [])
        avoid_right = trek_test.get("avoid_right_nodes", [])
        avoid_left_edges = trek_test.get("avoid_left_edges", [])
        avoid_right_edges = trek_test.get("avoid_right_edges", [])

        result = graph.get_trek_system(
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            avoid_left_nodes=avoid_left,
            avoid_right_nodes=avoid_right,
            avoid_left_edges=avoid_left_edges,
            avoid_right_edges=avoid_right_edges,
        )

        expected_exists = trek_test["expected_exists"]
        expected_active = trek_test["expected_active"]

        assert result.system_exists == expected_exists, (
            f"Trek system existence mismatch for from={from_nodes}, to={to_nodes}, "
            f"avoid_left={avoid_left}, avoid_right={avoid_right}"
        )

        assert sorted(result.active_from) == sorted(expected_active), (
            f"Active nodes mismatch for from={from_nodes}, to={to_nodes}, "
            f"avoid_left={avoid_left}, avoid_right={avoid_right}"
        )


@pytest.mark.parametrize("test_case", LATENT_DIGRAPH_EXAMPLES)
def test_tr_from(test_case):
    """Test tr_from with various configurations"""
    graph = test_case.to_latent_digraph()

    for tr_test in test_case.tr_from_tests:
        nodes = tr_test["nodes"]
        avoid_left = tr_test.get("avoid_left_nodes", [])
        avoid_right = tr_test.get("avoid_right_nodes", [])
        expected = tr_test["expected"]

        result = graph.tr_from(
            nodes=nodes,
            avoid_left_nodes=avoid_left,
            avoid_right_nodes=avoid_right,
        )

        assert sorted(result) == sorted(expected), (
            f"tr_from mismatch for nodes={nodes}, "
            f"avoid_left={avoid_left}, avoid_right={avoid_right}"
        )
