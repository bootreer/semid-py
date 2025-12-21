import pytest

from tests.conftest import LATENT_DIGRAPH_EXAMPLES


@pytest.mark.parametrize("test_case", LATENT_DIGRAPH_EXAMPLES)
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
