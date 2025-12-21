import numpy as np
import pytest

from semid import MixedGraph

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
