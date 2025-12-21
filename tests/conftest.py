from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy._typing import NDArray
from numpy.random import uniform

from semid import LatentDigraph, MixedGraph

import igraph as ig


# Common test graphs
VERMA_L = np.array(
    [[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.int32
)
VERMA_O = np.array(
    [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.int32
)


@dataclass
class Graph:
    L: list[int]
    O: list[int]  # noqa: E741
    dim: tuple[int, int]

    htc_id: int
    """
    1: identifiable
    0: non-identifiable
    -1: inconclusive
    """

    global_id: int

    def _symmetricize(self, arr: NDArray) -> NDArray:
        return ((arr + arr.T) > 0).astype(int)

    def to_mixed_graph(self) -> MixedGraph:
        L = np.reshape(self.L, self.dim)
        O_sym = self._symmetricize(np.reshape(self.O, self.dim))
        return MixedGraph(d_adj=L, b_adj=O_sym)


GRAPH_EXAMPLES: list[Graph] = [
    # Empty Graph
    Graph(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        (5, 5),
        1,
        1,
    ),
    # Verma graph
    Graph(
        [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        (4, 4),
        1,
        1,
    ),
    # Ex. 3a HTC id
    Graph(
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        (5, 5),
        1,
        0,
    ),
    # Ex. 3b HTC id
    Graph(
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        (5, 5),
        1,
        0,
    ),
    # Ex. 3c HTC inc
    Graph(
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        (5, 5),
        -1,
        0,
    ),
    # Ex. 3d HTC id
    Graph(
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        (5, 5),
        1,
        0,
    ),
    # Ex. 3e HTC inc
    Graph(
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        (5, 5),
        -1,
        0,
    ),
    # Ex. 4a HTC nonid
    Graph(
        [0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        (5, 5),
        0,
        0,
    ),
    # Ex. 4b HTC inc
    Graph(
        [0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        (5, 5),
        -1,
        0,
    ),
    # Ex. 4c HTC nonid
    Graph(
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        (5, 5),
        0,
        0,
    ),
    # Ex. 4d HTC inc
    Graph(
        [0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        (5, 5),
        -1,
        0,
    ),
    # Ex. 5a HTC inc
    Graph(
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        (5, 5),
        -1,
        0,
    ),
    # Ex. 5b HTC inc
    Graph(
        [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        (5, 5),
        -1,
        0,
    ),
    # Ex. 5c HTC inc
    Graph(
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        (5, 5),
        -1,
        0,
    ),
    # Ex. 5d HTC inc
    Graph(
        [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        (5, 5),
        -1,
        0,
    ),
    # Simple 2 node non-identifiable
    Graph(
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        (2, 2),
        0,
        0,
    ),
    # Ex. 7a (Fig 9a) - HTC inconclusive, but Tian-identifiable
    Graph(
        [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        (5, 5),
        -1,
        0,
    ),
    # Sink node marginalization example - ancestrally identifiable
    # Edges: 0->1, 0->2, 0->5, 1->2, 1->3, 1->4, 1->5, 2->3, 3->4
    # Bidirected: 0-5, 0-3, 1-2, 1-4, 1-5
    Graph(
        [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        (6, 6),
        -1,
        0,
    ),
    # Instrumental variable model
    Graph(
        [0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0],
        (3, 3),
        1,
        0,
    ),
]


@dataclass
class LatentDigraphTestCase:
    """Test case for LatentDigraph"""

    name: str
    """Descriptive name for the test case"""

    L: list[int]
    """Adjacency matrix as flattened list"""

    dim: tuple[int, int]
    """Dimensions of the adjacency matrix"""

    num_observed: Optional[int] = None
    """Number of observed nodes (None means all observed)"""

    # Expected property values
    expected_num_observed: int = 0
    expected_num_latents: int = 0
    expected_observed_nodes: list[int] = field(default_factory=list)
    expected_latent_nodes: list[int] = field(default_factory=list)

    # Expected results for method tests: list of {nodes: [...], expected: [...]}
    parents_tests: list[dict] = field(default_factory=list)
    observed_parents_tests: list[dict] = field(default_factory=list)
    ancestors_tests: list[dict] = field(default_factory=list)
    descendants_tests: list[dict] = field(default_factory=list)

    # {nodes: [...], expected_num_observed: int, expected_adj: [...]}
    induced_subgraph_tests: list[dict] = field(default_factory=list)

    # trek system and tr_from tests
    trek_system_tests: list[dict] = field(default_factory=list)
    tr_from_tests: list[dict] = field(default_factory=list)

    def to_latent_digraph(self) -> LatentDigraph:
        L = np.reshape(self.L, self.dim).astype(np.int32)
        return LatentDigraph(L, num_observed=self.num_observed)


# LatentDigraph test cases ported from R tests
# fmt: off
LATENT_DIGRAPH_EXAMPLES: list[LatentDigraphTestCase] = [
    # Empty graph (0 nodes)
    LatentDigraphTestCase(
        name="empty_graph",
        L=[],
        dim=(0, 0),
        num_observed=0,
        expected_num_observed=0,
        expected_num_latents=0,
        expected_observed_nodes=[],
        expected_latent_nodes=[],
        parents_tests=[
            {"nodes": [], "expected": []},
        ],
        observed_parents_tests=[
            {"nodes": [], "expected": []},
        ],
        ancestors_tests=[
            {"nodes": [], "expected": []},
        ],
        descendants_tests=[
            {"nodes": [], "expected": []},
        ],
        induced_subgraph_tests=[],
        trek_system_tests=[],
        tr_from_tests=[],
    ),
    # Single node graph
    LatentDigraphTestCase(
        name="single_node",
        L=[0],
        dim=(1, 1),
        num_observed=1,
        expected_num_observed=1,
        expected_num_latents=0,
        expected_observed_nodes=[0],
        expected_latent_nodes=[],
        parents_tests=[
            {"nodes": [], "expected": []},
            {"nodes": [0], "expected": []},
        ],
        observed_parents_tests=[
            {"nodes": [], "expected": []},
            {"nodes": [0], "expected": []},
        ],
        ancestors_tests=[
            {"nodes": [], "expected": []},
            {"nodes": [0], "expected": [0]},
        ],
        descendants_tests=[
            {"nodes": [], "expected": []},
            {"nodes": [0], "expected": [0]},
        ],
        induced_subgraph_tests=[
            {"nodes": [0], "expected_num_observed": 1, "expected_adj": [0]},
        ],
        trek_system_tests=[
            {
                "from_nodes": [0],
                "to_nodes": [0],
                "expected_exists": True,
                "expected_active": [0],
            }
        ],
        tr_from_tests=[
            {"nodes": [0], "expected": [0]},
            {"nodes": [0], "avoid_left_nodes": [0], "expected": []},
            {"nodes": [0], "avoid_right_nodes": [0], "expected": [0]},
        ],
    ),
    # Two nodes, no latents: 0 -> 1
    LatentDigraphTestCase(
        name="two_nodes_no_latents",
        L=[0, 1, 0, 0],
        dim=(2, 2),
        num_observed=2,
        expected_num_observed=2,
        expected_num_latents=0,
        expected_observed_nodes=[0, 1],
        expected_latent_nodes=[],
        parents_tests=[
            {"nodes": [], "expected": []},
            {"nodes": [0], "expected": []},
            {"nodes": [1], "expected": [0]},
        ],
        observed_parents_tests=[
            {"nodes": [], "expected": []},
            {"nodes": [0], "expected": []},
            {"nodes": [1], "expected": [0]},
        ],
        ancestors_tests=[
            {"nodes": [], "expected": []},
            {"nodes": [0], "expected": [0]},
            {"nodes": [1], "expected": [0, 1]},
        ],
        descendants_tests=[
            {"nodes": [], "expected": []},
            {"nodes": [0], "expected": [0, 1]},
            {"nodes": [1], "expected": [1]},
        ],
        induced_subgraph_tests=[
            {"nodes": [0, 1], "expected_num_observed": 2, "expected_adj": [0, 1, 0, 0]},
            {"nodes": [0], "expected_num_observed": 1, "expected_adj": [0]},
        ],
        trek_system_tests=[
            {
                "from_nodes": [0],
                "to_nodes": [0],
                "expected_exists": True,
                "expected_active": [0],
            },
            {
                "from_nodes": [1],
                "to_nodes": [0],
                "expected_exists": True,
                "expected_active": [1],
            },
            {
                "from_nodes": [0, 1],
                "to_nodes": [1, 0],
                "expected_exists": True,
                "expected_active": [0, 1],
            },
        ],
        tr_from_tests=[
            {"nodes": [0], "expected": [0, 1]},
            {"nodes": [1], "expected": [0, 1]},
            {"nodes": [1], "avoid_left_nodes": [0], "expected": [1]},
            {"nodes": [0], "avoid_right_nodes": [1], "expected": [0]},
        ],
    ),
    # Two observed (0,1), one latent (2): 0->1, 2->0, 2->1
    LatentDigraphTestCase(
        name="two_observed_one_latent",
        L=[0, 1, 0, 0, 0, 0, 1, 1, 0],
        dim=(3, 3),
        num_observed=2,
        expected_num_observed=2,
        expected_num_latents=1,
        expected_observed_nodes=[0, 1],
        expected_latent_nodes=[2],
        parents_tests=[
            {"nodes": [], "expected": []},
            {"nodes": [0], "expected": [2]},
            {"nodes": [1], "expected": [0, 2]},
            {"nodes": [2], "expected": []},
        ],
        observed_parents_tests=[
            {"nodes": [], "expected": []},
            {"nodes": [0], "expected": []},
            {"nodes": [1], "expected": [0]},
        ],
        ancestors_tests=[
            {"nodes": [], "expected": []},
            {"nodes": [0], "expected": [0, 2]},
            {"nodes": [1], "expected": [0, 1, 2]},
            {"nodes": [2], "expected": [2]},
        ],
        descendants_tests=[
            {"nodes": [], "expected": []},
            {"nodes": [0], "expected": [0, 1]},
            {"nodes": [1], "expected": [1]},
            {"nodes": [2], "expected": [0, 1, 2]},
        ],
        induced_subgraph_tests=[
            {"nodes": [0, 1, 2], "expected_num_observed": 2, "expected_adj": [0, 1, 0, 0, 0, 0, 1, 1, 0]},
            {"nodes": [0, 1], "expected_num_observed": 2, "expected_adj": [0, 1, 0, 0]},
            {"nodes": [0], "expected_num_observed": 1, "expected_adj": [0]},
        ],
        trek_system_tests=[
            {
                "from_nodes": [0],
                "to_nodes": [0],
                "expected_exists": True,
                "expected_active": [0],
            },
            {
                "from_nodes": [1],
                "to_nodes": [0],
                "expected_exists": True,
                "expected_active": [1],
            },
            {
                "from_nodes": [0, 1],
                "to_nodes": [1, 0],
                "expected_exists": True,
                "expected_active": [0, 1],
            },
            {
                "from_nodes": [2, 1],
                "to_nodes": [1, 0],
                "expected_exists": True,
                "expected_active": [1, 2],
            },
        ],
        tr_from_tests=[
            {"nodes": [0], "expected": [0, 1, 2]},
            {"nodes": [1], "expected": [0, 1, 2]},
            {"nodes": [2], "expected": [0, 1, 2]},
            {"nodes": [0], "avoid_left_nodes": [2], "expected": [0, 1]},
            {
                "nodes": [0],
                "avoid_left_nodes": [2],
                "avoid_right_nodes": [1],
                "expected": [0],
            },
        ],
    ),
    # Complex graph: 5 observed (0-4), 2 latents (5-6)
    # Edges: 0->1, 1->2, 2->1, 3->4, 3->6, 5->1, 5->2, 5->4, 6->0, 6->4
    LatentDigraphTestCase(
        name="complex_5obs_2latent",
        L=[
            0, 1, 0, 0, 0, 0, 0,  # 0
            0, 0, 1, 0, 0, 0, 0,  # 1
            0, 1, 0, 0, 0, 0, 0,  # 2
            0, 0, 0, 0, 1, 0, 1,  # 3
            0, 0, 0, 0, 0, 0, 0,  # 4
            0, 1, 1, 0, 1, 0, 0,  # 5 (latent)
            1, 0, 0, 0, 1, 0, 0,  # 6 (latent)
        ],
        dim=(7, 7),
        num_observed=5,
        expected_num_observed=5,
        expected_num_latents=2,
        expected_observed_nodes=[0, 1, 2, 3, 4],
        expected_latent_nodes=[5, 6],
        parents_tests=[
            {"nodes": [], "expected": []},
            {"nodes": [0], "expected": [6]},
            {"nodes": [1], "expected": [0, 2, 5]},
            {"nodes": [4], "expected": [3, 5, 6]},
            {"nodes": [0, 1, 3], "expected": [0, 2, 5, 6]},
        ],
        observed_parents_tests=[
            {"nodes": [], "expected": []},
            {"nodes": [0], "expected": []},
            {"nodes": [1], "expected": [0, 2]},
            {"nodes": [4], "expected": [3]},
        ],
        ancestors_tests=[
            {"nodes": [], "expected": []},
            {"nodes": [0], "expected": [0, 3, 6]},
            {"nodes": [1], "expected": [0, 1, 2, 3, 5, 6]},
            {"nodes": [3], "expected": [3]},
        ],
        descendants_tests=[
            {"nodes": [], "expected": []},
            {"nodes": [0], "expected": [0, 1, 2]},
            {"nodes": [3], "expected": [0, 1, 2, 3, 4, 6]},
            {"nodes": [5], "expected": [1, 2, 4, 5]},
        ],
        induced_subgraph_tests=[
            {"nodes": [0, 1, 2], "expected_num_observed": 3, "expected_adj": [0, 1, 0, 0, 0, 1, 0, 1, 0]},
            {"nodes": [0, 1, 6], "expected_num_observed": 2, "expected_adj": [0, 1, 0, 0, 0, 0, 1, 0, 0]},
        ],
        trek_system_tests=[
            {
                "from_nodes": [2, 4],
                "to_nodes": [4, 3],
                "expected_exists": True,
                "expected_active": [2, 4],
            },
            {
                "from_nodes": [0, 1],
                "to_nodes": [0, 4],
                "expected_exists": True,
                "expected_active": [0, 1],
            },
            {
                "from_nodes": [0, 1],
                "to_nodes": [0, 4],
                "avoid_left_nodes": [5],
                "expected_exists": False,
                "expected_active": [0],
            },
            {
                "from_nodes": [0],
                "to_nodes": [4],
                "avoid_left_nodes": [6],
                "expected_exists": False,
                "expected_active": [],
            },
            {
                "from_nodes": [0],
                "to_nodes": [3],
                "avoid_left_nodes": [6],
                "expected_exists": False,
                "expected_active": [],
            },
            {
                "from_nodes": [0],
                "to_nodes": [4],
                "expected_exists": True,
                "expected_active": [0],
            },
        ],
        tr_from_tests=[
            # R: trFrom(2) -> c(2,4,6,8,10,12) = Python: [0,1,2,3,4,6]
            {"nodes": [0], "expected": [0, 1, 2, 3, 4, 6]},
            # R: trFrom(2, avoidLeftNodes=12) -> c(2,4,6) = Python: [0,1,2]
            {"nodes": [0], "avoid_left_nodes": [6], "expected": [0, 1, 2]},
            # R: trFrom(6) -> c(2,4,6,8,10,11,12) = Python: [0,1,2,3,4,5,6]
            {"nodes": [2], "expected": [0, 1, 2, 3, 4, 5, 6]},
            # R: trFrom(6, avoidLeftNodes=12, avoidRightNodes=4) -> c(2,4,6,10,11) = Python: [0,1,2,4,5]
            {
                "nodes": [2],
                "avoid_left_nodes": [6],
                "avoid_right_nodes": [1],
                "expected": [0, 1, 2, 4, 5],
            },
            # R: trFrom(6, avoidLeftNodes=2) -> c(4,6,10,11) = Python: [1,2,4,5]
            {"nodes": [2], "avoid_left_nodes": [0], "expected": [1, 2, 4, 5]},
            # R: trFrom(c(6,8), avoidRightNodes=12, avoidLeftNodes=c(4,11)) -> c(4,6,8,10) = Python: [1,2,3,4]
            {
                "nodes": [2, 3],
                "avoid_right_nodes": [6],
                "avoid_left_nodes": [1, 5],
                "expected": [1, 2, 3, 4],
            },
        ],
    ),
]


# Helper function for testing parameter identification
def random_identification_test(identifier, L, O, solved_parents, seed=None): # noqa: E741
    if seed is not None:
        np.random.seed(seed)

    n = L.shape[0]

    L1 = L.astype(float) * np.random.uniform(0.1, 1.0, size=(n, n))

    rand_matrix = np.random.uniform(0.1, 1.0, size=(n, n))
    O1 = (np.eye(n) + O.astype(float)) * rand_matrix
    O1 = O1 + O1.T

    I_minus_L = np.eye(n) - L1
    Sigma = np.linalg.solve(
        I_minus_L.T, O1 @ np.linalg.inv(I_minus_L)
    )

    identified = identifier(Sigma)
    L_res = identified.Lambda
    O_res = identified.Omega

    # Check that solved edges are identified (not NaN) and match the true values
    for i in range(n):
        for parent in solved_parents[i]:
            assert not np.isnan(
                L_res[parent, i]
            ), f"Lambda[{parent},{i}] should be identified but is NaN"
            diff = abs(L_res[parent, i] - L1[parent, i])
            assert diff < 1e-6, (
                f"Lambda[{parent},{i}] recovery failed: got {L_res[parent, i]}, "
                f"expected {L1[parent, i]}, diff={diff}"
            )

    # Check Omega recovery for entries that are identified
    O_diff = np.abs(O_res - O1)
    assert np.all(
        O_diff[~np.isnan(O_res)] < 1e-6
    ), f"Omega recovery failed, max diff: {np.nanmax(O_diff)}"


# Functions for generating random adjacency matrices
def random_connected(num_nodes: int, p: float) -> NDArray:
    n = num_nodes * (num_nodes - 1) // 2
    weights = uniform(size=n)

    span_tree = ig.Graph.Full(num_nodes).spanning_tree(weights, return_tree=True)

    adj = np.array(span_tree.get_adjacency().data, dtype=bool)
    upper_tri = np.triu(np.ones((num_nodes, num_nodes), dtype=bool), k=1)
    rand = np.random.random(size=(num_nodes, num_nodes)) < p

    adj |= (upper_tri & rand)
    adj |= adj.T
    return adj.astype(np.int32)

def random_directed_acyclic(num_nodes: int, p: float) -> NDArray:
    upper_tri = np.triu(np.ones((num_nodes, num_nodes), dtype=bool), k=1)
    rand = np.random.random(size=(num_nodes, num_nodes)) < p
    return (upper_tri & rand).astype(np.int32)


def random_directed(num_nodes: int, p: float) -> NDArray:
    not_diag = ~np.eye(num_nodes, dtype=bool)
    rand = np.random.random(size=(num_nodes, num_nodes)) < p
    return (not_diag & rand).astype(np.int32)

def random_undirected(num_nodes: int, p: float) -> NDArray:
    mat = (uniform(size=(num_nodes, num_nodes)) < p).astype(np.int32)
    mat = np.triu(mat, k=1)
    return mat + mat.T
