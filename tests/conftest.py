from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy._typing import NDArray

from semid_py import LatentDigraph, MixedGraph


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
]


@dataclass
class LatentDigraphTestCase:
    """Test case for LatentDigraph"""

    L: list[int]
    """Adjacency matrix as flattened list"""

    dim: tuple[int, int]
    """Dimensions of the adjacency matrix"""

    num_observed: Optional[int] = None
    """Number of observed nodes (None means all observed)"""

    # Expected results for trek system tests
    trek_system_tests: list[dict] = None
    """List of trek system test cases with from_nodes, to_nodes, and expected results"""

    # Expected results for tr_from tests
    tr_from_tests: list[dict] = None
    """List of tr_from test cases"""

    def __post_init__(self):
        if self.trek_system_tests is None:
            self.trek_system_tests = []
        if self.tr_from_tests is None:
            self.tr_from_tests = []

    def to_latent_digraph(self) -> LatentDigraph:
        L = np.reshape(self.L, self.dim).astype(np.int32)
        return LatentDigraph(L, num_observed=self.num_observed)


# LatentDigraph test cases ported from R tests
# fmt: off
LATENT_DIGRAPH_EXAMPLES: list[LatentDigraphTestCase] = [
    # Single node graph
    LatentDigraphTestCase(
        L=[0],
        dim=(1, 1),
        num_observed=1,
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
        L=[0, 1, 0, 0],
        dim=(2, 2),
        num_observed=2,
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
        L=[0, 1, 0, 0, 0, 0, 1, 1, 0],
        dim=(3, 3),
        num_observed=2,
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
    # Edges: 0->1, 1->2, 2->1, 3->4, 5->0, 5->2, 5->4, 6->0, 6->1, 6->4
    LatentDigraphTestCase(
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
