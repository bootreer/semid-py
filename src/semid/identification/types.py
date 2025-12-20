"""Data types for identification results."""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from semid.mixed_graph import MixedGraph
from semid.utils import IdentifierResult


@dataclass
class GenericIDResult:
    """
    Result from generic identification algorithms.

    Attributes:
        `solved_parents`: List of solved parent nodes for each node
        `unsolved_parents`: List of unsolved parent nodes for each node
        `solved_siblings`: List of solved sibling nodes for each node
        `unsolved_siblings`: List of unsolved sibling nodes for each node
        `identifier`: Function that takes covariance matrix and returns identified parameters
        `mixed_graph`: The input mixed graph
        `tian_decompose`: Whether Tian decomposition was used
    """

    solved_parents: list[list[int]]
    unsolved_parents: list[list[int]]
    solved_siblings: list[list[int]]
    unsolved_siblings: list[list[int]]
    identifier: Callable[[NDArray], IdentifierResult]
    mixed_graph: MixedGraph
    tian_decompose: bool = False

    def __str__(self) -> str:
        """Pretty print the result."""
        n = self.mixed_graph.num_nodes
        num_dir_edges = np.sum(self.mixed_graph.d_adj)
        num_bi_edges = np.sum(self.mixed_graph.b_adj) // 2

        num_solved_dir = sum(len(parents) for parents in self.solved_parents)
        num_solved_bi = sum(len(siblings) for siblings in self.solved_siblings) // 2

        lines = [
            "Generic Identifiability Result",
            "=" * 40,
            f"Mixed Graph: {n} nodes, {num_dir_edges} directed edges, {num_bi_edges} bidirected edges",
            f"Tian decomposition: {self.tian_decompose}",
            "",
            "Identification Summary:",
            f"  Directed edges identified: {num_solved_dir}/{num_dir_edges}",
            f"  Bidirected edges identified: {num_solved_bi}/{num_bi_edges}",
            "",
        ]

        dir_edges: list[str] = []
        if num_solved_dir > 0:
            lines.append("Identified directed edges:")
            count = 0
            for i in range(n):
                for parent in self.solved_parents[i]:
                    if count < 10:
                        dir_edges.append(f"{parent} -> {i}")
                        count += 1
                    else:
                        dir_edges.append("...")
                        break
        else:
            dir_edges.append("None")

        dir_str = ", ".join(dir_edges)
        lines.append(f"  {dir_str}")
        lines.append("")

        if num_solved_dir < num_dir_edges:
            num_unsolved = num_dir_edges - num_solved_dir
            lines.append(f"Unidentified directed edges: {num_unsolved}")

        bidir: list[str] = []
        if num_solved_bi > 0:
            lines.append("Identified bidir. edges:")
            count = 0
            for i in range(n):
                for sibling in filter(lambda j: i < j, self.solved_siblings[i]):
                    if count < 10:
                        bidir.append(f"{i} <-> {sibling}")
                        count += 1
                    else:
                        bidir.append("...")
                        break
        else:
            bidir.append("None")

        bidir_str = ", ".join(bidir)
        lines.append(f"  {bidir_str}")
        lines.append("")

        if num_solved_dir < num_dir_edges:
            num_unsolved = num_bi_edges - num_solved_bi
            lines.append(f"Unidentified bidir. edges: {num_unsolved}")

        return "\n".join(lines)


@dataclass
class SEMIDResult:
    """
    Complete SEMID result including global and generic identifiability.

    Attributes:
        `is_global_id`: Whether the graph is globally identifiable (None if not tested)
        `is_generic_non_id`: Whether generically non-identifiable (None if not tested)
        `generic_id_result`: GenericIDResult from identification algorithms
        `mixed_graph`: The input mixed graph
        `tian_decompose`: Whether Tian decomposition was used
    """

    is_global_id: Optional[bool]
    is_generic_non_id: Optional[bool]
    generic_id_result: Optional[GenericIDResult]
    mixed_graph: MixedGraph
    tian_decompose: bool = False

    def __str__(self) -> str:
        """Pretty print the result."""
        lines = [
            "SEMID Result",
            "=" * 40,
            f"Tian decomposition: {self.tian_decompose}",
            "",
        ]

        if self.is_global_id is not None:
            lines.append(f"Globally identifiable: {self.is_global_id}")
            lines.append("")

        if self.is_generic_non_id is not None:
            if self.is_generic_non_id:
                status = "TRUE (infinite-to-one parameterization exists)"
            elif (
                self.generic_id_result
                and sum(len(p) for p in self.generic_id_result.unsolved_parents) == 0
            ):
                status = "FALSE (all parameters identified)"
            else:
                status = "INCONCLUSIVE"
            lines.append(f"Generically non-identifiable: {status}")
            lines.append("")

        if self.generic_id_result:
            lines.append(str(self.generic_id_result))

        return "\n".join(lines)
