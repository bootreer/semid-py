"""Ancestral identification algorithm."""

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from semid.mixed_graph import MixedGraph
from semid.utils import CComponent

from .base import tian_sigma_for_component
from .types import GenericIDResult, IdentifierResult, IdentifyStepResult
from .core import general_generic_id


def create_ancestral_identifier(
    id_func: Callable[[NDArray], IdentifierResult],
    sources: list[int],
    targets: list[int],
    node: int,
    htr_sources: list[int],
    ancestral_subset: list[int],
    c_component: CComponent,
) -> Callable[[NDArray], IdentifierResult]:
    """
    Create an ancestral identification function.

    A helper function for `ancestralIdentifyStep`, creates an identifier function
    based on its given parameters. This created identifier function will
    identify the directed edges from 'targets' to 'node.'

    Args:
        id_func: Previous identifier function to compose with
        sources: Source nodes for the identification (active_from)
        targets: Target nodes (parents being identified)
        node: The node whose incoming edges we're identifying
        htr_sources: Sources that are half-trek reachable from node
        ancestral_subset: The ancestral set containing node
        c_component: The CComponent object with internal, incoming, top_order, L, O

    Returns:
        An identifier function
    """

    def identifier(Sigma: NDArray) -> IdentifierResult:
        # Get previously identified parameters
        identified_params = id_func(Sigma)
        Lambda = identified_params.Lambda.copy()
        Omega = identified_params.Omega.copy()

        top_order = c_component.top_order
        internal = c_component.internal
        incoming = c_component.incoming

        # Create mapping from original indices to ancestral indices
        anc_map = {node: idx for idx, node in enumerate(ancestral_subset)}

        # Map everything to ancestral indices
        top_order_anc = [anc_map[x] for x in top_order]
        sources_anc = [anc_map[x] for x in sources]
        targets_anc = [anc_map[x] for x in targets]
        htr_sources_anc = [anc_map[x] for x in htr_sources]
        internal_anc = [anc_map[x] for x in internal]
        incoming_anc = [anc_map[x] for x in incoming]
        node_anc = anc_map[node]

        Sigma_anc = Sigma[np.ix_(ancestral_subset, ancestral_subset)]
        Sigma_anc_tian = tian_sigma_for_component(
            Sigma_anc, internal_anc, incoming_anc, top_order_anc
        )

        # Create mapping for tian indices
        tian_map = {node: idx for idx, node in enumerate(top_order_anc)}

        sources_tian = [tian_map[x] for x in sources_anc]
        targets_tian = [tian_map[x] for x in targets_anc]
        htr_sources_tian = [tian_map[x] for x in htr_sources_anc]
        node_tian = tian_map[node_anc]

        Lambda_anc_tian = Lambda[np.ix_(ancestral_subset, ancestral_subset)]
        Lambda_anc_tian = Lambda_anc_tian[np.ix_(top_order_anc, top_order_anc)]

        SigmaMinus = Sigma_anc_tian.copy()
        if htr_sources_tian:
            SigmaMinus[htr_sources_tian, :] = (
                Sigma_anc_tian[htr_sources_tian, :]
                - Lambda_anc_tian[:, htr_sources_tian].T @ Sigma_anc_tian
            )

        try:
            solutions = np.linalg.solve(
                SigmaMinus[np.ix_(sources_tian, targets_tian)],
                SigmaMinus[sources_tian, node_tian],
            )

            # Map back to original indices
            for i, target_orig_idx in enumerate(targets):
                Lambda[target_orig_idx, node] = solutions[i]

        except np.linalg.LinAlgError as e:
            raise ValueError(
                "In identification, found near-singular system. Is the input matrix generic?"
            ) from e

        return IdentifierResult(Lambda, Omega)

    return identifier


# Main identification algorithm


def ancestral_identify_step(
    mixed_graph: MixedGraph,
    unsolved_parents: list[list[int]],
    solved_parents: list[list[int]],
    identifier: Callable[[NDArray], IdentifierResult],
) -> IdentifyStepResult:
    """
    Perform one iteration of ancestral identification.

    A function that does one step through all the nodes in a mixed graph
    and tries to determine if directed edge coefficients are generically
    identifiable by leveraging decomposition by ancestral subsets. See
    Algorithm 1 of Drton and Weihs (2015); this version of the algorithm
    is somewhat different from Drton and Weihs (2015) in that it also works
    on cyclic graphs.

    Args:
        `mixed_graph`: The mixed graph
        `unsolved_parents`: List of unsolved parent edges for each node
        `solved_parents`: List of solved parent edges for each node
        `identifier`: Current identifier function

    Returns:
        IdentifyStepResult with identified_edges, unsolved_parents,
        solved_parents, and identifier
    """
    identified_edges = []
    all_nodes = mixed_graph.nodes

    # Find nodes that are already solved (no unsolved parents)
    solved_nodes = [i for i in all_nodes if len(unsolved_parents[i]) == 0]

    # Cache ancestral components for each node
    ancestral_comps = {}

    for i in all_nodes:
        unsolved = unsolved_parents[i]
        if not unsolved:
            continue

        if i not in ancestral_comps:
            node_ancestors = mixed_graph.ancestors(i)
            anc_graph = mixed_graph.induced_subgraph(node_ancestors)
            try:
                tian_comp = anc_graph.tian_component(i)
                tian_graph = MixedGraph(
                    tian_comp.L,
                    tian_comp.O,
                    vertex_nums=tian_comp.top_order,
                )
                ancestral_comps[i] = {
                    "ancestors": node_ancestors,
                    "component": tian_comp,
                    "anc_graph": anc_graph,
                    "tian_graph": tian_graph,
                }
            except ValueError:
                # If Tian component fails, skip this node
                continue

        # Try first ancestral graph
        anc_info = ancestral_comps[i]
        tian_graph = anc_info["tian_graph"]

        node_parents = tian_graph.parents(i)
        htr_from_node = set(tian_graph.htr_from([i]))

        # Filter solved nodes that are in this Tian graph
        solved_in_tian = [s for s in solved_nodes if s in tian_graph.nodes]
        siblings_in_tian = tian_graph.siblings(i)

        allowed_nodes = []
        for node in tian_graph.nodes:
            if node in solved_in_tian and node not in siblings_in_tian:
                allowed_nodes.append(node)
            elif node not in htr_from_node:
                allowed_nodes.append(node)

        if len(allowed_nodes) >= len(node_parents):
            half_trek_result = tian_graph.get_half_trek_system(
                from_nodes=allowed_nodes, to_nodes=node_parents
            )

            if half_trek_result.system_exists:
                for parent in node_parents:
                    if parent in unsolved_parents[i]:
                        identified_edges.append((parent, i))

                solved_parents[i] = node_parents
                unsolved_parents[i] = []

                active_from = half_trek_result.active_from
                htr_sources = [node for node in active_from if node in htr_from_node]

                identifier = create_ancestral_identifier(
                    identifier,
                    active_from,
                    node_parents,
                    i,
                    htr_sources,
                    anc_info["ancestors"],
                    anc_info["component"],
                )

                solved_nodes.append(i)
                continue

        # Try second ancestor graph
        node_ancestors = mixed_graph.ancestors(i)
        anc_siblings = set(node_ancestors) | set(mixed_graph.siblings(node_ancestors))

        htr_from_i = set(mixed_graph.htr_from([i]))
        all_nodes_set = set(mixed_graph.nodes)
        siblings_of_i = set(mixed_graph.siblings(i))

        allowed_anc_sibs = (
            anc_siblings
            & (set(solved_nodes) | (all_nodes_set - htr_from_i)) - {i} - siblings_of_i
        )

        anc_set = sorted(set(mixed_graph.ancestors([i] + list(allowed_anc_sibs))))
        anc_graph = mixed_graph.induced_subgraph(anc_set)

        try:
            tian_comp = anc_graph.tian_component(i)
            tian_graph = MixedGraph(
                tian_comp.L,
                tian_comp.O,
                vertex_nums=tian_comp.top_order,
            )

            htr_from = set(anc_graph.htr_from([i]))
            siblings = set(anc_graph.siblings(i))
            solved_in = [s for s in solved_nodes if s in anc_graph.nodes]

            allowed_nodes = set(solved_in) - siblings
            allowed_nodes |= set(anc_graph.nodes) - htr_from
            allowed_nodes_larger = sorted(allowed_nodes & set(tian_graph.nodes))

            if len(allowed_nodes_larger) >= len(node_parents):
                half_trek_result = tian_graph.get_half_trek_system(
                    from_nodes=allowed_nodes_larger, to_nodes=node_parents
                )

                if half_trek_result.system_exists:
                    # All parents of i are now identified
                    for parent in node_parents:
                        if parent in unsolved_parents[i]:
                            identified_edges.append((parent, i))

                    solved_parents[i] = node_parents
                    unsolved_parents[i] = []

                    active_from = half_trek_result.active_from
                    htr_sources = [node for node in active_from if node in htr_from]

                    identifier = create_ancestral_identifier(
                        identifier,
                        active_from,
                        node_parents,
                        i,
                        htr_sources,
                        anc_set,
                        tian_comp,
                    )

                    solved_nodes.append(i)

        except ValueError:
            pass

    return IdentifyStepResult(
        identified_edges,
        unsolved_parents,
        solved_parents,
        identifier,
    )


def ancestral_id(
    mixed_graph: MixedGraph,
    tian_decompose: bool = True,
) -> GenericIDResult:
    """
    Determines which edges in a mixed graph are ancestralID-identifiable

    Uses the an identification criterion of Drton and Weihs (2015); this version
    of the algorithm is somewhat different from Drton and Weihs (2015) in that it
    also works on cyclic graphs.

    Args:
        `mixed_graph`: The mixed graph to analyze
        `tian_decompose`: Whether to use Tian decomposition (default True)

    Returns:
        GenericIDResult with identified edges and identifier function
    """
    return general_generic_id(mixed_graph, [ancestral_identify_step], tian_decompose)
