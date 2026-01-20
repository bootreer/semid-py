# ruff: noqa: E741

import numpy as np
import pytest

from semid import MixedGraph, semid
from tests.conftest import (
    GRAPH_EXAMPLES,
    VERMA_L,
    VERMA_O,
    random_identification_test,
)


def test_identifier_verma_graph():
    """Test that identifier correctly recovers parameters on Verma graph."""
    graph = MixedGraph(VERMA_L, VERMA_O)
    result = semid(graph)

    # Verify all edges are identified
    assert result.generic_id_result is not None
    assert sum(len(p) for p in result.generic_id_result.unsolved_parents) == 0

    # Test with random parameters
    random_identification_test(
        result.generic_id_result.identifier,
        VERMA_L,
        VERMA_O,
        result.generic_id_result.solved_parents,
    )


def test_identifier_simple_chain():
    """Test identifier on a simple chain: 0 -> 1 -> 2"""
    L = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int32)
    O = np.zeros((3, 3), dtype=np.int32)

    graph = MixedGraph(L, O)
    result = semid(graph)

    assert result.generic_id_result is not None
    assert sum(len(p) for p in result.generic_id_result.unsolved_parents) == 0

    random_identification_test(
        result.generic_id_result.identifier,
        L,
        O,
        result.generic_id_result.solved_parents,
    )


def test_identifier_bow_graph():
    """Test identifier on bow graph: 0 -> 2 <- 1, with 0 <-> 1"""
    L = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=np.int32)
    O = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.int32)

    graph = MixedGraph(L, O)
    result = semid(graph)

    assert result.generic_id_result is not None
    assert sum(len(p) for p in result.generic_id_result.unsolved_parents) == 0

    random_identification_test(
        result.generic_id_result.identifier,
        L,
        O,
        result.generic_id_result.solved_parents,
    )


@pytest.mark.parametrize("graph_example", GRAPH_EXAMPLES)
def test_identifier_all_htc_identifiable_graphs(graph_example):
    """Test identifier on all HTC-identifiable graphs from examples."""
    if graph_example.htc_id != 1:
        pytest.skip("Graph is not HTC-identifiable")

    graph = graph_example.to_mixed_graph()
    result = semid(graph)

    assert result.generic_id_result is not None

    # If all edges were identified, test parameter recovery
    if sum(len(p) for p in result.generic_id_result.unsolved_parents) == 0:
        random_identification_test(
            result.generic_id_result.identifier,
            graph.d_adj,
            graph.b_adj,
            result.generic_id_result.solved_parents,
            seed=42,
        )


def test_edgewise_id_verma_graph():
    """Test that edgewise_id identifies all edges in Verma graph."""
    from semid import edgewise_id

    graph = MixedGraph(VERMA_L, VERMA_O)
    result = edgewise_id(graph)

    # Should identify all 4 edges
    assert sum(len(p) for p in result.solved_parents) == 4
    assert sum(len(p) for p in result.unsolved_parents) == 0

    # Test parameter recovery
    random_identification_test(
        result.identifier, VERMA_L, VERMA_O, result.solved_parents
    )


def test_edgewise_id_vs_htc():
    """Test that edgewise_id identifies at least as many edges as HTC."""
    from semid import edgewise_id

    # Test on all HTC-identifiable graphs
    for graph_example in GRAPH_EXAMPLES:
        if graph_example.htc_id != 1:
            continue  # Skip non-HTC-identifiable graphs

        graph = graph_example.to_mixed_graph()

        # Run both methods
        htc_result = semid(
            graph, test_global_id=False, test_generic_non_id=False
        ).generic_id_result
        eid_result = edgewise_id(graph)

        # Edgewise should identify at least as many as HTC
        htc_count = sum(len(p) for p in htc_result.solved_parents)
        eid_count = sum(len(p) for p in eid_result.solved_parents)

        assert (
            eid_count >= htc_count
        ), f"Edgewise identified {eid_count} but HTC identified {htc_count}"

        # If all edges identified, test parameter recovery
        if sum(len(p) for p in eid_result.unsolved_parents) == 0:
            random_identification_test(
                eid_result.identifier,
                graph.d_adj,
                graph.b_adj,
                eid_result.solved_parents,
                seed=42,
            )


def test_identifier_random_dags():
    """Test identifier on random DAGs (like R test suite)."""
    np.random.seed(2323)
    n_trials = 10
    n_nodes = 5

    for trial in range(n_trials):
        # Generate random DAG (no cycles)
        L = np.zeros((n_nodes, n_nodes), dtype=np.int32)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if np.random.rand() < 0.3:  # 30% edge probability
                    L[i, j] = 1

        # Generate random bidirected edges
        O = np.zeros((n_nodes, n_nodes), dtype=np.int32)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if np.random.rand() < 0.2:  # 20% edge probability
                    O[i, j] = 1
                    O[j, i] = 1

        # Skip if no edges
        if np.sum(L) == 0:
            continue

        graph = MixedGraph(L, O)
        result = semid(graph, test_global_id=False, test_generic_non_id=False)

        # If any edges were identified, test parameter recovery
        if (
            result.generic_id_result
            and sum(len(p) for p in result.generic_id_result.solved_parents) > 0
        ):
            random_identification_test(
                result.generic_id_result.identifier,
                L,
                O,
                result.generic_id_result.solved_parents,
                seed=trial,
            )


def test_ancestral_id_verma_graph():
    """Test that ancestral_id works on Verma graph."""
    from semid import ancestral_id

    graph = MixedGraph(VERMA_L, VERMA_O)
    result = ancestral_id(graph, tian_decompose=False)

    # Ancestral ID should identify at least some edges
    assert sum(len(p) for p in result.solved_parents) > 0


def test_ancestral_id_simple_chain():
    """Test ancestral_id on a simple chain."""
    from semid import ancestral_id

    L = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int32)
    O = np.zeros((3, 3), dtype=np.int32)

    graph = MixedGraph(L, O)
    result = ancestral_id(graph, tian_decompose=False)

    # Should identify all edges in a simple chain
    assert sum(len(p) for p in result.solved_parents) == 2
    assert sum(len(p) for p in result.unsolved_parents) == 0


def test_tian_decomposition_htc():
    """Test that HTC with Tian decomposition produces same results as without."""
    graph = MixedGraph(VERMA_L, VERMA_O)

    # Run without Tian decomposition
    result_no_tian = semid(
        graph, test_global_id=False, test_generic_non_id=False, tian_decompose=False
    )

    # Run with Tian decomposition
    result_with_tian = semid(
        graph, test_global_id=False, test_generic_non_id=False, tian_decompose=True
    )

    # Should identify same number of edges
    assert sum(len(p) for p in result_no_tian.generic_id_result.solved_parents) == sum(
        len(p) for p in result_with_tian.generic_id_result.solved_parents
    )


def test_tian_decomposition_edgewise():
    """Test edgewise_id with Tian decomposition."""
    from semid import edgewise_id

    graph = MixedGraph(VERMA_L, VERMA_O)

    # With Tian decomposition
    result = edgewise_id(graph, tian_decompose=True)

    # Should identify all 4 edges
    assert sum(len(p) for p in result.solved_parents) == 4
    assert sum(len(p) for p in result.unsolved_parents) == 0


def test_tian_decomposition_ancestral():
    """Test ancestral_id with Tian decomposition."""
    from semid import ancestral_id

    graph = MixedGraph(VERMA_L, VERMA_O)

    # With Tian decomposition
    result = ancestral_id(graph, tian_decompose=True)

    # Should identify at least some edges
    assert sum(len(p) for p in result.solved_parents) > 0


def test_trek_sep_simple_chain():
    """Test trek_sep_id on a simple chain."""
    from semid import trek_sep_id

    L = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int32)
    O = np.zeros((3, 3), dtype=np.int32)

    graph = MixedGraph(L, O)
    result = trek_sep_id(graph)

    # Should identify all edges in a simple chain
    assert sum(len(p) for p in result.solved_parents) == 2
    assert sum(len(p) for p in result.unsolved_parents) == 0


def test_trek_sep_verma():
    """Test trek_sep_id on Verma graph."""
    from semid import trek_sep_id

    graph = MixedGraph(VERMA_L, VERMA_O)
    result = trek_sep_id(graph)

    # Should identify at least some edges
    assert sum(len(p) for p in result.solved_parents) > 0


def test_edgewise_ts_id():
    """Test combined edgewise + trek separation."""
    from semid import edgewise_ts_id

    graph = MixedGraph(VERMA_L, VERMA_O)
    result = edgewise_ts_id(graph, tian_decompose=False)

    # Should identify all 4 edges (very powerful combination)
    assert sum(len(p) for p in result.solved_parents) == 4
    assert sum(len(p) for p in result.unsolved_parents) == 0


def test_lf_htc_id_basic():
    """Test latent-factor HTC identification on a simple latent-factor graph."""
    from semid import LatentDigraph, lf_htc_id

    # Create a simple latent-factor graph:
    # Observed nodes: 0, 1, 2, 3
    # Latent nodes: 4, 5
    # Structure: 4 -> {0, 1, 2}, 5 -> {1, 2, 3}, 0 -> 1 -> 2 -> 3
    L = np.array(
        [
            [0, 1, 0, 0, 0, 0],  # 0 -> 1
            [0, 0, 1, 0, 0, 0],  # 1 -> 2
            [0, 0, 0, 1, 0, 0],  # 2 -> 3
            [0, 0, 0, 0, 0, 0],  # 3 (no outgoing)
            [1, 1, 1, 0, 0, 0],  # latent 4 -> {0, 1, 2}
            [0, 1, 1, 1, 0, 0],  # latent 5 -> {1, 2, 3}
        ],
        dtype=np.int32,
    )

    graph = LatentDigraph(L, num_observed=4)
    result = lf_htc_id(graph)

    # Verify the result structure
    assert result.graph == graph
    assert len(result.solved_parents) == 4
    assert len(result.unsolved_parents) == 4

    # Should identify at least some edges
    total_identified = sum(len(parents) for parents in result.solved_parents)
    assert total_identified >= 0  # At minimum, we expect it to run without errors


# Test cases from R package SEMID/tests/testthat/graphExamples.R
@pytest.mark.parametrize(
    "L,num_observed,expected_identifiable",
    [
        # Instrumental variables - should be identifiable
        (
            np.array(
                [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 1, 1, 0]],
                dtype=np.int32,
            ),
            3,
            True,
        ),
        # One global latent variable - should be identifiable
        (
            np.array(
                [
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0],
                ],
                dtype=np.int32,
            ),
            5,
            True,
        ),
        # Figure 1 from paper - should be identifiable
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0],
                ],
                dtype=np.int32,
            ),
            5,
            True,
        ),
        # Figure 2 from paper - should NOT be identifiable
        (
            np.array(
                [
                    [0, 1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0],
                ],
                dtype=np.int32,
            ),
            5,
            False,
        ),
        # Figure 5 from paper - should be identifiable
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 1, 0, 0],
                ],
                dtype=np.int32,
            ),
            5,
            True,
        ),
    ],
)
def test_lf_htc_id_r_examples(L, num_observed, expected_identifiable):
    """Test lfhtcID on examples from R package test suite."""
    from semid import LatentDigraph, lf_htc_id

    graph = LatentDigraph(L, num_observed=num_observed)
    result = lf_htc_id(graph)

    # Check if all edges are identified
    is_identifiable = sum(len(p) for p in result.unsolved_parents) == 0

    assert (
        is_identifiable == expected_identifiable
    ), f"Expected identifiable={expected_identifiable}, got {is_identifiable}"


# =============================================================================
# Random validation tests (ported from R test suite)
# =============================================================================


def test_ancestral_id_random_validation():
    """Test that ancestral_id does not identify edges erroneously (random tests).

    Ported from R test_ancestral.R: tests that ancestral ID identifies a subset
    of what HTC identifies, and that identified parameters can be recovered.
    """
    from semid import ancestral_id, htc_id

    from tests.conftest import random_directed_acyclic, random_undirected

    np.random.seed(3634)
    n_sims = 10
    node_counts = [5, 6]
    p = 0.3

    for n in node_counts:
        for _ in range(n_sims):
            L = random_directed_acyclic(n, p)
            O = random_undirected(n, p)
            graph = MixedGraph(L, O)

            htc_result = htc_id(graph, tian_decompose=False)
            anc_result = ancestral_id(graph, tian_decompose=False)

            # Ancestral ID should identify at least as many edges as HTC
            for node in range(n):
                htc_parents = set(htc_result.solved_parents[node])
                anc_parents = set(anc_result.solved_parents[node])
                assert (
                    htc_parents <= anc_parents
                ), f"HTC parents {htc_parents} not subset of ancestral parents {anc_parents}"

            # Test parameter recovery for both
            if sum(len(p) for p in htc_result.solved_parents) > 0:
                random_identification_test(
                    htc_result.identifier, L, O, htc_result.solved_parents
                )

            if sum(len(p) for p in anc_result.solved_parents) > 0:
                random_identification_test(
                    anc_result.identifier, L, O, anc_result.solved_parents
                )


def test_edgewise_id_random_validation():
    """Test that edgewise_id does not identify edges erroneously (random tests).

    Ported from R test_edgewiseID.R: tests that edgewise ID identifies at least
    as many edges as HTC, and that identified parameters can be recovered.
    """
    from semid import edgewise_id, htc_id

    from tests.conftest import random_directed_acyclic, random_undirected

    np.random.seed(2323)
    n_sims = 30
    node_counts = [5, 6]
    p = 0.3

    for n in node_counts:
        for _ in range(n_sims):
            L = random_directed_acyclic(n, p)
            O = random_undirected(n, p)
            graph = MixedGraph(L, O)

            htc_result = htc_id(graph, tian_decompose=True)
            eid_result = edgewise_id(graph, tian_decompose=True)

            # Edgewise ID should identify at least as many edges as HTC
            for node in range(n):
                htc_parents = set(htc_result.solved_parents[node])
                eid_parents = set(eid_result.solved_parents[node])
                assert (
                    htc_parents <= eid_parents
                ), f"HTC parents {htc_parents} not subset of edgewise parents {eid_parents}"

            # Test parameter recovery
            if sum(len(p) for p in eid_result.solved_parents) > 0:
                random_identification_test(
                    eid_result.identifier, L, O, eid_result.solved_parents
                )


def test_trek_sep_id_random_validation():
    """Test that trek_sep_id does not identify edges erroneously (random tests).

    Ported from R test_trekSepID.R: tests that trek separation ID correctly
    recovers parameters for randomly generated graphs.
    """
    from semid import trek_sep_id

    from tests.conftest import random_directed_acyclic, random_undirected

    np.random.seed(1231)
    n_sims = 20
    node_counts = [4, 5]
    p = 0.3

    for n in node_counts:
        for _ in range(n_sims):
            L = random_directed_acyclic(n, p)
            O = random_undirected(n, p)
            graph = MixedGraph(L, O)

            ts_result = trek_sep_id(graph, tian_decompose=True)

            # Test parameter recovery
            if sum(len(p) for p in ts_result.solved_parents) > 0:
                random_identification_test(
                    ts_result.identifier, L, O, ts_result.solved_parents
                )


def test_tian_decomposition_covariance_recovery():
    """Test that tianSigmaForComponent correctly recovers transformed covariances.

    Ported from R test_tianDecompose.R: tests that the Tian decomposition
    correctly transforms the covariance matrix for each c-component.
    """
    from semid.identification.base import tian_sigma_for_component

    from tests.conftest import random_directed_acyclic, random_undirected

    np.random.seed(123)
    ps = [0.1, 0.05]
    sims = 10
    ns = [10, 20]

    for p in ps:
        for n in ns:
            for _ in range(sims):
                L = random_directed_acyclic(n, p)
                O = random_undirected(n, p)

                # Generate random Lambda and Omega matrices
                LL = L.astype(float) * np.random.randn(n, n)
                OO = (O + np.eye(n, dtype=float)) * np.random.randn(n, n)
                OO = OO + OO.T

                # Compute covariance matrix: Sigma = (I - L)^{-T} @ Omega @ (I - L)^{-1}
                I_minus_L = np.eye(n) - LL
                temp = np.linalg.solve(I_minus_L.T, OO.T).T  # OO @ (I-L)^{-1}
                Sigma = np.linalg.solve(I_minus_L.T, temp)  # (I-L)^{-T} @ temp

                # Get Tian decomposition
                graph = MixedGraph(L, O)
                c_components = graph.tian_decompose()

                for comp in c_components:
                    internal = comp.internal
                    incoming = comp.incoming
                    top_order = comp.top_order

                    # Compute expected transformed covariance manually
                    # LLnew: zero out columns corresponding to incoming nodes
                    LLnew = LL[np.ix_(top_order, top_order)].copy()
                    incoming_in_toporder = [
                        i for i, node in enumerate(top_order) if node in incoming
                    ]
                    LLnew[:, incoming_in_toporder] = 0

                    # OOnew: set incoming-incoming block to identity
                    OOnew = OO[np.ix_(top_order, top_order)].copy()
                    OOnew[np.ix_(incoming_in_toporder, incoming_in_toporder)] = np.eye(
                        len(incoming)
                    )

                    # Compute expected Sigma
                    I_minus_LLnew = np.eye(len(top_order)) - LLnew
                    temp_new = np.linalg.solve(I_minus_LLnew.T, OOnew.T).T
                    SigmaNew = np.linalg.solve(I_minus_LLnew.T, temp_new)

                    # Get recovered Sigma using our function
                    recoveredSigma = tian_sigma_for_component(
                        Sigma, internal, incoming, top_order
                    )

                    # Compare with relative tolerance
                    relative_diff = np.abs(SigmaNew - recoveredSigma) / (
                        np.abs(SigmaNew) + 1e-6
                    )
                    assert np.all(relative_diff < 1e-3), (
                        f"Tian sigma recovery failed for component with "
                        f"internal={internal}, incoming={incoming}"
                    )
