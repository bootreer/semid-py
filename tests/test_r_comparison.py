"""
Tests comparing Python implementation against R SEMID package.

These tests require R and the SEMID package to be installed.
They are skipped by default - run with: uv run pytest -m r_comparison

To install R dependencies:
    install.packages("SEMID")
    # Or from GitHub: devtools::install_github("Lucaweihs/SEMID")
"""

# ruff: noqa: E741

import pytest
import numpy as np
from numpy.typing import NDArray

# Try to import rpy2 - tests will be skipped if not available
try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr, isinstalled
    from rpy2.robjects.conversion import localconverter

    # Create numpy converter
    numpy_cv = ro.default_converter + numpy2ri.converter

    # Check if R and SEMID are available
    HAS_R_SEMID = isinstalled("SEMID")
except ImportError:
    HAS_R_SEMID = False
    ro = None
    numpy_cv = None
    localconverter = None

from semid import (
    MixedGraph,
    LatentDigraph,
    htc_id,
    edgewise_id,
    trek_sep_id,
    ancestral_id,
    lf_htc_id,
    semid,
    edgewise_ts_id,
)
from semid.identification import lsc_id

pytestmark = [
    pytest.mark.r_comparison,
    pytest.mark.skipif(not HAS_R_SEMID, reason="R or SEMID package not available"),
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def r_semid():
    """Import and return the R SEMID package."""
    with localconverter(numpy_cv):
        return importr("SEMID")


@pytest.fixture
def r_converter():
    """Return the numpy converter for R interop."""
    return numpy_cv


# =============================================================================
# Utility Functions
# =============================================================================


def r_to_py_index(r_index: int) -> int:
    """Convert R 1-based index to Python 0-based index."""
    return r_index - 1


def py_to_r_index(py_index: int) -> int:
    """Convert Python 0-based index to R 1-based index."""
    return py_index + 1


def r_list_to_py_list(r_list) -> list:
    """Convert R list/vector to Python list."""
    if r_list is None or (hasattr(r_list, "__len__") and len(r_list) == 0):
        return []
    return [int(x) for x in r_list]


def get_r_list_element(r_obj, key: str):
    """
    Robustly get element from R object (ListVector or NamedList).

    Handles differences between raw R objects and converted Python objects.
    """
    # If it's an R object (has rx2)
    if hasattr(r_obj, "rx2"):
        try:
            return r_obj.rx2(key)
        except (ValueError, LookupError):
            pass

    # If it's a NamedList (from rpy2 conversion)
    if hasattr(r_obj, "names"):
        try:
            names_attr = r_obj.names
            if callable(names_attr):
                names = list(names_attr())
            else:
                names = list(names_attr)

            if key in names:
                idx = names.index(key)
                return r_obj[idx]
        except (ValueError, AttributeError, TypeError):
            pass

    # Try dictionary access
    try:
        return r_obj[key]
    except (TypeError, KeyError, IndexError):
        pass

    raise KeyError(f"Key '{key}' not found in R object: {type(r_obj)}")


def extract_r_solved_edges(r_result, n_nodes: int) -> set[tuple[int, int]]:
    """
    Extract solved parent edges from R result.

    R returns solvedParents as a list where solvedParents[[i]] contains
    the parents of node i that have been solved.
    """
    solved_edges = set()
    solved_parents = get_r_list_element(r_result, "solvedParents")

    for child_r in range(1, n_nodes + 1):
        if hasattr(solved_parents, "rx2"):
            parents_r = solved_parents.rx2(child_r)
        else:
            parents_r = solved_parents[child_r - 1]

        if (
            parents_r is not None
            and hasattr(parents_r, "__len__")
            and len(parents_r) > 0
        ):
            for parent_r in parents_r:
                parent_py = r_to_py_index(int(parent_r))
                child_py = r_to_py_index(child_r)
                solved_edges.add((parent_py, child_py))

    return solved_edges


def extract_py_solved_edges(py_result) -> set[tuple[int, int]]:
    """
    Extract solved parent edges from Python result.

    Python returns solved_parents as a list where solved_parents[i] contains
    the parents of node i that have been solved.
    """
    solved_edges = set()
    solved_parents = py_result.solved_parents

    for child, parents in enumerate(solved_parents):
        for parent in parents:
            solved_edges.add((parent, child))

    return solved_edges


def extract_r_unsolved_edges(r_result, n_nodes: int) -> set[tuple[int, int]]:
    """Extract unsolved parent edges from R result."""
    unsolved_edges = set()
    unsolved_parents = get_r_list_element(r_result, "unsolvedParents")

    for child_r in range(1, n_nodes + 1):
        if hasattr(unsolved_parents, "rx2"):
            parents_r = unsolved_parents.rx2(child_r)
        else:
            parents_r = unsolved_parents[child_r - 1]

        if (
            parents_r is not None
            and hasattr(parents_r, "__len__")
            and len(parents_r) > 0
        ):
            for parent_r in parents_r:
                parent_py = r_to_py_index(int(parent_r))
                child_py = r_to_py_index(child_r)
                unsolved_edges.add((parent_py, child_py))

    return unsolved_edges


def extract_py_unsolved_edges(py_result) -> set[tuple[int, int]]:
    """
    Extract unsolved parent edges from Python result.

    Python returns unsolved_parents as a list where unsolved_parents[i] contains
    the parents of node i that are unsolved.
    """
    unsolved_edges = set()
    unsolved_parents = py_result.unsolved_parents

    for child, parents in enumerate(unsolved_parents):
        for parent in parents:
            unsolved_edges.add((parent, child))

    return unsolved_edges


def generate_random_covariance(
    n: int, seed: int, L: NDArray | None = None, O: NDArray | None = None
) -> NDArray:
    """
    Generate a random valid covariance matrix.

    If L and O are provided, generates a covariance matrix consistent with the graph.
    Otherwise generates a generic covariance matrix.
    """
    np.random.seed(seed)

    if L is not None and O is not None:
        # Generate random weights for Lambda (L)
        Lambda = np.zeros_like(L)
        Lambda[L != 0] = np.random.uniform(
            0.5, 1.5, size=np.sum(L != 0)
        ) * np.random.choice([-1, 1], size=np.sum(L != 0))

        # Generate random weights for Omega (O)
        # Construct symmetric Omega
        Omega = np.zeros_like(O)
        # Diagonal elements (variances)
        np.fill_diagonal(Omega, np.random.uniform(1.0, 2.0, size=n))

        # Off-diagonal elements
        rows, cols = np.where(np.triu(O, k=1))
        for r, c in zip(rows, cols):
            val = np.random.uniform(0.2, 0.5) * np.random.choice([-1, 1])
            Omega[r, c] = Omega[c, r] = val

        # Ensure positive definiteness of Omega (add to diagonal if needed)
        min_eig = np.min(np.linalg.eigvals(Omega))
        if min_eig <= 0:
            np.fill_diagonal(Omega, np.diag(Omega) + abs(min_eig) + 0.1)

        # Calculate Sigma = (I - Lambda.T)^-1 Omega (I - Lambda.T)^-T
        I = np.eye(n)
        try:
            I_minus_L_T_inv = np.linalg.inv(I - Lambda.T)
            Sigma = I_minus_L_T_inv @ Omega @ I_minus_L_T_inv.T
            return Sigma
        except np.linalg.LinAlgError:
            # Fallback if singular (unlikely with random weights)
            pass

    A = np.random.randn(n, n)
    return A @ A.T + np.eye(n)


def compare_identifier_outputs(
    py_identifier,
    r_identifier,
    n_nodes: int,
    converter,
    n_tests: int = 5,
    rtol: float = 1e-9,
    atol: float = 1e-9,
    L: NDArray | None = None,
    O: NDArray | None = None,
):
    """
    Compare Python and R identifier function outputs on random covariance matrices.
    """
    for seed in range(n_tests):
        Sigma = generate_random_covariance(n_nodes, seed, L, O)

        # Python result
        py_result = py_identifier(Sigma)
        py_Lambda = py_result.Lambda
        py_Omega = py_result.Omega

        # R result
        with localconverter(converter):
            r_result = r_identifier(Sigma)
            r_Lambda = np.array(get_r_list_element(r_result, "Lambda"))
            r_Omega = np.array(get_r_list_element(r_result, "Omega"))

        # Compare Lambda (handling NaN positions)
        py_Lambda_masked = np.where(np.isnan(py_Lambda), 0, py_Lambda)
        r_Lambda_masked = np.where(np.isnan(r_Lambda), 0, r_Lambda)
        py_Lambda_nan_mask = np.isnan(py_Lambda)
        r_Lambda_nan_mask = np.isnan(r_Lambda)

        np.testing.assert_array_equal(
            py_Lambda_nan_mask,
            r_Lambda_nan_mask,
            err_msg=f"Lambda NaN positions differ for seed {seed}",
        )
        np.testing.assert_allclose(
            py_Lambda_masked,
            r_Lambda_masked,
            rtol=rtol,
            atol=atol,
            err_msg=f"Lambda values differ for seed {seed}",
        )

        # Compare Omega (handling NaN positions)
        py_Omega_masked = np.where(np.isnan(py_Omega), 0, py_Omega)
        r_Omega_masked = np.where(np.isnan(r_Omega), 0, r_Omega)
        py_Omega_nan_mask = np.isnan(py_Omega)
        r_Omega_nan_mask = np.isnan(r_Omega)

        np.testing.assert_array_equal(
            py_Omega_nan_mask,
            r_Omega_nan_mask,
            err_msg=f"Omega NaN positions differ for seed {seed}",
        )
        np.testing.assert_allclose(
            py_Omega_masked,
            r_Omega_masked,
            rtol=rtol,
            atol=atol,
            err_msg=f"Omega values differ for seed {seed}",
        )


# =============================================================================
# Test Graph Definitions
# =============================================================================

# Standard test graphs covering various structures
TEST_GRAPHS = {
    "empty_2": {
        "L": np.array([[0, 0], [0, 0]], dtype=float),
        "O": np.array([[0, 0], [0, 0]], dtype=float),
    },
    "single_edge": {
        "L": np.array([[0, 1], [0, 0]], dtype=float),
        "O": np.array([[0, 0], [0, 0]], dtype=float),
    },
    "chain_3": {
        "L": np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float),
        "O": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float),
    },
    "fork_3": {
        "L": np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=float),
        "O": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float),
    },
    "collider_3": {
        "L": np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=float),
        "O": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float),
    },
    "chain_3_confounded": {
        "L": np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float),
        "O": np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=float),
    },
    "bow_graph": {
        "L": np.array([[0, 1], [0, 0]], dtype=float),
        "O": np.array([[0, 1], [1, 0]], dtype=float),
    },
    "diamond_4": {
        "L": np.array(
            [[0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=float
        ),
        "O": np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=float
        ),
    },
    "diamond_4_confounded": {
        "L": np.array(
            [[0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=float
        ),
        "O": np.array(
            [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=float
        ),
    },
    "full_bidir_3": {
        "L": np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float),
        "O": np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float),
    },
    "complex_5": {
        "L": np.array(
            [
                [0, 1, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ],
            dtype=float,
        ),
        "O": np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=float,
        ),
    },
}


def get_test_graph_ids():
    """Get list of test graph IDs for parametrization."""
    return list(TEST_GRAPHS.keys())


# =============================================================================
# MixedGraph Method Tests
# =============================================================================


class TestMixedGraphMethods:
    """Test that MixedGraph methods match R implementation."""

    @pytest.mark.parametrize("graph_id", get_test_graph_ids())
    def test_parents(self, graph_id, r_semid, r_converter):
        """Test parents() method matches R."""
        graph_data = TEST_GRAPHS[graph_id]
        L, O = graph_data["L"], graph_data["O"]
        n = L.shape[0]

        py_graph = MixedGraph(L, O)

        with localconverter(r_converter):
            r_graph = r_semid.MixedGraph(L, O)

            for node in range(n):
                py_parents = set(py_graph.parents([node]))
                r_parents_raw = r_graph.rx2("parents")(py_to_r_index(node))
                r_parents = (
                    {r_to_py_index(int(p)) for p in r_parents_raw}
                    if len(r_parents_raw) > 0
                    else set()
                )

                assert py_parents == r_parents, f"Parents of node {node} differ"

    @pytest.mark.parametrize("graph_id", get_test_graph_ids())
    def test_siblings(self, graph_id, r_semid, r_converter):
        """Test siblings() method matches R."""
        graph_data = TEST_GRAPHS[graph_id]
        L, O = graph_data["L"], graph_data["O"]
        n = L.shape[0]

        py_graph = MixedGraph(L, O)

        with localconverter(r_converter):
            r_graph = r_semid.MixedGraph(L, O)

            for node in range(n):
                py_siblings = set(py_graph.siblings([node]))
                r_siblings_raw = r_graph.rx2("siblings")(py_to_r_index(node))
                r_siblings = (
                    {r_to_py_index(int(s)) for s in r_siblings_raw}
                    if len(r_siblings_raw) > 0
                    else set()
                )

                assert py_siblings == r_siblings, f"Siblings of node {node} differ"

    @pytest.mark.parametrize("graph_id", get_test_graph_ids())
    def test_descendants(self, graph_id, r_semid, r_converter):
        """Test descendants() method matches R."""
        graph_data = TEST_GRAPHS[graph_id]
        L, O = graph_data["L"], graph_data["O"]
        n = L.shape[0]

        py_graph = MixedGraph(L, O)

        with localconverter(r_converter):
            r_graph = r_semid.MixedGraph(L, O)

            for node in range(n):
                py_desc = set(py_graph.descendants([node]))
                r_desc_raw = r_graph.rx2("descendants")(py_to_r_index(node))
                r_desc = (
                    {r_to_py_index(int(d)) for d in r_desc_raw}
                    if len(r_desc_raw) > 0
                    else set()
                )

                assert py_desc == r_desc, f"Descendants of node {node} differ"

    @pytest.mark.parametrize("graph_id", get_test_graph_ids())
    def test_ancestors(self, graph_id, r_semid, r_converter):
        """Test ancestors() method matches R."""
        graph_data = TEST_GRAPHS[graph_id]
        L, O = graph_data["L"], graph_data["O"]
        n = L.shape[0]

        py_graph = MixedGraph(L, O)

        with localconverter(r_converter):
            r_graph = r_semid.MixedGraph(L, O)

            for node in range(n):
                py_anc = set(py_graph.ancestors([node]))
                r_anc_raw = r_graph.rx2("ancestors")(py_to_r_index(node))
                r_anc = (
                    {r_to_py_index(int(a)) for a in r_anc_raw}
                    if len(r_anc_raw) > 0
                    else set()
                )

                assert py_anc == r_anc, f"Ancestors of node {node} differ"


# =============================================================================
# HTC Identification Tests
# =============================================================================


class TestHTCIdentification:
    """Test HTC identification matches R implementation."""

    @pytest.mark.parametrize("graph_id", get_test_graph_ids())
    def test_htc_solved_edges(self, graph_id, r_semid, r_converter):
        """Test that htcID identifies the same edges as R."""
        graph_data = TEST_GRAPHS[graph_id]
        L, O = graph_data["L"], graph_data["O"]
        n = L.shape[0]

        py_graph = MixedGraph(L, O)
        py_result = htc_id(py_graph)

        with localconverter(r_converter):
            r_graph = r_semid.MixedGraph(L, O)
            r_result = r_semid.htcID(r_graph)

        py_solved = extract_py_solved_edges(py_result)
        r_solved = extract_r_solved_edges(r_result, n)

        assert py_solved == r_solved, f"Solved edges differ for {graph_id}"

    @pytest.mark.parametrize("graph_id", get_test_graph_ids())
    def test_htc_unsolved_edges(self, graph_id, r_semid, r_converter):
        """Test that htcID has same unsolved edges as R."""
        graph_data = TEST_GRAPHS[graph_id]
        L, O = graph_data["L"], graph_data["O"]
        n = L.shape[0]

        py_graph = MixedGraph(L, O)
        py_result = htc_id(py_graph)

        with localconverter(r_converter):
            r_graph = r_semid.MixedGraph(L, O)
            r_result = r_semid.htcID(r_graph)

        py_unsolved = extract_py_unsolved_edges(py_result)
        r_unsolved = extract_r_unsolved_edges(r_result, n)

        assert py_unsolved == r_unsolved, f"Unsolved edges differ for {graph_id}"

    @pytest.mark.parametrize("graph_id", get_test_graph_ids())
    def test_htc_identifier_function(self, graph_id, r_semid, r_converter):
        """Test that htcID identifier function produces same results as R."""
        graph_data = TEST_GRAPHS[graph_id]
        L, O = graph_data["L"], graph_data["O"]
        n = L.shape[0]

        py_graph = MixedGraph(L, O)
        py_result = htc_id(py_graph)

        with localconverter(r_converter):
            r_graph = r_semid.MixedGraph(L, O)
            r_result = r_semid.htcID(r_graph)

        compare_identifier_outputs(
            py_result.identifier,
            get_r_list_element(r_result, "identifier"),
            n,
            r_converter,
            L=L,
            O=O,
        )


# =============================================================================
# Edgewise Identification Tests
# =============================================================================


class TestEdgewiseIdentification:
    """Test edgewise identification matches R implementation."""

    @pytest.mark.parametrize("graph_id", get_test_graph_ids())
    def test_edgewise_solved_edges(self, graph_id, r_semid, r_converter):
        """Test that edgewiseID identifies the same edges as R."""
        graph_data = TEST_GRAPHS[graph_id]
        L, O = graph_data["L"], graph_data["O"]
        n = L.shape[0]

        py_graph = MixedGraph(L, O)
        py_result = edgewise_id(py_graph)

        with localconverter(r_converter):
            r_graph = r_semid.MixedGraph(L, O)
            r_result = r_semid.edgewiseID(r_graph)

        py_solved = extract_py_solved_edges(py_result)
        r_solved = extract_r_solved_edges(r_result, n)

        assert py_solved == r_solved, f"Solved edges differ for {graph_id}"

    @pytest.mark.parametrize("graph_id", get_test_graph_ids())
    def test_edgewise_identifier_function(self, graph_id, r_semid, r_converter):
        """Test that edgewiseID identifier function produces same results as R."""
        graph_data = TEST_GRAPHS[graph_id]
        L, O = graph_data["L"], graph_data["O"]
        n = L.shape[0]

        py_graph = MixedGraph(L, O)
        py_result = edgewise_id(py_graph)

        with localconverter(r_converter):
            r_graph = r_semid.MixedGraph(L, O)
            r_result = r_semid.edgewiseID(r_graph)

        compare_identifier_outputs(
            py_result.identifier,
            get_r_list_element(r_result, "identifier"),
            n,
            r_converter,
            L=L,
            O=O,
        )


# =============================================================================
# Trek Separation Identification Tests
# =============================================================================


class TestTrekSeparationIdentification:
    """Test trek separation identification matches R implementation."""

    @pytest.fixture(scope="class")
    def r_trek_sep_id(self, r_semid):
        """Define R function for trek separation ID."""
        ro.r("""
            trekSeparationID <- function(mixedGraph) {
                return(generalGenericID(mixedGraph, list(trekSeparationIdentifyStep)))
            }
        """)
        return ro.globalenv["trekSeparationID"]

    @pytest.mark.parametrize("graph_id", get_test_graph_ids())
    def test_trek_sep_solved_edges(self, graph_id, r_semid, r_converter, r_trek_sep_id):
        """Test that trekSepID identifies the same edges as R."""
        graph_data = TEST_GRAPHS[graph_id]
        L, O = graph_data["L"], graph_data["O"]
        n = L.shape[0]

        py_graph = MixedGraph(L, O)
        py_result = trek_sep_id(py_graph)

        with localconverter(r_converter):
            r_graph = r_semid.MixedGraph(L, O)
            r_result = r_trek_sep_id(r_graph)

        py_solved = extract_py_solved_edges(py_result)
        r_solved = extract_r_solved_edges(r_result, n)

        assert py_solved == r_solved, f"Solved edges differ for {graph_id}"

    @pytest.mark.parametrize("graph_id", get_test_graph_ids())
    def test_trek_sep_identifier_function(
        self, graph_id, r_semid, r_converter, r_trek_sep_id
    ):
        """Test that trekSepID identifier function produces same results as R."""
        graph_data = TEST_GRAPHS[graph_id]
        L, O = graph_data["L"], graph_data["O"]
        n = L.shape[0]

        py_graph = MixedGraph(L, O)
        py_result = trek_sep_id(py_graph)

        with localconverter(r_converter):
            r_graph = r_semid.MixedGraph(L, O)
            r_result = r_trek_sep_id(r_graph)

        compare_identifier_outputs(
            py_result.identifier,
            get_r_list_element(r_result, "identifier"),
            n,
            r_converter,
            L=L,
            O=O,
        )


# =============================================================================
# Ancestral Identification Tests
# =============================================================================


class TestAncestralIdentification:
    """Test ancestral identification matches R implementation."""

    @pytest.mark.parametrize("graph_id", get_test_graph_ids())
    def test_ancestral_solved_edges(self, graph_id, r_semid, r_converter):
        """Test that ancestralID identifies the same edges as R."""
        graph_data = TEST_GRAPHS[graph_id]
        L, O = graph_data["L"], graph_data["O"]
        n = L.shape[0]

        py_graph = MixedGraph(L, O)
        py_result = ancestral_id(py_graph)

        with localconverter(r_converter):
            r_graph = r_semid.MixedGraph(L, O)
            r_result = r_semid.ancestralID(r_graph)

        py_solved = extract_py_solved_edges(py_result)
        r_solved = extract_r_solved_edges(r_result, n)

        assert py_solved == r_solved, f"Solved edges differ for {graph_id}"


# =============================================================================
# SEMID (Combined) Tests
# =============================================================================


class TestSEMID:
    """Test combined SEMID algorithm matches R implementation."""

    @pytest.mark.parametrize("graph_id", get_test_graph_ids())
    def test_semid_identification_status(self, graph_id, r_semid, r_converter):
        """Test that graphID returns same identification status as R."""
        graph_data = TEST_GRAPHS[graph_id]
        L, O = graph_data["L"], graph_data["O"]

        py_graph = MixedGraph(L, O)
        py_result = semid(py_graph)

        with localconverter(r_converter):
            r_graph = r_semid.MixedGraph(L, O)
            r_result = r_semid.semID(r_graph)

        # Compare global identifiability
        # R semID returns isGlobalID
        py_global = py_result.is_global_id
        r_global_raw = get_r_list_element(r_result, "isGlobalID")
        r_global = bool(r_global_raw[0]) if r_global_raw is not None else None

        # If R returns NA (which converts to something like NA_logical), it's tricky.
        # But for valid graphs it should be bool.
        # Python uses None if not tested, but default is tested.
        if py_global is not None:
            assert py_global == r_global, f"Global ID status differs for {graph_id}"

        # Compare generic non-identifiability
        # R semID returns isGenericNonID
        py_generic_non_id = py_result.is_generic_non_id
        r_generic_non_id_raw = get_r_list_element(r_result, "isGenericNonID")
        r_generic_non_id = (
            bool(r_generic_non_id_raw[0]) if r_generic_non_id_raw is not None else None
        )

        if py_generic_non_id is not None:
            assert py_generic_non_id == r_generic_non_id, (
                f"Generic Non-ID status differs for {graph_id}"
            )

    @pytest.mark.parametrize("graph_id", get_test_graph_ids())
    def test_semid_identifier_function(self, graph_id, r_semid, r_converter):
        """Test that graphID identifier function produces same results as R."""
        graph_data = TEST_GRAPHS[graph_id]
        L, O = graph_data["L"], graph_data["O"]
        n = L.shape[0]

        py_graph = MixedGraph(L, O)
        py_result = semid(py_graph)

        with localconverter(r_converter):
            r_graph = r_semid.MixedGraph(L, O)
            r_result = r_semid.semID(r_graph)

        # semID returns genericIDResult which contains the identifier
        r_generic_result = get_r_list_element(r_result, "genericIDResult")

        # Check if generic identification was run and we have a result
        if r_generic_result is None or len(r_generic_result) == 0:
            # If R didn't run generic ID, Python might have (defaults match?)
            # semid default is test_generic_non_id=True, id_step_functions=[htc]
            # R semID default is genericIdStepFunctions = list(htcIdentifyStep)
            pass
        else:
            r_identifier = get_r_list_element(r_generic_result, "identifier")
            if py_result.generic_id_result and py_result.generic_id_result.identifier:
                compare_identifier_outputs(
                    py_result.generic_id_result.identifier,
                    r_identifier,
                    n,
                    r_converter,
                    L=L,
                    O=O,
                )


# =============================================================================
# Random Graph Tests
# =============================================================================


class TestRandomGraphs:
    """Test with randomly generated graphs."""

    @staticmethod
    def random_dag(n: int, edge_prob: float, seed: int) -> NDArray:
        """Generate random DAG adjacency matrix."""
        np.random.seed(seed)
        L = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.random() < edge_prob:
                    L[i, j] = 1
        return L

    @staticmethod
    def random_bidirected(n: int, edge_prob: float, seed: int) -> NDArray:
        """Generate random symmetric bidirected adjacency matrix."""
        np.random.seed(seed)
        O = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.random() < edge_prob:
                    O[i, j] = O[j, i] = 1
        return O

    @pytest.mark.parametrize("seed", range(20))
    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_htc_random_graphs(self, n, seed, r_semid, r_converter):
        """Test HTC on random graphs."""
        L = self.random_dag(n, 0.4, seed)
        O = self.random_bidirected(n, 0.3, seed + 1000)

        py_graph = MixedGraph(L, O)
        py_result = htc_id(py_graph)

        with localconverter(r_converter):
            r_graph = r_semid.MixedGraph(L, O)
            r_result = r_semid.htcID(r_graph)

        py_solved = extract_py_solved_edges(py_result)
        r_solved = extract_r_solved_edges(r_result, n)

        assert py_solved == r_solved, f"HTC differs for n={n}, seed={seed}"

    @pytest.mark.parametrize("seed", range(20))
    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_edgewise_random_graphs(self, n, seed, r_semid, r_converter):
        """Test edgewise on random graphs."""
        L = self.random_dag(n, 0.4, seed)
        O = self.random_bidirected(n, 0.3, seed + 1000)

        py_graph = MixedGraph(L, O)
        py_result = edgewise_id(py_graph)

        with localconverter(r_converter):
            r_graph = r_semid.MixedGraph(L, O)
            r_result = r_semid.edgewiseID(r_graph)

        py_solved = extract_py_solved_edges(py_result)
        r_solved = extract_r_solved_edges(r_result, n)

        assert py_solved == r_solved, f"Edgewise differs for n={n}, seed={seed}"

    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_semid_random_graphs(self, n, seed, r_semid, r_converter):
        """Test SEMID on random graphs."""
        L = self.random_dag(n, 0.4, seed)
        O = self.random_bidirected(n, 0.3, seed + 1000)

        py_graph = MixedGraph(L, O)
        py_result = semid(py_graph)

        with localconverter(r_converter):
            r_graph = r_semid.MixedGraph(L, O)
            r_result = r_semid.semID(r_graph)

        # Compare identifiability status
        # semid returns isGenericNonID (TRUE if non-ID, FALSE if ID conclusive, NA if inconclusive)
        py_generic_non_id = py_result.is_generic_non_id

        r_generic_non_id_raw = get_r_list_element(r_result, "isGenericNonID")
        r_generic_non_id = (
            bool(r_generic_non_id_raw[0]) if r_generic_non_id_raw is not None else None
        )

        assert py_generic_non_id == r_generic_non_id, (
            f"SEMID generic non-ID differs for n={n}, seed={seed}"
        )


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_node_graph(self, r_semid, r_converter):
        """Test single node graph."""
        L = np.array([[0.0]])
        O = np.array([[0.0]])

        py_graph = MixedGraph(L, O)
        py_result = htc_id(py_graph)

        with localconverter(r_converter):
            r_graph = r_semid.MixedGraph(L, O)
            r_result = r_semid.htcID(r_graph)

        py_solved = extract_py_solved_edges(py_result)
        r_solved = extract_r_solved_edges(r_result, 1)

        assert py_solved == r_solved

    def test_fully_connected_dag(self, r_semid, r_converter):
        """Test fully connected DAG (upper triangular)."""
        n = 4
        L = np.triu(np.ones((n, n)), k=1)
        O = np.zeros((n, n))

        py_graph = MixedGraph(L, O)
        py_result = htc_id(py_graph)

        with localconverter(r_converter):
            r_graph = r_semid.MixedGraph(L, O)
            r_result = r_semid.htcID(r_graph)

        py_solved = extract_py_solved_edges(py_result)
        r_solved = extract_r_solved_edges(r_result, n)

        assert py_solved == r_solved

    def test_no_directed_edges(self, r_semid, r_converter):
        """Test graph with only bidirected edges."""
        n = 3
        L = np.zeros((n, n))
        O = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)

        py_graph = MixedGraph(L, O)
        py_result = htc_id(py_graph)

        with localconverter(r_converter):
            r_graph = r_semid.MixedGraph(L, O)
            r_result = r_semid.htcID(r_graph)

        py_solved = extract_py_solved_edges(py_result)
        r_solved = extract_r_solved_edges(r_result, n)

        assert py_solved == r_solved


# =============================================================================
# Parameter Variation Tests
# =============================================================================


class TestParameterVariations:
    """Test algorithms with different parameter settings."""

    @pytest.mark.parametrize("tian_decompose", [True, False])
    @pytest.mark.parametrize("graph_id", ["chain_3", "diamond_4_confounded"])
    def test_htc_tian_decompose(self, graph_id, tian_decompose, r_semid, r_converter):
        """Test HTC with different tian_decompose settings."""
        graph_data = TEST_GRAPHS[graph_id]
        L, O = graph_data["L"], graph_data["O"]
        n = L.shape[0]

        py_graph = MixedGraph(L, O)
        py_result = htc_id(py_graph, tian_decompose=tian_decompose)

        with localconverter(r_converter):
            r_graph = r_semid.MixedGraph(L, O)
            r_result = r_semid.htcID(r_graph, tianDecompose=tian_decompose)

        py_solved = extract_py_solved_edges(py_result)
        r_solved = extract_r_solved_edges(r_result, n)

        assert py_solved == r_solved

    @pytest.mark.parametrize("subset_size", [1, 2, 3])
    def test_edgewise_subset_size(self, subset_size, r_semid, r_converter):
        """Test edgewise with different subset size controls."""
        L = np.array(
            [[0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=float
        )
        O = np.array(
            [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=float
        )
        n = L.shape[0]

        py_graph = MixedGraph(L, O)
        py_result = edgewise_id(py_graph, subset_size_control=subset_size)

        with localconverter(r_converter):
            r_graph = r_semid.MixedGraph(L, O)
            r_result = r_semid.edgewiseID(r_graph, subsetSizeControl=subset_size)

        py_solved = extract_py_solved_edges(py_result)
        r_solved = extract_r_solved_edges(r_result, n)

        assert py_solved == r_solved


# =============================================================================
# Edgewise + Trek Separation Identification Tests
# =============================================================================


class TestEdgewiseTSIdentification:
    """Test edgewise + trek separation identification matches R implementation."""

    @pytest.mark.parametrize("graph_id", get_test_graph_ids())
    def test_edgewise_ts_solved_edges(self, graph_id, r_semid, r_converter):
        """Test that edgewiseTSID identifies the same edges as R."""
        graph_data = TEST_GRAPHS[graph_id]
        L, O = graph_data["L"], graph_data["O"]
        n = L.shape[0]

        py_graph = MixedGraph(L, O)
        py_result = edgewise_ts_id(py_graph)

        with localconverter(r_converter):
            r_graph = r_semid.MixedGraph(L, O)
            r_result = r_semid.edgewiseTSID(r_graph)

        py_solved = extract_py_solved_edges(py_result)
        r_solved = extract_r_solved_edges(r_result, n)

        assert py_solved == r_solved, f"Solved edges differ for {graph_id}"

    @pytest.mark.parametrize("graph_id", get_test_graph_ids())
    def test_edgewise_ts_identifier_function(self, graph_id, r_semid, r_converter):
        """Test that edgewiseTSID identifier function produces same results as R."""
        graph_data = TEST_GRAPHS[graph_id]
        L, O = graph_data["L"], graph_data["O"]
        n = L.shape[0]

        py_graph = MixedGraph(L, O)
        py_result = edgewise_ts_id(py_graph)

        with localconverter(r_converter):
            r_graph = r_semid.MixedGraph(L, O)
            r_result = r_semid.edgewiseTSID(r_graph)

        compare_identifier_outputs(
            py_result.identifier,
            get_r_list_element(r_result, "identifier"),
            n,
            r_converter,
            L=L,
            O=O,
        )


# =============================================================================
# Latent-factor half-trek criterion identification Tests
# =============================================================================

# Test graphs for LFHTC (LatentDigraph where latent nodes are sources)
# Note: Some edge cases removed due to R package bugs:
# - Graphs with single latent child per latent node (R combn error "n < m")
# - Graphs with only one sibling pair (R indicesOmega vector indexing error)
LFHTC_TEST_GRAPHS = {
    "shared_latent": {
        # 3 observed (0, 1, 2), 1 latent (3)
        # obs: 0 -> 1, latent -> all observed
        "L": np.array(
            [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 0]],
            dtype=float,
        ),
        "num_observed": 3,
    },
    "complex_latent": {
        # 3 observed (0, 1, 2), 2 latents (3, 4)
        # obs: 0 -> 1 -> 2
        # L1 -> 0, L1 -> 1
        # L2 -> 1, L2 -> 2
        "L": np.array(
            [
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 1, 1, 0, 0],
            ],
            dtype=float,
        ),
        "num_observed": 3,
    },
}


def get_lfhtc_test_graph_ids():
    """Get list of LFHTC test graph IDs for parametrization."""
    return list(LFHTC_TEST_GRAPHS.keys())


def generate_random_latent_digraph(
    num_observed: int,
    num_latents: int,
    edge_prob: float,
    seed: int,
) -> np.ndarray:
    """
    Generate a random LatentDigraph adjacency matrix valid for LFHTC.

    Requirements to avoid R bugs:
    - Latent nodes must be sources (no parents)
    - Each latent must have at least 2 children (avoids R combn "n < m" error)
    - Must have at least 2 sibling pairs total (avoids R indicesOmega vector error)
      This requires at least one latent with 3+ children, or multiple latents sharing children

    Args:
        num_observed: Number of observed nodes (0 to num_observed-1)
        num_latents: Number of latent nodes (num_observed to num_observed+num_latents-1)
        edge_prob: Probability of edge between observed nodes
        seed: Random seed

    Returns:
        Adjacency matrix L of shape (num_observed + num_latents, num_observed + num_latents)
    """
    rng = np.random.default_rng(seed)
    n = num_observed + num_latents

    L = np.zeros((n, n), dtype=float)

    # Add random edges between observed nodes (no self-loops)
    for i in range(num_observed):
        for j in range(num_observed):
            if i != j and rng.random() < edge_prob:
                L[i, j] = 1

    # First latent gets at least 3 children to ensure >= 2 sibling pairs
    first_latent = num_observed
    num_children_first = rng.integers(3, max(4, num_observed + 1))
    children_first = rng.choice(
        num_observed, size=min(num_children_first, num_observed), replace=False
    )
    for child in children_first:
        L[first_latent, child] = 1

    # Other latents get at least 2 children each
    for latent in range(num_observed + 1, n):
        num_children = rng.integers(2, max(3, num_observed + 1))
        children = rng.choice(
            num_observed, size=min(num_children, num_observed), replace=False
        )
        for child in children:
            L[latent, child] = 1

    return L


def get_random_lfhtc_seeds():
    """Get seeds for random LFHTC test generation."""
    return list(range(10))


class TestLFHTCIdentification:
    """Test latent-factor half-trek criterion identification matches R implementation."""

    @pytest.mark.parametrize("graph_id", get_lfhtc_test_graph_ids())
    def test_lf_htc_solved_edges(self, graph_id, r_semid, r_converter):
        """Test that lfhtcID identifies the same edges as R."""
        graph_data = LFHTC_TEST_GRAPHS[graph_id]
        L = graph_data["L"]
        num_observed = graph_data["num_observed"]

        py_graph = LatentDigraph(L, num_observed=num_observed)
        py_result = lf_htc_id(py_graph)

        with localconverter(r_converter):
            # R LatentDigraph uses 1-based node lists as IntVectors
            observed_nodes_r = ro.IntVector(range(1, num_observed + 1))
            latent_nodes_r = ro.IntVector(range(num_observed + 1, L.shape[0] + 1))
            r_graph = r_semid.LatentDigraph(L, observed_nodes_r, latent_nodes_r)
            r_result = r_semid.lfhtcID(r_graph)

        py_solved = extract_py_solved_edges(py_result)
        r_solved = extract_r_solved_edges(r_result, num_observed)

        assert py_solved == r_solved, f"Solved edges differ for {graph_id}"

    @pytest.mark.parametrize("graph_id", get_lfhtc_test_graph_ids())
    def test_lf_htc_unsolved_edges(self, graph_id, r_semid, r_converter):
        """Test that lfhtcID has same unsolved edges as R."""
        graph_data = LFHTC_TEST_GRAPHS[graph_id]
        L = graph_data["L"]
        num_observed = graph_data["num_observed"]

        py_graph = LatentDigraph(L, num_observed=num_observed)
        py_result = lf_htc_id(py_graph)

        with localconverter(r_converter):
            observed_nodes_r = ro.IntVector(range(1, num_observed + 1))
            latent_nodes_r = ro.IntVector(range(num_observed + 1, L.shape[0] + 1))
            r_graph = r_semid.LatentDigraph(L, observed_nodes_r, latent_nodes_r)
            r_result = r_semid.lfhtcID(r_graph)

        py_unsolved = extract_py_unsolved_edges(py_result)
        r_unsolved = extract_r_unsolved_edges(r_result, num_observed)

        assert py_unsolved == r_unsolved, f"Unsolved edges differ for {graph_id}"

    @pytest.mark.parametrize("graph_id", get_lfhtc_test_graph_ids())
    def test_lf_htc_identifier_function(self, graph_id, r_semid, r_converter):
        """Test that lfhtcID identifier function produces same results as R."""
        graph_data = LFHTC_TEST_GRAPHS[graph_id]
        L = graph_data["L"]
        num_observed = graph_data["num_observed"]
        n = L.shape[0]

        py_graph = LatentDigraph(L, num_observed=num_observed)
        py_result = lf_htc_id(py_graph)

        with localconverter(r_converter):
            observed_nodes_r = ro.IntVector(range(1, num_observed + 1))
            latent_nodes_r = ro.IntVector(range(num_observed + 1, n + 1))
            r_graph = r_semid.LatentDigraph(L, observed_nodes_r, latent_nodes_r)
            r_result = r_semid.lfhtcID(r_graph)

        # Identifier works with covariance matrices of observed nodes only
        compare_identifier_outputs(
            py_result.identifier,
            get_r_list_element(r_result, "identifier"),
            num_observed,
            r_converter,
        )

    @pytest.mark.parametrize("seed", get_random_lfhtc_seeds())
    def test_lf_htc_random_solved_edges(self, seed, r_semid, r_converter):
        """Test lfhtcID on random graphs - solved edges match R."""
        # Generate random graph with varying sizes
        rng = np.random.default_rng(seed)
        num_observed = rng.integers(3, 6)
        num_latents = rng.integers(1, 3)
        edge_prob = rng.uniform(0.2, 0.5)

        L = generate_random_latent_digraph(num_observed, num_latents, edge_prob, seed)
        n = L.shape[0]

        py_graph = LatentDigraph(L, num_observed=num_observed)
        py_result = lf_htc_id(py_graph)

        with localconverter(r_converter):
            observed_nodes_r = ro.IntVector(range(1, num_observed + 1))
            latent_nodes_r = ro.IntVector(range(num_observed + 1, n + 1))
            r_graph = r_semid.LatentDigraph(L, observed_nodes_r, latent_nodes_r)
            r_result = r_semid.lfhtcID(r_graph)

        py_solved = extract_py_solved_edges(py_result)
        r_solved = extract_r_solved_edges(r_result, num_observed)

        assert py_solved == r_solved, f"Solved edges differ for random seed {seed}"

    @pytest.mark.parametrize("seed", get_random_lfhtc_seeds())
    def test_lf_htc_random_identifier(self, seed, r_semid, r_converter):
        """Test lfhtcID identifier on random graphs - outputs match R."""
        rng = np.random.default_rng(seed)
        num_observed = rng.integers(3, 6)
        num_latents = rng.integers(1, 3)
        edge_prob = rng.uniform(0.2, 0.5)

        L = generate_random_latent_digraph(num_observed, num_latents, edge_prob, seed)
        n = L.shape[0]

        py_graph = LatentDigraph(L, num_observed=num_observed)
        py_result = lf_htc_id(py_graph)

        with localconverter(r_converter):
            observed_nodes_r = ro.IntVector(range(1, num_observed + 1))
            latent_nodes_r = ro.IntVector(range(num_observed + 1, n + 1))
            r_graph = r_semid.LatentDigraph(L, observed_nodes_r, latent_nodes_r)
            r_result = r_semid.lfhtcID(r_graph)

        compare_identifier_outputs(
            py_result.identifier,
            get_r_list_element(r_result, "identifier"),
            num_observed,
            r_converter,
        )
