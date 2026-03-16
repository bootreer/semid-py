"""Tests for factor analysis sign-identifiability algorithms."""

import numpy as np

from semid.identification.factor_analysis import (
    ExtMIDResult,
    LocalBBResult,
    MatchingResult,
    MIDResult,
    ZUTAResult,
    _flow_graph_matrix,
    _transform_lambda,
    check_local_bb_criterion,
    check_matching_criterion,
    ext_m_id,
    m_id,
    zuta,
)


class TestTransformLambda:
    def test_basic(self):
        lam = np.array([[1, 0], [1, 1], [0, 1]])  # 3 observed, 2 latent
        adj, latent, observed = _transform_lambda(lam)

        assert latent == [0, 1]
        assert observed == [2, 3, 4]
        assert adj.shape == (5, 5)
        assert adj[0, 2] == 1
        assert adj[0, 3] == 1
        assert adj[0, 4] == 0
        assert adj[1, 2] == 0
        assert adj[1, 3] == 1
        assert adj[1, 4] == 1
        assert adj[:2, :2].sum() == 0
        assert adj[2:, :].sum() == 0

    def test_single_latent(self):
        lam = np.array([[1], [1], [0]])
        adj, latent, observed = _transform_lambda(lam)
        assert latent == [0]
        assert observed == [1, 2, 3]
        assert adj[0, 1] == 1
        assert adj[0, 2] == 1
        assert adj[0, 3] == 0


class TestZUTA:
    def test_identity_matrix(self):
        lam = np.eye(3, dtype=int)
        result = zuta(lam)
        assert isinstance(result, ZUTAResult)
        assert result.zuta is True

    def test_upper_triangular(self):
        lam = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
        result = zuta(lam)
        assert result.zuta is True

    def test_not_zuta(self):
        lam = np.array([[1, 1], [1, 1]])
        result = zuta(lam)
        assert result.zuta is False

    def test_single_row_two_latents(self):
        # 1 observed loading on 2 latents -> not ZUTA (sum > 1 in single col)
        lam = np.array([[1, 1]])
        result = zuta(lam)
        assert result.zuta is False

    def test_single_column(self):
        lam = np.array([[1], [1], [1]])
        result = zuta(lam)
        assert result.zuta is True

    def test_zero_row_and_column(self):
        lam = np.array([[1, 0], [0, 0], [0, 1]])
        result = zuta(lam)
        assert result.zuta is True

    def test_single_observed_single_latent(self):
        lam = np.array([[1]])
        result = zuta(lam)
        assert result.zuta is True


class TestMatchingCriterion:
    def test_found_with_overlap(self):
        # 2 latent, 5 observed with overlapping children
        # Latent 0 -> obs 2,3,4; Latent 1 -> obs 4,5,6
        lam = np.array([[1, 0], [1, 0], [1, 1], [0, 1], [0, 1]])
        adj, latent, observed = _transform_lambda(lam)
        flow_adj = _flow_graph_matrix(adj, latent, observed)

        result = check_matching_criterion(flow_adj, adj, 0, latent, observed)
        assert isinstance(result, MatchingResult)
        assert result.found is True
        assert result.h == 0
        assert result.v is not None
        assert result.W is not None
        assert result.U is not None

    def test_not_found_too_small(self):
        # 2 latent, 3 observed upper triangular - too small for matching
        lam = np.array([[1, 0], [1, 1], [0, 1]])
        adj, latent, observed = _transform_lambda(lam)
        flow_adj = _flow_graph_matrix(adj, latent, observed)
        result = check_matching_criterion(flow_adj, adj, 0, latent, observed)
        assert result.found is False

    def test_not_found_single_child(self):
        lam = np.array([[1], [0]])
        adj, latent, observed = _transform_lambda(lam)
        flow_adj = _flow_graph_matrix(adj, latent, observed)
        result = check_matching_criterion(flow_adj, adj, 0, latent, observed)
        assert result.found is False


class TestLocalBBCriterion:
    def test_not_found_simple(self):
        lam = np.array([[1, 0], [0, 1]])
        adj, latent, observed = _transform_lambda(lam)
        result = check_local_bb_criterion(adj, latent, observed)
        assert isinstance(result, LocalBBResult)
        assert result.found is False

    def test_not_found_small_graph(self):
        # Need > 2 children for local BB to apply
        lam = np.array([[1, 0], [1, 1], [0, 1]])
        adj, latent, observed = _transform_lambda(lam)
        result = check_local_bb_criterion(adj, latent, observed)
        assert result.found is False


class TestMID:
    def test_identifiable_overlapping(self):
        # 2 latent, 5 observed with overlapping children -> M-identifiable
        lam = np.array([[1, 0], [1, 0], [1, 1], [0, 1], [0, 1]])
        result = m_id(lam)
        assert isinstance(result, MIDResult)
        assert result.identifiable is True
        assert len(result.steps) == 2

    def test_not_identifiable_small_upper_tri(self):
        # 3x3 upper triangular: too dense for matching criterion
        lam = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
        result = m_id(lam)
        assert result.identifiable is False

    def test_not_identifiable_2x3(self):
        # 2 latent, 3 observed upper triangular - not M-identifiable
        lam = np.array([[1, 0], [1, 1], [0, 1]])
        result = m_id(lam)
        assert result.identifiable is False

    def test_not_identifiable_full(self):
        lam = np.array([[1, 1], [1, 1]])
        result = m_id(lam)
        assert result.identifiable is False

    def test_childless_latent_trivially_identified(self):
        lam = np.array([[1, 0], [0, 0]])
        result = m_id(lam)
        assert isinstance(result, MIDResult)

    def test_identity_not_identifiable(self):
        lam = np.eye(2, dtype=int)
        result = m_id(lam)
        assert result.identifiable is False

    def test_max_card(self):
        lam = np.array([[1, 0], [1, 0], [1, 1], [0, 1], [0, 1]])
        result = m_id(lam, max_card=1)
        assert isinstance(result, MIDResult)

    def test_steps_structure(self):
        from semid.identification.factor_analysis import IdentificationStep

        lam = np.array([[1, 0], [1, 0], [1, 1], [0, 1], [0, 1]])
        result = m_id(lam)
        assert result.identifiable is True
        for step in result.steps:
            assert isinstance(step, IdentificationStep)
            assert step.criterion == "matching"
            assert step.h is not None
            assert step.v is not None
            assert step.W is not None
            assert step.U is not None

    def test_node_info(self):
        lam = np.array([[1, 0], [1, 0], [1, 1], [0, 1], [0, 1]])
        result = m_id(lam)
        assert result.latent_nodes == [0, 1]
        assert result.observed_nodes == [2, 3, 4, 5, 6]


class TestExtMID:
    def test_identifiable_overlapping(self):
        lam = np.array([[1, 0], [1, 0], [1, 1], [0, 1], [0, 1]])
        result = ext_m_id(lam)
        assert isinstance(result, ExtMIDResult)
        assert result.identifiable is True

    def test_not_identifiable_full(self):
        lam = np.array([[1, 1], [1, 1]])
        result = ext_m_id(lam)
        assert result.identifiable is False

    def test_result_has_node_info(self):
        lam = np.array([[1, 0], [1, 1], [0, 1]])
        result = ext_m_id(lam)
        assert result.latent_nodes == [0, 1]
        assert result.observed_nodes == [2, 3, 4]

    def test_steps_have_criterion(self):
        from semid.identification.factor_analysis import IdentificationStep

        lam = np.array([[1, 0], [1, 0], [1, 1], [0, 1], [0, 1]])
        result = ext_m_id(lam)
        assert result.identifiable is True
        for step in result.steps:
            assert isinstance(step, IdentificationStep)
            assert step.criterion in ("matching", "localBB")


def test_mid_result_steps_are_typed():
    """MIDResult.steps contains IdentificationStep objects, not raw dicts."""
    import numpy as np
    from semid.identification.factor_analysis import m_id, IdentificationStep

    # 2 latent, 5 observed with overlapping children -> M-identifiable, produces steps
    lam = np.array([[1, 0], [1, 0], [1, 1], [0, 1], [0, 1]])
    result = m_id(lam)

    assert isinstance(result.steps, list)
    assert len(result.steps) > 0
    step = result.steps[0]
    assert isinstance(step, IdentificationStep)
    assert step.criterion in ("matching", "localBB")
    _ = step.h
    _ = step.new_nodes_in_S


class TestImports:
    def test_top_level_imports(self):
        from semid import (
            ExtMIDResult,
            LocalBBResult,
            MatchingResult,
            MIDResult,
            ZUTAResult,
            check_local_bb_criterion,
            check_matching_criterion,
            ext_m_id,
            m_id,
            zuta,
        )
