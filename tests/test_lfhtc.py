"""Tests for latent-factor HTC identification."""

import pytest

from semid import lf_htc_id

from tests.conftest import LFHTC_EXAMPLES


@pytest.mark.parametrize(
    "example",
    LFHTC_EXAMPLES,
    ids=[ex.name for ex in LFHTC_EXAMPLES],
)
def test_lf_htc_id_r_examples(example):
    """Test lfhtcID on all examples from R package test suite (digraphExamples)."""
    graph = example.to_latent_digraph()
    result = lf_htc_id(graph)

    is_identifiable = sum(len(p) for p in result.unsolved_parents) == 0
    assert is_identifiable == example.lfhtc_id, (
        f"Example '{example.name}': expected lfhtc_id={example.lfhtc_id}, "
        f"got {is_identifiable} (unsolved={result.unsolved_parents})"
    )
