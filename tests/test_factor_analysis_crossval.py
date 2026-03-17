"""Cross-validate Python factor analysis against R id-factor-analysis results.

Compares m_id, ext_m_id, and zuta outputs against pre-computed R results
from the id-factor-analysis experiments.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from semid.identification.factor_analysis import ext_m_id, m_id, zuta

RESULTS_DIR = Path(__file__).resolve().parent / "data"

RESULTS_3x7 = RESULTS_DIR / "3latent_7observed_graphs_results.json"
RESULTS_4x9 = RESULTS_DIR / "4latent_9observed_graphs_results.json"


def _extract_lambda(graph: dict, n_latent: int, n_observed: int) -> np.ndarray:
    """Reconstruct the lambda matrix from the R adjacency matrix.

    R stores matrices in column-major order. The adjacency matrix has
    latent nodes 1..k and observed nodes (k+1)..(k+p) (1-indexed).
    adj[latent, observed] = t(lambda), so lambda = adj[latent, observed].T.
    """
    n = n_latent + n_observed
    adj_r = np.array(graph["adjMatrix"]).reshape(n, n, order="F")
    lam = adj_r[:n_latent, n_latent:].T
    return lam.astype(int)


# ---------------------------------------------------------------------------
# 3-latent, 7-observed (562 graphs — run all)
# ---------------------------------------------------------------------------


def _load_3x7() -> list[dict]:
    if not RESULTS_3x7.exists():
        pytest.skip("R results file not found")
    with open(RESULTS_3x7) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def graphs_3x7() -> list[dict]:
    return _load_3x7()


def test_3x7_m_id(graphs_3x7: list[dict]) -> None:
    """Validate m_id against all 562 R-computed 3-latent 7-observed graphs."""
    mismatches = []
    for i, g in enumerate(graphs_3x7):
        lam = _extract_lambda(g, 3, 7)
        result = m_id(lam)
        if result.identifiable != g["Mid"]:
            mismatches.append(
                f"graph {i}: Python m_id={result.identifiable}, R Mid={g['Mid']}, "
                f"edges={g['edges']}"
            )
    assert mismatches == [], (
        f"{len(mismatches)} / {len(graphs_3x7)} mismatches:\n"
        + "\n".join(mismatches[:20])
    )


def test_3x7_ext_m_id(graphs_3x7: list[dict]) -> None:
    """Validate ext_m_id against all 562 R-computed 3-latent 7-observed graphs."""
    mismatches = []
    for i, g in enumerate(graphs_3x7):
        lam = _extract_lambda(g, 3, 7)
        result = ext_m_id(lam)
        if result.identifiable != g["ExtMid"]:
            mismatches.append(
                f"graph {i}: Python ext_m_id={result.identifiable}, "
                f"R ExtMid={g['ExtMid']}, edges={g['edges']}"
            )
    assert mismatches == [], (
        f"{len(mismatches)} / {len(graphs_3x7)} mismatches:\n"
        + "\n".join(mismatches[:20])
    )


def test_3x7_zuta(graphs_3x7: list[dict]) -> None:
    """Validate zuta against all 562 R-computed 3-latent 7-observed graphs."""
    mismatches = []
    for i, g in enumerate(graphs_3x7):
        lam = _extract_lambda(g, 3, 7)
        result = zuta(lam)
        if result.zuta != g["ZUTA"]:
            mismatches.append(
                f"graph {i}: Python zuta={result.zuta}, R ZUTA={g['ZUTA']}, "
                f"edges={g['edges']}"
            )
    assert mismatches == [], (
        f"{len(mismatches)} / {len(graphs_3x7)} mismatches:\n"
        + "\n".join(mismatches[:20])
    )


# ---------------------------------------------------------------------------
# 4-latent, 9-observed (64112 graphs — sample to keep test fast)
# ---------------------------------------------------------------------------

SAMPLE_SIZE_4x9 = 1000
_CHUNKS_4x9 = 20  # chunks distributed across xdist workers


def _load_4x9_chunks() -> list[list[dict]] | None:
    """Load and chunk the 4x9 sample at collection time for parametrize.

    Returns None if the data file is not available.
    """
    if not RESULTS_4x9.exists():
        return None
    with open(RESULTS_4x9) as f:
        data = json.load(f)

    rng = np.random.default_rng(42)

    # Stratified sample: include all cases where ExtMid != Mid (617),
    # then fill the rest randomly from the remaining graphs.
    diff_cases = [g for g in data if g["ExtMid"] != g["Mid"]]
    same_cases = [g for g in data if g["ExtMid"] == g["Mid"]]

    n_remaining = max(0, SAMPLE_SIZE_4x9 - len(diff_cases))
    indices = rng.choice(
        len(same_cases), size=min(n_remaining, len(same_cases)), replace=False
    )
    sample = diff_cases + [same_cases[i] for i in indices]
    rng.shuffle(sample)

    size = max(1, (len(sample) + _CHUNKS_4x9 - 1) // _CHUNKS_4x9)
    return [sample[i : i + size] for i in range(0, len(sample), size)]


_4x9_chunks = _load_4x9_chunks()
_4x9_params: list = (
    _4x9_chunks
    if _4x9_chunks is not None
    else [pytest.param([], marks=pytest.mark.skip(reason="R results file not found"))]
)
_4x9_ids = [f"chunk{i}" for i in range(len(_4x9_params))]


@pytest.mark.parametrize("chunk", _4x9_params, ids=_4x9_ids)
def test_4x9_m_id(chunk: list[dict]) -> None:
    """Validate m_id against sampled 4-latent 9-observed graphs."""
    mismatches = []
    for i, g in enumerate(chunk):
        lam = _extract_lambda(g, 4, 9)
        result = m_id(lam)
        if result.identifiable != g["Mid"]:
            mismatches.append(
                f"graph {i}: Python m_id={result.identifiable}, R Mid={g['Mid']}, "
                f"edges={g['edges']}"
            )
    assert mismatches == [], (
        f"{len(mismatches)} / {len(chunk)} mismatches:\n"
        + "\n".join(mismatches[:20])
    )


@pytest.mark.parametrize("chunk", _4x9_params, ids=_4x9_ids)
def test_4x9_ext_m_id(chunk: list[dict]) -> None:
    """Validate ext_m_id against sampled 4-latent 9-observed graphs."""
    mismatches = []
    for i, g in enumerate(chunk):
        lam = _extract_lambda(g, 4, 9)
        result = ext_m_id(lam)
        if result.identifiable != g["ExtMid"]:
            mismatches.append(
                f"graph {i}: Python ext_m_id={result.identifiable}, "
                f"R ExtMid={g['ExtMid']}, edges={g['edges']}, "
                f"{lam}"
            )
    assert mismatches == [], (
        f"{len(mismatches)} / {len(chunk)} mismatches:\n"
        + "\n".join(mismatches[:20])
    )


@pytest.mark.parametrize("chunk", _4x9_params, ids=_4x9_ids)
def test_4x9_zuta(chunk: list[dict]) -> None:
    """Validate zuta against sampled 4-latent 9-observed graphs."""
    mismatches = []
    for i, g in enumerate(chunk):
        lam = _extract_lambda(g, 4, 9)
        result = zuta(lam)
        if result.zuta != g["ZUTA"]:
            mismatches.append(
                f"graph {i}: Python zuta={result.zuta}, R ZUTA={g['ZUTA']}, "
                f"edges={g['edges']}"
            )
    assert mismatches == [], (
        f"{len(mismatches)} / {len(chunk)} mismatches:\n"
        + "\n".join(mismatches[:20])
    )


# ---------------------------------------------------------------------------
# 10-latent, 25-observed (5000 graphs per density — sample to keep test fast)
# ---------------------------------------------------------------------------

SAMPLE_SIZE_10x25 = 200


@pytest.fixture(
    scope="module",
    params=["0.2", "0.25", "0.3", "0.35", "0.4", "0.45"],
)
def graphs_10x25(request: pytest.FixtureRequest) -> list[dict]:
    prob = request.param
    path = RESULTS_DIR / f"10latent_25observed_graphs_{prob}_results.json"
    if not path.exists():
        pytest.skip("R results file not found")
    with open(path) as f:
        data = json.load(f)

    # Stratify on ExtMid so both True/False are represented even at extreme densities
    rng = np.random.default_rng(42)
    true_cases = [g for g in data if g["ExtMid"]]
    false_cases = [g for g in data if not g["ExtMid"]]
    half = SAMPLE_SIZE_10x25 // 2
    sample = [
        true_cases[i]
        for i in rng.choice(len(true_cases), min(half, len(true_cases)), replace=False)
    ] + [
        false_cases[i]
        for i in rng.choice(
            len(false_cases), min(half, len(false_cases)), replace=False
        )
    ]
    rng.shuffle(sample)
    return list(sample)


def test_10x25_ext_m_id(graphs_10x25: list[dict]) -> None:
    """Validate ext_m_id against sampled 10-latent 25-observed graphs (per density)."""
    mismatches = []
    for i, g in enumerate(graphs_10x25):
        lam = _extract_lambda(g, 10, 25)
        result = ext_m_id(lam)
        if result.identifiable != g["ExtMid"]:
            mismatches.append(
                f"graph {i}: Python ext_m_id={result.identifiable}, "
                f"R ExtMid={g['ExtMid']}, edges={g['edges']}"
            )
    assert mismatches == [], (
        f"{len(mismatches)} / {len(graphs_10x25)} mismatches:\n"
        + "\n".join(mismatches[:20])
    )
