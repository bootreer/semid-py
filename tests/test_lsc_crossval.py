"""Cross-validate Python lsc_id against R LSC results.

Compares lsc_id outputs against pre-computed R results from
R/LSC/experiments/random-exps.R (100 graphs × 7 edge densities, 15 nodes:
10 observed + 5 latent).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from semid import LatentDigraph
from semid.identification.lsc import lsc_id

RESULTS_FILE = Path(__file__).resolve().parent / "data" / "lsc_O10L5_results.json"

N_NODES = 15
N_OBSERVED = 10
SAMPLE_SIZE = 200


def _load_sample() -> list[dict]:
    if not RESULTS_FILE.exists():
        pytest.skip("R results file not found")
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    all_graphs = [
        v
        for v in data.values()
        if isinstance(v.get("adjMatrix"), list) and len(v["adjMatrix"]) == N_NODES**2
    ]

    # Stratify on res3['id'] (strictest control) so both True/False are represented
    rng = np.random.default_rng(42)
    identified = [g for g in all_graphs if g["res3"]["id"]]
    not_identified = [g for g in all_graphs if not g["res3"]["id"]]
    half = SAMPLE_SIZE // 2
    sample = [
        identified[i]
        for i in rng.choice(len(identified), min(half, len(identified)), replace=False)
    ] + [
        not_identified[i]
        for i in rng.choice(len(not_identified), min(half, len(not_identified)), replace=False)
    ]
    rng.shuffle(sample)
    return list(sample)


@pytest.fixture(scope="module")
def lsc_graphs() -> list[dict]:
    return _load_sample()


def _make_graph(entry: dict) -> LatentDigraph:
    """Reconstruct LatentDigraph from R-exported row-major adjacency matrix."""
    adj = np.array(entry["adjMatrix"]).reshape(N_NODES, N_NODES)
    return LatentDigraph(adj, num_observed=N_OBSERVED)


@pytest.mark.parametrize("subset_size_control,r_key", [(1, "res1"), (2, "res2"), (3, "res3")])
def test_lsc_id_matches_r(
    lsc_graphs: list[dict], subset_size_control: int, r_key: str
) -> None:
    """Validate lsc_id against all R-computed results for given subset_size_control."""
    mismatches = []
    for i, entry in enumerate(lsc_graphs):
        g = _make_graph(entry)
        result = lsc_id(g, subset_size_control=subset_size_control)
        r_id = entry[r_key]["id"]
        if result.identified != r_id:
            mismatches.append(
                f"graph {i} (pErdos={entry['pErdos']}): "
                f"Python identified={result.identified}, R id={r_id}"
            )
    assert mismatches == [], (
        f"{len(mismatches)} / {len(lsc_graphs)} mismatches for "
        f"subset_size_control={subset_size_control}:\n" + "\n".join(mismatches[:20])
    )
