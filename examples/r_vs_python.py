import marimo

__generated_with = "0.21.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import semid

    return mo, np, semid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # R SEMID vs Python `semid`: Syntax and API

    Python `semid` is a faithful port of the [R SEMID package](https://github.com/Lucaweihs/SEMID).
    This notebook highlights the key syntax and API differences between the two.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Graph Construction

    **R** uses column-major `matrix()` with `t()` transpose and **1-based** node indices.
    **Python** uses row-major `np.array()` with **0-based** indices.

    ````r
    # R
    L = t(matrix(
      c(0, 1, 0, 0, 0,
        0, 0, 0, 1, 1,
        0, 0, 0, 1, 0,
        0, 1, 0, 0, 1,
        0, 0, 0, 1, 0), 5, 5))

    O = t(matrix(
      c(0, 0, 0, 0, 0,
        0, 0, 1, 0, 1,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0), 5, 5)); O = O + t(O)

    g = MixedGraph(L, O)
    g$plot()
    ````
    """)
    return


@app.cell
def _(np, semid):
    # Python
    L = np.array(
        [
            [0, 1, 0, 0, 0],  # 0 -> 1
            [0, 0, 0, 1, 1],  # 1 -> 3, 1 -> 4
            [0, 0, 0, 1, 0],  # 2 -> 3
            [0, 1, 0, 0, 1],  # 3 -> 1, 3 -> 4
            [0, 0, 0, 1, 0],  # 4 -> 3
        ],
        dtype=np.int32,
    )
    O = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1],  # 1 <-> 2, 1 <-> 4
            [0, 1, 0, 1, 0],  # 2 <-> 1, 2 <-> 3
            [0, 0, 1, 0, 0],  # 3 <-> 2
            [0, 1, 0, 0, 0],  # 4 <-> 1
        ],
        dtype=np.int32,
    )
    g = semid.MixedGraph(L, O)
    _ = g.plot()
    return (g,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Identification Functions

    Function names use `camelCase` in R and `snake_case` in Python.
    R functions are called as standalone functions; Python functions are imported from `semid`.

    | R | Python |
    |---|--------|
    | `globalID(g)` | `global_id(g)` |
    | `htcID(g)` | `htc_id(g)` |
    | `ancestralID(g)` | `ancestral_id(g)` |
    | `edgewiseID(g)` | `edgewise_id(g)` |
    | `edgewiseTSID(g)` | `edgewise_ts_id(g)` |
    | `generalGenericID(g, ...)` | `general_generic_id(g, ...)` |
    | `lfhtcID(g)` | `lf_htc_id(g)` |
    | `mID(lambda)` | `m_id(lam)` |
    | `extmID(lambda)` | `ext_m_id(lam)` |
    | `ZUTA(lambda)` | `zuta(lam)` |

    ````r
    # R
    htcID(g)
    ````
    """)
    return


@app.cell
def _(g, semid):
    # Python
    print(semid.htc_id(g))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step Functions and `generalGenericID`

    In R, step functions are bare names passed in a list.
    In Python they are imported from `semid.identification`.

    ````r
    # R
    generalGenericID(
      mixedGraph = g,
      idStepFunctions = list(htcIdentifyStep,
                             ancestralIdentifyStep,
                             edgewiseIdentifyStep,
                             trekSeparationIdentifyStep),
      tianDecompose = TRUE
    )
    ````
    """)
    return


@app.cell
def _(g, semid):
    # Python
    from semid.identification import (
        ancestral_identify_step,
        edgewise_identify_step,
        htc_identify_step,
        trek_separation_identify_step,
    )

    print(semid.general_generic_id(
        g,
        id_step_functions=[
            htc_identify_step,
            ancestral_identify_step,
            edgewise_identify_step,
            trek_separation_identify_step,
        ],
        decompose=True,
    ))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Non-HTC Identifiability Check

    In R, this is a standalone function taking raw matrices.
    In Python it is a method on `MixedGraph`.

    ````r
    # R
    graphID.nonHtcID(g$L(), g$O())
    ````
    """)
    return


@app.cell
def _(g):
    # Python
    g.non_htc_id()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Latent-Factor Graphs

    R's `LatentDigraph` takes explicit `observedNodes` and `latentNodes` vectors (1-based).
    Python takes `num_observed` — observed nodes are always the first rows/columns.

    ````r
    # R
    L = matrix(c(0, 1, 0, 0, 0, 0,
                 0, 0, 1, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 0, 0,
                 1, 1, 1, 1, 1, 0), 6, 6, byrow=TRUE)
    observedNodes = seq(1, 5)
    latentNodes = c(6)
    g = LatentDigraph(L, observedNodes, latentNodes)
    lfhtcID(g)
    ````
    """)
    return


@app.cell
def _(np, semid):
    # Python — observed nodes first (0-4), latent last (5)
    L_lat = np.array(
        [
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0],  # latent -> all observed
        ],
        dtype=np.int32,
    )
    g_lat = semid.LatentDigraph(L_lat, num_observed=5)
    print(semid.lf_htc_id(g_lat))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Factor Analysis

    Lambda is specified `byrow` in R and as a standard numpy array in Python.
    Node indices in R output are 1-based; Python output is 0-based.

    Python also accepts a `LatentDigraph` directly — R has no equivalent.

    ````r
    # R — lambda matrix only
    lambda = matrix(c(1, 0, 0,
                      1, 1, 0,
                      0, 1, 1,
                      1, 0, 1,
                      0, 1, 0,
                      0, 0, 1), 6, 3, byrow=TRUE)
    mID(lambda)
    ZUTA(lambda)
    ````
    """)
    return


@app.cell
def _(np, semid):
    # Python — lambda matrix
    lam = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.int32,
    )
    print(semid.m_id(lam))
    print()
    print(semid.zuta(lam))
    return (lam,)


@app.cell
def _(lam, np, semid):
    # Python only — LatentDigraph input (R has no equivalent)
    # Factor analysis functions only use latent->observed edges;
    # any observed->observed edges in the graph are ignored.
    p, k = lam.shape
    adj = np.zeros((p + k, p + k), dtype=np.int32)
    adj[p:, :p] = lam.T  # latent -> observed
    g_fa = semid.LatentDigraph(adj, num_observed=p)
    print(semid.m_id(g_fa))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python-only: `max_hops`

    Python adds a `max_hops` parameter to `htc_id` that limits the depth of half-trek
    path search. R always runs with unlimited depth. See `max_hops_benchmark.py` for a
    detailed comparison of depth vs. identification rate.

    ````python
    # Python only
    htc_id(g, max_hops=3)   # faster; may miss some identifiable edges
    htc_id(g, max_hops=None)  # equivalent to R's htcId(g)
    ````
    """)
    return


if __name__ == "__main__":
    app.run()
