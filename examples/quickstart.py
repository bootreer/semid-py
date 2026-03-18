import marimo

__generated_with = "0.21.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    from semid import (
        LatentDigraph,
        MixedGraph,
        ancestral_id,
        edgewise_id,
        edgewise_ts_id,
        ext_m_id,
        general_generic_id,
        global_id,
        htc_id,
        lf_htc_id,
        m_id,
        zuta,
    )
    from semid.identification import (
        ancestral_identify_step,
        edgewise_identify_step,
        htc_identify_step,
        trek_separation_identify_step,
    )

    return (
        LatentDigraph,
        MixedGraph,
        ancestral_id,
        ancestral_identify_step,
        edgewise_id,
        edgewise_identify_step,
        edgewise_ts_id,
        ext_m_id,
        general_generic_id,
        global_id,
        htc_id,
        htc_identify_step,
        lf_htc_id,
        m_id,
        mo,
        np,
        trek_separation_identify_step,
        zuta,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SEMID Quick Tour

    Python port of the [R SEMID package](https://github.com/Lucaweihs/SEMID) for
    determining parameter identifiability in linear structural equation models (SEMs)
    with latent variables.

    This notebook follows the structure of the R package README. Node indices are
    0-based in Python (versus 1-based in R).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mixed Graphs
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Mixed graphs are specified by a directed adjacency matrix `L` and a symmetric
    bidirected adjacency matrix `O`.  `L[i, j] = 1` encodes a directed edge `i → j`;
    `O[i, j] = 1` encodes a bidirected edge `i ↔ j`.

    This graph has 5 nodes, 7 directed edges, and 3 bidirected edges.
    """)
    return


@app.cell
def _(MixedGraph, np):
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
    g = MixedGraph(L, O)
    g.plot()
    return (g,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Global Identifiability
    """)
    return


@app.cell
def _(g, global_id):
    global_id(g)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generic Identifiability

    No "if and only if" graphical condition for generic identifiability is known in
    general, but several **sufficient conditions** are implemented.
    """)
    return


@app.cell
def _(g, htc_id):
    # Half-trek criterion
    print(htc_id(g))
    return


@app.cell
def _(ancestral_id, g):
    # Ancestor decomposition
    print(ancestral_id(g))
    return


@app.cell
def _(edgewise_id, g):
    # Edgewise identification
    print(edgewise_id(g))
    return


@app.cell
def _(edgewise_ts_id, g):
    # Edgewise + trek-separation
    print(edgewise_ts_id(g))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Criteria can be combined and applied repeatedly via `general_generic_id`.
    Each step function is tried in order until one makes progress; the loop
    continues until no new edges are identified.
    """)
    return


@app.cell
def _(
    ancestral_identify_step,
    edgewise_identify_step,
    g,
    general_generic_id,
    htc_identify_step,
    trek_separation_identify_step,
):
    print(general_generic_id(
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
    ### Necessary Condition for Non-Identifiability

    When sufficient conditions leave edges unidentified, we can check a **necessary**
    condition for generic identifiability.  If `non_htc_id()` returns `True` the
    graph is provably generically non-identifiable.
    """)
    return


@app.cell
def _(g):
    g.non_htc_id()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Latent-Factor Graphs
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The latent-factor half-trek criterion (LF-HTC) checks generic identifiability
    in models with explicitly represented latent variables.  The adjacency matrix
    lists observed nodes first (rows/columns `0 .. num_observed-1`) followed by
    latent nodes.
    """)
    return


@app.cell
def _(LatentDigraph, np):
    # 5 observed nodes (0-4), 1 global latent (5) connected to all observed.
    # Directed edges among observed: 0->1, 1->2, 3->4.
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
    g_lat = LatentDigraph(L_lat, num_observed=5)
    g_lat.plot()
    return (g_lat,)


@app.cell
def _(g_lat, lf_htc_id):
    print(lf_htc_id(g_lat))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sparse Factor Analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The **matching criterion** checks M-identifiability of the factor loading matrix
    (up to column sign).  The input `lambda` is a binary matrix of shape
    `(n_observed, n_latent)` where entry `[i, j] = 1` means observed variable `i`
    loads on latent factor `j`.
    """)
    return


@app.cell
def _(m_id, np):
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
    print(m_id(lam))
    return (lam,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    M-identifiability applies only when the **Zero Upper Triangular Assumption
    (ZUTA)** holds.  `zuta()` verifies this.
    """)
    return


@app.cell
def _(lam, zuta):
    print(zuta(lam))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `ext_m_id` implements an extended, more powerful sufficient condition that
    combines the matching criterion with a local building-block rule.
    """)
    return


@app.cell
def _(ext_m_id, np):
    lam_ext = np.array(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
        ],
        dtype=np.int32,
    )
    print(ext_m_id(lam_ext))
    return


if __name__ == "__main__":
    app.run()
