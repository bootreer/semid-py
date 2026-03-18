import marimo

__generated_with = "0.21.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import json

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import semid

    return json, mo, np, plt, semid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Factor Analysis: A Bug in the R Implementation

    Python `semid` implements a **more powerful** version of the `ext-M-identifiability`
    criterion than the R SEMID package. The difference traces to a single bug in R's
    `fullFactorCriterion`.

    ## The Bug

    Criterion (ii) of `fullFactorCriterion` checks that every child of a latent node
    outside the anchor set U can be "confirmed" by finding a shared latent ancestor
    in a restricted set L. Once a child v is confirmed, it should serve as an anchor
    for confirming later children; a kind of chaining effect.

    **R code (buggy) — `R/factor_analysis_localBBCriterion.R`:**
    ```r
    while(length(remainingVtoCheck) > 0){
        setUWithCheckedV <- U         # ← reset every iteration; prior confirmations lost
        for(v in remainingVtoCheck){
            for(u in setUWithCheckedV){
                if(all(jointParentsVandU %in% setOfL)){
                    foundU <- TRUE; break
                }
            }
            if(foundU){
                setUWithCheckedV <- union(setUWithCheckedV, v)  # ← discarded next pass
                break
            }
        }
    }
    ```

    **Python code (correct) — `src/semid/identification/factor_analysis.py`:**
    ```python
    checked = list(U)          # initialised once; grows monotonically
    while remaining_v:
        for v in list(remaining_v):
            if any(all(p in set_of_l for p in joint_parents([u, v]))
                   for u in checked):
                remaining_v.discard(v)
                checked.append(v)  # retained across while iterations
                break
        else:
            return False
    ```

    On graphs that **require chaining** (a v can only be confirmed using a previously
    confirmed v as anchor, not any u in U directly), R concludes "not identifiable"
    while Python correctly identifies the latent factors.

    ## Second Bug: Non-Termination of `extmID`

    R SEMID's `extmID` never returns when no progress is made — the outer while loop
    sets a flag but is missing a `return` (`R/SEMID/R/factor_analysis_extmID.R`):

    ```r
    if(!foundIdentifiableNode){
      result$identifiable <- FALSE
      result$tupleList <- tupleList
      # ← no return(result); loop restarts with identical state
    }
    ```

    This means **any** call to `extmID` that should return `FALSE` will loop forever,
    because every such case must have at least one iteration where no progress is made.
    The exhaustive enumeration of non-identifiable graphs (used to find the examples
    below) was therefore run using the original paper code
    (`checkExtendedMidentifiability`), which has this fixed:

    ```r
    if(!foundIdentifiableNode){
      return(list("identifiable" = FALSE, "tupleList" = list()))
    }
    ```

    The original paper code still has the chaining bug — it terminates but returns the
    wrong answer on graphs that require chaining. R results in this notebook were
    obtained with a 30-second timeout against R SEMID.
    """)
    return


@app.cell(hide_code=True)
def _(json, np):
    import pathlib
    _data = pathlib.Path(__file__).parent / "data" / "r_factor_results.json"
    with open(_data) as _f:
        R_FA = json.load(_f)

    # Lambda matrices — must match generate_r_results.R exactly.
    LAMBDAS = {
        "counterexample": np.array([
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 0, 0, 1],
        ]),
        "control": np.array([
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]),
        "chaining": np.array([
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]),
        "chaining_bug": np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 0, 1, 1],
        ]),
    }
    return LAMBDAS, R_FA


@app.cell(hide_code=True)
def _(mo):
    graph_select = mo.ui.dropdown(
        options=["counterexample", "control", "chaining", "chaining_bug"],
        value="counterexample",
        label="Select graph",
    )
    mo.vstack([
        mo.md("""
        | Graph | What it shows |
        |---|---|
        | `counterexample` | Python identifies; R times out (chaining bug + infinite loop) |
        | `control` | Both agree — identifiable in one localBB round |
        | `chaining` | Both agree — identifiable in two localBB rounds |
        | `chaining_bug` | Python identifies; R times out — from exhaustive enumeration, original paper code also returns False (chaining bug in isolation) |
        """),
        graph_select,
    ])
    return (graph_select,)


@app.cell(hide_code=True)
def _(LAMBDAS, R_FA, graph_select, mo, semid):
    _lam = LAMBDAS[graph_select.value]
    _py_result = semid.ext_m_id(_lam)
    _r_result = R_FA[graph_select.value]

    _rows = [
        {
            "Implementation": "Python `ext_m_id`",
            "Identifiable": str(_py_result.identifiable),
            "Steps found": len(_py_result.steps),
        },
        {
            "Implementation": "R `extmID` (30s timeout)",
            "Identifiable": str(_r_result["identifiable"]),
            "Steps found": _r_result["n_tuples"],
        },
    ]
    _r_matrix = (
        "lambda <- matrix(c(\n  "
        + ",\n  ".join(", ".join(str(x) for x in row) for row in _lam.tolist())
        + f"\n), nrow={_lam.shape[0]}, ncol={_lam.shape[1]}, byrow=TRUE)"
    )

    mo.vstack([
        mo.md(
            f"### Result for `{graph_select.value}`  \n"
            f"Lambda: {_lam.shape[0]} observed × {_lam.shape[1]} latent"
        ),
        mo.ui.table(_rows, selection=None),
        mo.md(f"**R matrix constructor:**\n```r\n{_r_matrix}\n```"),
    ])

    print(_py_result)
    return


@app.cell(hide_code=True)
def _(LAMBDAS, graph_select, mo, np, plt):
    from semid.latent_digraph import LatentDigraph
    import semid as _semid

    _lam = LAMBDAS[graph_select.value]
    _p, _k = _lam.shape
    # LatentDigraph: observed nodes occupy the first _p rows, latent the remaining _k rows.
    _adj = np.zeros((_p + _k, _p + _k), dtype=int)
    _adj[_p:, :_p] = _lam.T  # latent rows → observed cols
    _ldg = LatentDigraph(_adj, num_observed=_p)

    _fig, _ax = plt.subplots(figsize=(6, 3))
    _semid.plot_latent_digraph(_ldg, ax=_ax)
    _ax.set_title(f"Factor analysis graph: {graph_select.value}")
    plt.tight_layout()
    mo.as_html(_fig)
    return


if __name__ == "__main__":
    app.run()
