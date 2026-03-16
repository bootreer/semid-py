import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import semid

    L = [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    O = [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    graph = semid.MixedGraph(L, O, n=5)
    graph.get_half_trek_system([1, 2], [3, 4])
    return graph, semid


@app.cell
def _(graph, semid):
    htc_res = semid.htc_id(graph)
    print(htc_res)
    return


@app.cell
def _(graph, semid):
    ancestral_res = semid.ancestral_id(graph)
    print(ancestral_res)
    return


@app.cell
def _(graph, semid):
    semid_res = semid.semid(graph, tian_decompose=True)
    print(semid_res)
    return


if __name__ == "__main__":
    app.run()
