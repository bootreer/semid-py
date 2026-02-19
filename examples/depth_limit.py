import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Demo: Testing HTC identifiability with depth limited treks
    """)
    return


@app.cell(hide_code=True)
def _():
    import semid
    import numpy as np
    import igraph as ig
    import marimo as mo

    # generating random MixedGraphs
    def generate_random_mixed(n: int, p: float, q: float) -> semid.MixedGraph:
        assert 0 < p < 1 and 0 < q < 1
        d_adj = np.zeros((n, n), dtype=int)
        b_adj = np.zeros((n, n), dtype=int)
        tree: ig.Graph = ig.Graph.Tree_Game(n, directed=False)
        for edge in tree.get_edgelist():
            i, j = edge
            b_adj[i, j] = b_adj[j, i] = 1
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.rand() < p:
                    b_adj[i, j] = b_adj[j, i] = 1
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.rand() < q:
                    d_adj[i, j] = 1
        return semid.MixedGraph(d_adj, b_adj)

    return generate_random_mixed, mo, semid


@app.cell
def _():
    # running this takes around 10-20 minutes
    graph_sizes = [8, 12, 25, 50, 75, 100]
    ps = [0.1, 0.2, 0.3]
    qs = [0.2, 0.3, 0.4, 0.5, 0.6]
    depths = [1, 2, 3, 5, 10, None]  # None = unlimited
    return depths, graph_sizes, ps, qs


@app.cell(hide_code=True)
def _(depths, generate_random_mixed, graph_sizes, mo, ps, qs, semid):
    import itertools
    import time
    import pandas as pd

    N = 10  # graphs per combo

    def count_identified(result):
        dir_id = sum(len(p) for p in result.solved_parents)
        bi_id = sum(len(s) for s in result.solved_siblings) // 2
        return dir_id + bi_id

    combos = list(itertools.product(graph_sizes, ps, qs))
    records = []

    with mo.status.progress_bar(
        total=len(combos) * N, title="Running experiments"
    ) as bar:
        for size, p, q in combos:
            times = {d: [] for d in depths}
            id_counts = {d: 0 for d in depths}

            for _ in range(N):
                mg = generate_random_mixed(size, p, q)

                t0 = time.perf_counter()
                unlimited_result = semid.htc_id(mg, max_hops=None)
                unlimited_time = time.perf_counter() - t0
                unlimited_count = count_identified(unlimited_result)

                times[None].append(unlimited_time)
                id_counts[None] += unlimited_count

                for d in [d for d in depths if d is not None]:
                    t0 = time.perf_counter()
                    result = semid.htc_id(mg, max_hops=d)
                    times[d].append(time.perf_counter() - t0)
                    id_counts[d] += count_identified(result)

                bar.update()

            unlimited_total = id_counts[None]
            for d in depths:
                records.append(
                    {
                        "size": size,
                        "p": p,
                        "q": q,
                        "depth": "∞" if d is None else d,
                        "id_rate_vs_unlimited": id_counts[d] / unlimited_total if unlimited_total else float("nan"),
                        "mean_time_ms": sum(times[d]) / len(times[d]) * 1000,
                    }
                )

    df = pd.DataFrame(records)
    mo.ui.table(df)
    return (df,)


@app.cell
def _(df, mo):
    agg = (
        df.groupby(["size", "depth"])[["id_rate_vs_unlimited", "mean_time_ms"]]
        .mean()
        .round(3)
        .reset_index()
    )
    mo.ui.table(agg)
    return (agg,)


@app.cell
def _(agg, mo):
    import matplotlib.pyplot as plt

    # Map depth labels to numeric x positions for plotting
    depth_order = [d for d in sorted(
        agg["depth"].unique(),
        key=lambda d: float("inf") if d == "∞" else int(d)
    )]
    x_pos = {d: i for i, d in enumerate(depth_order)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("HTC identification vs trek depth limit")

    for s, group in agg.groupby("size"):
        group = group.set_index("depth").reindex(depth_order).reset_index()
        xs = [x_pos[d] for d in group["depth"]]
        ax1.plot(xs, group["id_rate_vs_unlimited"], marker="o", label=f"n={s}")
        ax2.plot(xs, group["mean_time_ms"], marker="o", label=f"n={s}")

    for ax in (ax1, ax2):
        ax.set_xticks(list(x_pos.values()))
        ax.set_xticklabels(depth_order)
        ax.set_xlabel("Max hops")
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax1.set_ylabel("Identified edges vs unlimited")
    ax1.set_ylim(0, 1.05)
    ax2.set_ylabel("Mean time (ms)")

    plt.tight_layout()
    mo.as_html(fig)
    return


if __name__ == "__main__":
    app.run()
