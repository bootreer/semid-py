"""Tests for graph visualization."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from semid import LatentDigraph, MixedGraph, plot_latent_digraph, plot_mixed_graph


class TestPlotMixedGraph:
    """Test plotting of mixed graphs."""

    def test_basic_plot(self):
        """Test basic mixed graph plotting."""
        L = np.array([[0, 1], [0, 0]], dtype=np.int32)
        O = np.array([[0, 1], [1, 0]], dtype=np.int32)
        graph = MixedGraph(L, O)

        fig, ax = plot_mixed_graph(graph, show=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_method(self):
        """Test plot() method on MixedGraph."""
        L = np.array([[0, 1], [0, 0]], dtype=np.int32)
        O = np.array([[0, 1], [1, 0]], dtype=np.int32)
        graph = MixedGraph(L, O)

        fig, ax = graph.plot(show=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_custom_labels(self):
        """Test custom node labels."""
        L = np.array([[0, 1], [0, 0]], dtype=np.int32)
        O = np.array([[0, 1], [1, 0]], dtype=np.int32)
        graph = MixedGraph(L, O, vertex_nums=[10, 20])

        fig, ax = plot_mixed_graph(graph, node_labels=["A", "B"], show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_layout_options(self):
        """Test different layout algorithms."""
        L = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int32)
        O = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int32)
        graph = MixedGraph(L, O)

        layouts = ["circle", "fr", "kk", "grid", "auto"]
        for layout in layouts:
            fig, ax = plot_mixed_graph(graph, layout=layout, show=False)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_custom_colors(self):
        """Test custom color options."""
        L = np.array([[0, 1], [0, 0]], dtype=np.int32)
        O = np.array([[0, 1], [1, 0]], dtype=np.int32)
        graph = MixedGraph(L, O)

        fig, ax = plot_mixed_graph(
            graph,
            node_color="lightblue",
            directed_edge_color="black",
            bidirected_edge_color="purple",
            show=False,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_node(self):
        """Test plotting a graph with a single node."""
        L = np.array([[0]], dtype=np.int32)
        O = np.array([[0]], dtype=np.int32)
        graph = MixedGraph(L, O)

        fig, ax = plot_mixed_graph(graph, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_larger_graph(self):
        """Test plotting a larger graph."""
        n = 5
        L = np.array(
            [
                [0, 1, 1, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        O = np.array(
            [
                [0, 0, 0, 1, 0],
                [0, 0, 1, 0, 1],
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        graph = MixedGraph(L, O)

        fig, ax = plot_mixed_graph(graph, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotLatentDigraph:
    """Test plotting of latent digraphs."""

    def test_basic_plot(self):
        """Test basic latent digraph plotting."""
        adj = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=np.int32)
        graph = LatentDigraph(adj, num_observed=2)

        fig, ax = plot_latent_digraph(graph, show=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_method(self):
        """Test plot() method on LatentDigraph."""
        adj = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=np.int32)
        graph = LatentDigraph(adj, num_observed=2)

        fig, ax = graph.plot(show=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_hide_latent(self):
        """Test hiding latent nodes."""
        adj = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=np.int32)
        graph = LatentDigraph(adj, num_observed=2)

        fig, ax = plot_latent_digraph(graph, show_latent=False, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_node_colors(self):
        """Test custom node colors for observed and latent nodes."""
        adj = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=np.int32)
        graph = LatentDigraph(adj, num_observed=2)

        fig, ax = plot_latent_digraph(
            graph,
            observed_node_color="lightblue",
            latent_node_color="pink",
            show=False,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_shapes(self):
        """Test custom node shapes."""
        adj = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=np.int32)
        graph = LatentDigraph(adj, num_observed=2)

        fig, ax = plot_latent_digraph(
            graph,
            observed_node_shape="o",
            latent_node_shape="^",
            show=False,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_observed(self):
        """Test graph with all observed nodes."""
        adj = np.array([[0, 1], [0, 0]], dtype=np.int32)
        graph = LatentDigraph(adj, num_observed=2)

        fig, ax = plot_latent_digraph(graph, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_latent(self):
        """Test graph with all latent nodes."""
        adj = np.array([[0, 1], [0, 0]], dtype=np.int32)
        graph = LatentDigraph(adj, num_observed=0)

        fig, ax = plot_latent_digraph(graph, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_labels(self):
        """Test custom node labels."""
        adj = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=np.int32)
        graph = LatentDigraph(adj, num_observed=2)

        fig, ax = plot_latent_digraph(graph, node_labels=["X", "Y", "L1"], show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_edge_colors(self):
        """Test custom edge colors."""
        adj = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        graph = LatentDigraph(adj, num_observed=2)

        fig, ax = plot_latent_digraph(
            graph,
            edge_color="blue",
            latent_edge_color="red",
            show=False,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestVisualizationIntegration:
    """Integration tests for visualization."""

    def test_existing_axes(self):
        """Test plotting on existing matplotlib axes."""
        L = np.array([[0, 1], [0, 0]], dtype=np.int32)
        O = np.array([[0, 1], [1, 0]], dtype=np.int32)
        graph = MixedGraph(L, O)

        fig, ax = plt.subplots(figsize=(8, 6))
        returned_fig, returned_ax = plot_mixed_graph(graph, ax=ax, show=False)

        assert returned_fig is fig
        assert returned_ax is ax
        plt.close(fig)

    def test_custom_figsize(self):
        """Test custom figure size."""
        L = np.array([[0, 1], [0, 0]], dtype=np.int32)
        O = np.array([[0, 1], [1, 0]], dtype=np.int32)
        graph = MixedGraph(L, O)

        fig, ax = plot_mixed_graph(graph, figsize=(12, 10), show=False)
        assert fig.get_size_inches()[0] == 12
        assert fig.get_size_inches()[1] == 10
        plt.close(fig)
