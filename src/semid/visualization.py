"""Graph visualization using matplotlib and igraph layouts."""

from typing import TYPE_CHECKING, Literal

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .latent_digraph import LatentDigraph
    from .mixed_graph import MixedGraph

# Type aliases
LayoutType = Literal["auto", "circle", "fr", "kk", "grid", "tree"]


def _get_layout(
    graph: ig.Graph, layout: LayoutType, num_nodes: int
) -> NDArray[np.float64]:
    """
    Compute node positions using igraph layouts.

    Args:
        graph: The igraph graph object
        layout: The layout algorithm to use
        num_nodes: Number of nodes in the graph

    Returns:
        Node positions as (x, y) coordinates, shape (num_nodes, 2)
    """
    # Auto-select layout based on graph size
    if layout == "auto":
        if num_nodes <= 10:
            layout = "circle"
        else:
            layout = "fr"

    # Compute layout using igraph
    if layout == "circle":
        pos = graph.layout_circle()
    elif layout == "fr":
        pos = graph.layout_fruchterman_reingold()
    elif layout == "kk":
        pos = graph.layout_kamada_kawai()
    elif layout == "grid":
        pos = graph.layout_grid()
    elif layout == "tree":
        pos = graph.layout_reingold_tilford()
    else:
        # Default to circle if unknown layout
        pos = graph.layout_circle()

    return np.array(pos.coords)


def _draw_directed_edges(
    ax: Axes,
    positions: NDArray[np.float64],
    adj_matrix: NDArray[np.int32],
    color: str,
    width: float,
    node_size: float = 500,
) -> None:
    """
    Draw directed edges with arrows.

    Args:
        ax: Matplotlib axes to draw on
        positions: Node positions, shape (n, 2)
        adj_matrix: Adjacency matrix for directed edges, shape (n, n)
        color: Edge color
        width: Edge width
        node_size: Size of nodes for arrow shrinking (default 500)
    """
    n = len(positions)
    # Calculate shrink amount based on node size
    # matplotlib scatter size is in points^2, so we take sqrt to get radius
    node_radius = np.sqrt(node_size) / 2

    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] == 1:
                # Check for reciprocal edge to add curvature
                is_reciprocal = adj_matrix[j, i] == 1

                arrow = FancyArrowPatch(
                    positions[i],
                    positions[j],
                    arrowstyle="->",
                    color=color,
                    linewidth=width,
                    connectionstyle=f"arc3,rad={0.2 if is_reciprocal else 0}",
                    mutation_scale=20,
                    shrinkA=node_radius,
                    shrinkB=node_radius,
                    zorder=1,
                )
                ax.add_patch(arrow)


def _draw_bidirected_edges(
    ax: Axes,
    positions: NDArray[np.float64],
    adj_matrix: NDArray[np.int32],
    color: str,
    width: float,
    node_size: float = 500,
) -> None:
    """
    Draw bidirected edges with double-headed arrows.

    Args:
        ax: Matplotlib axes to draw on
        positions: Node positions, shape (n, 2)
        adj_matrix: Adjacency matrix for bidirected edges, shape (n, n)
        color: Edge color
        width: Edge width
        node_size: Size of nodes for arrow shrinking (default 500)
    """
    n = len(positions)
    drawn = set()
    # Calculate shrink amount based on node size
    node_radius = np.sqrt(node_size) / 2

    for i in range(n):
        for j in range(i + 1, n):  # Upper triangle only
            if adj_matrix[i, j] == 1:
                if (i, j) in drawn or (j, i) in drawn:
                    continue
                drawn.add((i, j))

                # Draw double-headed arrow
                arrow = FancyArrowPatch(
                    positions[i],
                    positions[j],
                    arrowstyle="<->",
                    color=color,
                    linewidth=width,
                    mutation_scale=20,
                    shrinkA=node_radius,
                    shrinkB=node_radius,
                    zorder=1,
                )
                ax.add_patch(arrow)


def _draw_nodes(
    ax: Axes,
    positions: NDArray[np.float64],
    labels: list[str],
    size: float,
    color: str,
    shape: str,
    zorder: int = 2,
) -> None:
    """
    Draw nodes as scatter plot with labels.

    Args:
        ax: Matplotlib axes to draw on
        positions: Node positions, shape (n, 2)
        labels: Node labels
        size: Node size
        color: Node color
        shape: Node shape marker
        zorder: Z-order for layering (default 2)
    """
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=size,
        c=color,
        marker=shape,
        edgecolors="#1e293b",
        linewidths=1.5,
        zorder=zorder,
    )

    # Add labels
    for i, (x, y) in enumerate(positions):
        ax.text(
            x,
            y,
            labels[i],
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#1e293b",
            zorder=zorder + 1,
        )


def plot_mixed_graph(
    graph: MixedGraph,
    layout: LayoutType = "auto",
    node_labels: list[str] | None = None,
    node_size: float = 500,
    node_color: str = "#93c5fd",
    directed_edge_color: str = "#1e293b",
    bidirected_edge_color: str = "#dc2626",
    edge_width: float = 1.5,
    figsize: tuple[float, float] = (10, 8),
    ax: Axes | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot a mixed graph with directed and bidirected edges.

    Args:
        graph: The mixed graph to plot
        layout: Layout algorithm - "auto", "circle", "fr" (Fruchterman-Reingold),
                 "kk" (Kamada-Kawai), "grid", or "tree". Default "auto" chooses
                 "circle" for ≤10 nodes, "fr" otherwise.
        node_labels: Custom node labels. If None, uses graph.nodes (vertex_nums)
        node_size: Size of nodes (default 500)
        node_color: Color for nodes (default "#93c5fd")
        directed_edge_color: Color for directed edges (default "#1e293b")
        bidirected_edge_color: Color for bidirected edges (default "#dc2626")
        edge_width: Width of edges (default 1.5)
        figsize: Figure size if creating new figure (default (10, 8))
        ax: Matplotlib axes to plot on. If None, creates new figure
        show: Whether to call plt.show() (default True)

    Returns:
        Tuple of (matplotlib Figure, matplotlib Axes)

    Examples:
        >>> import semid
        >>> import numpy as np
        >>> L = np.array([[0, 1], [0, 0]])
        >>> O = np.array([[0, 1], [1, 0]])
        >>> graph = semid.MixedGraph(L, O)
        >>> fig, ax = semid.plot_mixed_graph(graph)

        Or use the method directly:

        >>> graph.plot()
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Get node positions using igraph layout
    positions = _get_layout(graph.directed, layout, graph.num_nodes)

    # Draw directed edges
    _draw_directed_edges(
        ax, positions, graph.d_adj, directed_edge_color, edge_width, node_size
    )

    # Draw bidirected edges
    _draw_bidirected_edges(
        ax, positions, graph.b_adj, bidirected_edge_color, edge_width, node_size
    )

    # Prepare node labels
    if node_labels is None:
        node_labels = [str(v) for v in graph.nodes]

    # Draw nodes
    _draw_nodes(ax, positions, node_labels, node_size, node_color, "o")

    # Configure axes
    ax.set_aspect("equal")
    ax.axis("off")

    # Auto-scale with padding
    margin = 0.1
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    x_range = x_max - x_min if x_max > x_min else 1
    y_range = y_max - y_min if y_max > y_min else 1
    ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
    ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)

    if show:
        plt.show()

    return fig, ax


def plot_latent_digraph(
    graph: LatentDigraph,
    layout: LayoutType = "auto",
    node_labels: list[str] | None = None,
    observed_node_size: float = 500,
    latent_node_size: float = 400,
    observed_node_color: str = "#93c5fd",
    latent_node_color: str = "#d1d5db",
    observed_node_shape: str = "o",
    latent_node_shape: str = "s",
    edge_color: str = "#1e293b",
    latent_edge_color: str = "#dc2626",
    edge_width: float = 1.5,
    show_latent: bool = True,
    figsize: tuple[float, float] = (10, 8),
    ax: Axes | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot a latent digraph distinguishing observed and latent nodes.

    Args:
        graph: The latent digraph to plot
        layout: Layout algorithm - "auto", "circle", "fr", "kk", "grid", or "tree"
                 (default "auto")
        node_labels: Custom node labels. If None, uses numeric indices
        observed_node_size: Size of observed nodes (default 500)
        latent_node_size: Size of latent nodes (default 400)
        observed_node_color: Color for observed nodes (default "#93c5fd")
        latent_node_color: Color for latent nodes (default "#d1d5db")
        observed_node_shape: Shape for observed nodes (default "o" for circle)
        latent_node_shape: Shape for latent nodes (default "s" for square)
        edge_color: Color for edges from observed nodes (default "#1e293b")
        latent_edge_color: Color for edges from latent nodes (default "#dc2626")
        edge_width: Width of edges (default 1.5)
        show_latent: Whether to display latent nodes (default True)
        figsize: Figure size (default (10, 8))
        ax: Matplotlib axes to plot on. If None, creates new figure
        show: Whether to call plt.show() (default True)

    Returns:
        Tuple of (matplotlib Figure, matplotlib Axes)

    Examples:
        >>> import semid
        >>> import numpy as np
        >>> adj = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        >>> graph = semid.LatentDigraph(adj, num_observed=2)
        >>> fig, ax = semid.plot_latent_digraph(graph)

        Or use the method directly:

        >>> graph.plot()
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Get observed and latent node indices
    observed_nodes = graph.observed_nodes()
    latent_nodes = graph.latent_nodes()

    # Get node positions using igraph layout
    if show_latent:
        # Use full graph for layout
        positions = _get_layout(graph.digraph, layout, graph.num_nodes)
    else:
        # Only layout observed nodes
        if len(observed_nodes) > 0:
            # Create subgraph with only observed nodes
            obs_subgraph = graph.digraph.induced_subgraph(observed_nodes)
            obs_positions = _get_layout(obs_subgraph, layout, len(observed_nodes))
            # Create full position array (latent positions won't be used)
            positions = np.zeros((graph.num_nodes, 2))
            positions[observed_nodes] = obs_positions
        else:
            positions = np.zeros((graph.num_nodes, 2))

    # Draw edges with different colors based on source node type
    n = graph.num_nodes
    # Calculate shrink amounts for observed and latent nodes
    obs_node_radius = np.sqrt(observed_node_size) / 2
    lat_node_radius = np.sqrt(latent_node_size) / 2

    for i in range(n):
        for j in range(n):
            if graph.adj[i, j] == 1:
                # Skip edges involving latent nodes if show_latent is False
                if not show_latent and (i in latent_nodes or j in latent_nodes):
                    continue

                # Check if source is latent or observed
                is_from_latent = i in latent_nodes
                color = latent_edge_color if is_from_latent else edge_color

                # Check for reciprocal edge to add curvature
                is_reciprocal = graph.adj[j, i] == 1

                # Determine shrink amounts based on node types
                shrink_a = lat_node_radius if i in latent_nodes else obs_node_radius
                shrink_b = lat_node_radius if j in latent_nodes else obs_node_radius

                arrow = FancyArrowPatch(
                    positions[i],
                    positions[j],
                    arrowstyle="->",
                    color=color,
                    linewidth=edge_width,
                    connectionstyle=f"arc3,rad={0.2 if is_reciprocal else 0}",
                    mutation_scale=20,
                    shrinkA=shrink_a,
                    shrinkB=shrink_b,
                    zorder=1,
                )
                ax.add_patch(arrow)

    # Prepare node labels
    if node_labels is None:
        node_labels = [str(i) for i in range(graph.num_nodes)]

    # Draw observed nodes
    if len(observed_nodes) > 0:
        obs_positions = positions[observed_nodes]
        obs_labels = [node_labels[i] for i in observed_nodes]
        _draw_nodes(
            ax,
            obs_positions,
            obs_labels,
            observed_node_size,
            observed_node_color,
            observed_node_shape,
            zorder=2,
        )

    # Draw latent nodes if requested
    if show_latent and len(latent_nodes) > 0:
        lat_positions = positions[latent_nodes]
        lat_labels = [node_labels[i] for i in latent_nodes]
        _draw_nodes(
            ax,
            lat_positions,
            lat_labels,
            latent_node_size,
            latent_node_color,
            latent_node_shape,
            zorder=2,
        )

    # Configure axes
    ax.set_aspect("equal")
    ax.axis("off")

    # Auto-scale with padding
    margin = 0.1
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    x_range = x_max - x_min if x_max > x_min else 1
    y_range = y_max - y_min if y_max > y_min else 1
    ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
    ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)

    if show:
        plt.show()

    return fig, ax
