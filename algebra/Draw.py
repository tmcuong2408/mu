import os
import sys
from typing import Any, Optional, Tuple, Union, List, Dict
from fractions import Fraction

# Setup non-GUI backend when no display is found
_has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
if not _has_display:
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        pass

import matplotlib.pyplot as plt


def _extract_xy(pts: List[Tuple[Any, Any]]) -> Tuple[List[float], List[float]]:
    """Converts coordinate pairs to float values for plotting."""
    xs, ys = [], []
    for x, y in pts:
        try:
            x_f = float(x.real) if isinstance(x, complex) else float(x)
            y_f = float(y.real) if isinstance(y, complex) else float(y)
            xs.append(x_f)
            ys.append(y_f)
        except Exception:
            continue
    return xs, ys


def draw_ascii(
    graph_obj: Any,
    width: int = 50,
    height: int = 15,
    title: Optional[str] = None,
) -> str:
    """
    Renders an ASCII visualization of the graph in the terminal.
    Useful in headless environments (e.g. servers, WSL, Docker, CI/CD).
    """
    from .UncertainGraph import UncertainGraph, CompleteGraph, pgr

    if not hasattr(graph_obj, "points") and not hasattr(graph_obj, "all_points"):
        graph_obj = pgr(graph_obj, None)

    pts = graph_obj.all_points() if hasattr(graph_obj, "all_points") else graph_obj.points
    if not pts:
        return "[Empty Graph: No points to display]"

    xs, ys = _extract_xy(pts)
    if not xs or not ys:
        return "[Unable to convert points to numeric coordinates]"

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    span_x = max_x - min_x if max_x != min_x else 1.0
    span_y = max_y - min_y if max_y != min_y else 1.0

    # Initialize character grid
    grid = [[" " for _ in range(width)] for _ in range(height)]

    for x, y in zip(xs, ys):
        col = int((x - min_x) / span_x * (width - 1))
        row = height - 1 - int((y - min_y) / span_y * (height - 1))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        grid[row][col] = "*"

    lines = []
    plot_title = title or "Uncertain Graph Visualization (ASCII)"
    lines.append(f"--- {plot_title} ---")
    lines.append(f"  y_max = {max_y:.2f}")

    for r in range(height):
        prefix = " | "
        row_str = "".join(grid[r])
        lines.append(prefix + row_str)

    lines.append(" +-" + "-" * width + "-> X")
    lines.append(f"  x_min = {min_x:.2f}" + " " * max(1, width - 25) + f"x_max = {max_x:.2f}")
    lines.append(f"  y_min = {min_y:.2f}")
    lines.append(f"  Total Scenarios: {len(pts)}")

    result_str = "\n".join(lines)
    return result_str


def draw(
    graph_obj: Any,
    input_val: Optional[Any] = None,
    space: str = "m",
    title: Optional[str] = None,
    xlabel: str = "X (Input Scenarios)",
    ylabel: str = "f(X) (Output Scenarios)",
    show_intervals: bool = True,
    show_points: bool = True,
    connect_curves: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (9, 6),
    ax: Optional[plt.Axes] = None,
    return_fig: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Draws the calculated uncertain graph using matplotlib.
    
    Features:
        - Point-wise discrete scenario scattering with modern aesthetics.
        - Vertical uncertainty intervals spanning [min y, max y] at each x.
        - Support for both UncertainGraph (pgr) and CompleteGraph (gr).
        - Headless execution safety (saves or prints ASCII fallback if no display).
        - Returns (fig, ax) for programmatic inspection.

    Args:
        graph_obj: An UncertainGraph, CompleteGraph, or a callable f (if input_val is also given).
        input_val: Optional input uncertain number if graph_obj is a function.
        space: Arithmetic space ("m" or "pw") if creating graph on-the-fly.
        title: Plot title string.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        show_intervals: Whether to draw vertical uncertainty bars for Minkowski ranges.
        show_points: Whether to plot discrete points.
        connect_curves: Whether to connect points if each x has a unique y.
        save_path: File path to save image (e.g. "plot.png").
        show: Whether to call plt.show() if GUI display is available.
        figsize: Size of the matplotlib figure.
        ax: Optional existing matplotlib Axes to draw on.
        return_fig: Whether to return the (fig, ax) tuple.
    """
    from .UncertainGraph import UncertainGraph, CompleteGraph, pgr

    # Handle shorthand call: draw(f, X)
    if callable(graph_obj) and input_val is not None:
        graph_obj = pgr(graph_obj, input_val, space=space)

    # Determine graph type
    is_complete = isinstance(graph_obj, CompleteGraph)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=120)
        created_fig = True
    else:
        fig = ax.figure

    # Styling settings
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#ffffff")
    ax.grid(True, linestyle="--", linewidth=0.7, color="#cbd5e1", alpha=0.7, zorder=1)

    palette = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#db2777"]

    if is_complete:
        subgraphs = graph_obj.subgraphs
        for idx, subg in enumerate(subgraphs):
            pts = subg.points
            if not pts:
                continue
            xs, ys = _extract_xy(pts)
            c = palette[idx % len(palette)]
            lbl = f"X_{idx + 1}"
            _plot_subgraph(ax, xs, ys, c, lbl, show_intervals, show_points, connect_curves)
    else:
        pts = graph_obj.points if hasattr(graph_obj, "points") else []
        xs, ys = _extract_xy(pts)
        c = palette[0]
        space_name = getattr(graph_obj, "space_type", "minkowski").capitalize()
        lbl = f"Scenarios ({space_name})"
        _plot_subgraph(ax, xs, ys, c, lbl, show_intervals, show_points, connect_curves)

    # Title & Labels
    default_title = "Graph of Uncertain Function $pgr_f(A) = A \\times f(A)$"
    ax.set_title(title or default_title, fontsize=13, fontweight="bold", pad=12, color="#0f172a")
    ax.set_xlabel(xlabel, fontsize=11, fontweight="medium", color="#1e293b")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="medium", color="#1e293b")

    # Spines styling
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")
        spine.set_linewidth(1.0)

    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9, loc="best")
    plt.tight_layout()

    # Save to file if path requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Graph plot saved successfully to: {save_path}")

    # Display or fallback
    if show:
        if _has_display:
            try:
                plt.show()
            except Exception:
                print(draw_ascii(graph_obj, title=title))
        else:
            # Print ASCII visualization in terminal when headless
            print(draw_ascii(graph_obj, title=title))

    if return_fig:
        return fig, ax
    return None


def _plot_subgraph(
    ax: plt.Axes,
    xs: List[float],
    ys: List[float],
    color: str,
    label: str,
    show_intervals: bool,
    show_points: bool,
    connect_curves: bool,
) -> None:
    """Helper to plot points and uncertainty intervals for a single subgraph."""
    if not xs or not ys:
        return

    # Group Y values by X
    from collections import defaultdict
    x_to_y: Dict[float, List[float]] = defaultdict(list)
    for x, y in zip(xs, ys):
        x_to_y[x].append(y)

    unique_xs = sorted(x_to_y.keys())

    # Draw vertical uncertainty interval lines if multiple Y per X
    has_intervals = any(len(x_to_y[x]) > 1 for x in unique_xs)
    if show_intervals and has_intervals:
        for x in unique_xs:
            y_vals = x_to_y[x]
            y_min = min(y_vals)
            y_max = max(y_vals)
            if y_min < y_max:
                ax.vlines(
                    x, y_min, y_max,
                    colors=color,
                    linestyles="-",
                    linewidth=2.0,
                    alpha=0.6,
                    zorder=3,
                )
                cap_w = (max(unique_xs) - min(unique_xs)) * 0.02 if len(unique_xs) > 1 else 0.1
                ax.hlines(
                    [y_min, y_max],
                    x - cap_w, x + cap_w,
                    colors=color,
                    linewidth=1.8,
                    alpha=0.7,
                    zorder=3,
                )

    # If each x has exactly one y, optionally connect with a curve
    if connect_curves and not has_intervals and len(unique_xs) > 1:
        sorted_pairs = sorted(zip(xs, ys), key=lambda p: p[0])
        px = [p[0] for p in sorted_pairs]
        py = [p[1] for p in sorted_pairs]
        ax.plot(px, py, color=color, linestyle="-", linewidth=2.0, alpha=0.8, zorder=4)

    # Plot discrete scenario points
    if show_points:
        ax.scatter(
            xs, ys,
            color=color,
            edgecolors="#0f172a",
            linewidths=0.8,
            s=65,
            alpha=0.9,
            zorder=5,
            label=label,
        )


def draw_ast(graph_obj: Any, as_text: bool = True) -> str:
    """
    Renders the Abstract Syntax Tree (AST) of the Uncertain Graph.
    Shows the structural hierarchy of input domains and operators.
    """
    ast = getattr(graph_obj, "ast", {})
    if not ast:
        return "[AST]: No AST node found."

    def _ast_to_lines(node: Any, prefix: str = "", is_last: bool = True) -> List[str]:
        if not isinstance(node, dict):
            return [prefix + ("└── " if is_last else "├── ") + str(node)]

        node_type = node.get("type", "unknown")
        space = node.get("space_type", "")
        sym = node.get("operator_symbol", "")
        header = f"[{node_type}]"
        if space:
            header += f" space={space}"
        if sym:
            header += f" sym='{sym}'"

        lines = [prefix + ("└── " if is_last else "├── ") + header]
        new_prefix = prefix + ("    " if is_last else "│   ")

        children = []
        if "left" in node:
            children.append(("left", getattr(node["left"], "ast", node["left"])))
        if "right" in node:
            children.append(("right", getattr(node["right"], "ast", node["right"])))
        if "children" in node:
            for i, c in enumerate(node["children"]):
                children.append((f"child_{i}", c))

        for idx, (role, child) in enumerate(children):
            last_child = (idx == len(children) - 1)
            lines.append(new_prefix + f"({role})")
            lines.extend(_ast_to_lines(child, new_prefix, is_last=last_child))

        return lines

    lines = ["AST Hierarchy:"] + _ast_to_lines(ast, prefix="", is_last=True)
    res = "\n".join(lines)
    if as_text:
        print(res)
    return res
