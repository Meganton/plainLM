import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Tuple, Any
import matplotlib
import seaborn as sns
import warnings
import copy
import json
from pathlib import Path
from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pandas")
pd.set_option("future.no_silent_downcasting", True)

STYLE_DICT_DEFAULT = {
        "figsize": None,
        "title": "Experiment Results",
        "x_label": "Evaluations",
        "y_label": "Best Score",
        "x_scale": "linear",
        "y_scale": "linear",
        "x_limit_left": None,
        "x_limit_right": None,
        "y_limit_bottom": None,
        "y_limit_top": None,
        "title_fontsize": 16,
        "x_label_fontsize": 14,
        "y_label_fontsize": 14,
        "axis_ticks": "both",
        "legend_marker_width": 2.2,
        "legend_marker_length": 1.8,
        "legend_style": "outside",
        "outside_legend_n_cols": None,
        "right_hand_legend_line_style": "bezier",
    }


def combined_lower_legend(
    fig: matplotlib.figure.Figure,
    axes: np.ndarray[matplotlib.axes.Axes] | matplotlib.axes.Axes,
    marker_width: float = STYLE_DICT_DEFAULT["legend_marker_width"],
    marker_length: float = STYLE_DICT_DEFAULT["legend_marker_length"],
    n_cols: int | None = None,
):
    """Combine legends from multiple subplots into a single legend.
    This function collects all unique labels and handles
    from the subplots and creates a single legend at the bottom of the figure.

    Args:
        fig: The figure object containing the subplots.
        axes: The axes of the subplots from which to collect legends.
        marker_width: Width of the line markers. 1 means identical scale to the lines in 
        the plot, 2.2 is a good value for easy visibility.
        marker_length: Length of the line markers. 1.8 gives 2 dashes, 5 gives 4 dashes
    Returns:
        Tuple containing the figure and axes objects with the combined legend.
    """
    legend_dict = {}
    if isinstance(axes, matplotlib.axes.Axes):
        axes = np.array([[axes]])
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend().remove()  # Remove the legend from each subplot
        for handle, label in zip(handles, labels):
            if label not in legend_dict:
                new_handle = copy.copy(handle)
                new_handle.set_linewidth(marker_width)
                new_handle.set_markersize(5)
                legend_dict[label] = new_handle
    fig.legend(
        handles=legend_dict.values(),
        labels=legend_dict.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=np.ceil(np.sqrt(len(legend_dict.keys()))) if n_cols is None else n_cols,
        frameon=True,
        handlelength=marker_length,
    )
    return fig, axes


def right_hand_legend(
    ax: matplotlib.axes.Axes,
    final_positions: dict,
    group_colors: dict,
    group_styles: dict = {},
    line_width: float = 1.5,
    line_style: str = STYLE_DICT_DEFAULT["right_hand_legend_line_style"],
) -> matplotlib.axes.Axes:
    """
    Place legend labels on the right side of the plot at their final y-positions.
    Automatically spaces labels to avoid overlaps using force-directed algorithm.
    Handles both linear and log scales.
    
    Args:
        ax: The matplotlib axes object
        final_positions: Dictionary mapping group names to their final y-values
        group_colors: Dictionary mapping group names to their colors
        group_styles: Optional dictionary with additional styling (linestyle, etc.)
        line_width: Width of connecting lines
        line_style: "bezier" for curved lines (default) or "straight" for line segments
    Returns:
        The matplotlib axes object with the right-hand legend added.
    """
    from matplotlib.path import Path as MPath
    import matplotlib.patches as patches
    
    # Filter out infinite values
    finite_positions = {group: pos for group, pos in final_positions.items() 
                       if np.isfinite(pos)}
    
    if not finite_positions:
        return
    
    # Get y-axis limits and scale
    y_min, y_max = ax.get_ylim()
    is_log_scale = ax.get_yscale() == "log"
    
    # Helper functions for log/linear space conversion
    to_calc = lambda val: np.log10(val) if is_log_scale and val > 0 else val
    from_calc = lambda val: 10 ** val if is_log_scale else val
    
    calc_min, calc_max = to_calc(y_min), to_calc(y_max)
    calc_range = calc_max - calc_min
    
    # Sort groups by position and initialize label positions
    sorted_groups = sorted(finite_positions.items(), key=lambda x: to_calc(x[1]))
    label_positions = {group: to_calc(pos) for group, pos in sorted_groups}
    
    # Calculate minimum spacing
    min_spacing = max(calc_range * 0.12, calc_range / (len(sorted_groups) * 1.5))
    
    # Force-directed spacing algorithm
    for _ in range(50):
        if not any(label_positions[sorted_groups[i+1][0]] - label_positions[sorted_groups[i][0]] < min_spacing
                   for i in range(len(sorted_groups) - 1)):
            break
        for i in range(len(sorted_groups) - 1):
            g1, g2 = sorted_groups[i][0], sorted_groups[i + 1][0]
            gap = label_positions[g2] - label_positions[g1]
            if gap < min_spacing:
                overlap = min_spacing - gap
                label_positions[g1] -= overlap / 2
                label_positions[g2] += overlap / 2
    
    # Squeeze labels to fit within bounds
    positions = [label_positions[g] for g, _ in sorted_groups]
    current_min, current_max = min(positions), max(positions)
    current_span = current_max - current_min
    
    padding = calc_range * 0.05
    available_min, available_max = calc_min + padding, calc_max - padding
    
    if current_span > available_max - available_min or current_min < available_min or current_max > available_max:
        if current_span > 0:
            for group in label_positions:
                norm = (label_positions[group] - current_min) / current_span
                label_positions[group] = available_min + norm * (available_max - available_min)
    
    # Draw connecting lines and labels
    x_max = ax.get_xlim()[1]
    x_label_end = x_max * 1.1
    
    for group, label_pos_calc in label_positions.items():
        label_pos = from_calc(label_pos_calc)
        final_pos = final_positions[group]
        color = group_colors[group]
        linestyle = group_styles.get(group, {}).get("linestyle", "-")
        
        if line_style == "bezier":
            # Bezier curve with zero slope at endpoints
            x_ctrl_1 = x_max + (x_label_end - x_max) * 0.3
            x_ctrl_2 = x_label_end - (x_label_end - x_max) * 0.3
            
            verts = [(x_max, final_pos), (x_ctrl_1, final_pos), (x_ctrl_2, label_pos), (x_label_end, label_pos)]
            codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]
            matchpath = MPath(verts, codes)
            patch = patches.PathPatch(matchpath, facecolor='none', edgecolor=color, 
                                     linewidth=line_width, linestyle=linestyle, 
                                     alpha=1.0, clip_on=False)
            ax.add_patch(patch)
        elif line_style == "straight":
            # Straight line
            ax.plot([x_max, x_label_end], [final_pos, label_pos],
                   linestyle=linestyle, linewidth=line_width, color=color,
                   clip_on=False, alpha=1.0)
        
        # Draw label text
        text_color = group_styles.get(group, {}).get("text_color", "black")
        label_text = group_styles.get(group, {}).get("label", group)
        ax.text(x_label_end * 1.01, label_pos, f' {label_text}',
               verticalalignment='center', fontsize='large',
               color=text_color, clip_on=False, fontweight='bold')
    return ax


def activate_plot_style():
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    plt.rcParams.update({
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.sans-serif": ["DejaVu Sans"],
        "font.monospace": ["DejaVu Sans Mono"],
        "font.size": "10.90",
        "legend.fontsize": "10.90",
        "xtick.labelsize": "small",
        "ytick.labelsize": "small",
        "legend.title_fontsize": "small",
        "axes.formatter.useoffset": False,
        "pgf.preamble": r"""
            \usepackage[T1]{fontenc}
            \usepackage[utf8x]{inputenc}
            \usepackage{microtype}
            """,
    })


def _style_plot(
    ax: matplotlib.axes.Axes,
    x_axis: Literal["Evaluations", "Cost", "Fidelities"] = "Evaluations",
    style_dict: dict[str, Any] = {},
    plot_type: Literal["loss", "ranks"] = "loss"
):
    style_dict = {**STYLE_DICT_DEFAULT, **style_dict}
    if style_dict.get("title"):
        ax.set_title(style_dict.get("title", ""), fontsize=style_dict.get("title_fontsize", 16))
    if style_dict.get("y_label"):
        ax.set_ylabel(style_dict.get("y_label", ""), fontsize=style_dict.get("y_label_fontsize", 14))
    if style_dict.get("x_label"):
        ax.set_xlabel(style_dict.get("x_label", ""), fontsize=style_dict.get("x_label_fontsize", 14))
    ax.set_xscale(style_dict.get("x_scale", "linear"))
    ax.set_yscale(style_dict.get("y_scale", "linear"))
    sns.despine(ax=ax)
    x_limits = ax.get_xlim()
    x_tick_positions = ax.get_xticks()
    x_tick_positions = np.array(
        [pos for pos in x_tick_positions if x_limits[0] <= pos <= x_limits[1]]
    )
    x_tick_label_positions = [str(round(pos, 2) if int(pos) != pos else int(pos)) for pos in x_tick_positions]
    ax.set_xticks([float(pos) for pos in x_tick_label_positions])
    ax.tick_params(axis='both',
                   which='major',
                   direction='out',
                   bottom=style_dict.get("axis_ticks", "both") in ("both", "x"),
                   left=style_dict.get("axis_ticks", "both") in ("both", "y"),
                   width=0.6)
    ax.set_xticklabels(
        x_tick_label_positions
    )
    if x_axis == "Fidelities":
        ax.set_xticklabels([lab.get_text() + "x" for lab in ax.get_xticklabels()])
    ax.grid(visible=True, which="both", ls="-", alpha=0.7)
    # Make axes black and visible
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    if plot_type == "loss":
        # Set the number of y-ticks to be the minimum of 5 and the previously set number
        n_ticks = max(4, min(5, len(ax.get_yticks())))
        if ax.get_yscale() in ( "log", "symlog" ):
            ax.locator_params(axis="y", numticks=n_ticks)
        else:
            ax.locator_params(axis="y", nbins=n_ticks)

    # Set y-ticks now to prevent them from being reset
    elif plot_type == "ranks":
        ax.set_yticks(
            range(max(1, int(ax.get_ylim()[0])), int(np.floor(ax.get_ylim()[1])) + 1)
        )


def _extract_results(
    path: Path | str,
    replace_inf = True,
    use_fidelity: bool = False,
    only_best: bool = False
):
    """
    Extract results from a NEPS experiment directory.
    """
    path = Path(path)
    try:
        with open(path.with_suffix(".json"), 'r') as f:
            run_data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load JSON data from {path.with_suffix('.json')}: {e}")
    
    incumbents = run_data["incumbent_history"]
    cumulated_costs = run_data["cumulated_cost_history"]
    index_indicator = "Cumulative cost"

    if use_fidelity:
        fidelity_history = run_data.get("cumulated_fidelity_history")
        # Some runs store this key but leave it empty; fall back to cost history.
        if isinstance(fidelity_history, list) and len(fidelity_history) > 0:
            cumulated_costs = fidelity_history
            index_indicator = "Cumulative fidelity"

    if len(incumbents) != len(cumulated_costs):
        min_len = min(len(incumbents), len(cumulated_costs))
        if min_len == 0:
            raise ValueError(
                f"Mismatched history lengths in {path.with_suffix('.json')}: "
                f"len(incumbent_history)={len(incumbents)}, "
                f"len({index_indicator.lower().replace(' ', '_')}_history)={len(cumulated_costs)}"
            )
        incumbents = incumbents[:min_len]
        cumulated_costs = cumulated_costs[:min_len]

    # Create DataFrame
    df = pd.DataFrame({"Objective to minimize": incumbents, index_indicator: cumulated_costs})
    df = df.sort_values(by=index_indicator)
    max_cost_run = df[index_indicator].max()
    # df.drop_duplicates(inplace=True)
    if replace_inf:
        df["Objective to minimize"] = df["Objective to minimize"].replace(to_replace=[np.inf, -np.inf], value=np.nan)
        df["Objective to minimize"] = df["Objective to minimize"].replace([float("inf"), -float("inf")], np.nan).dropna()
    df.dropna(axis=0,subset=["Objective to minimize"], inplace=True)
    if df[index_indicator].max() < max_cost_run:
        df.loc[max_cost_run] = df["Objective to minimize"].min()
        df = df.sort_values(by=index_indicator)
    if only_best:
        # Set all objective values to the minimum, keep all cost points
        best_obj = df["Objective to minimize"].min()
        df["Objective to minimize"] = best_obj
    if df[index_indicator].isna().any():
        raise ValueError(f"{index_indicator} contains NaN values")
    df.set_index(index_indicator, inplace=True)
    return df


def plot_results(
    path: Path | str,
    groups: list[str],
    groupstyles: dict,
    seeds: int,
    benchmarks: list[str] | None = None,
    ax: matplotlib.axes.Axes = None,
    fig: matplotlib.figure.Figure = None,
    x_axis: Literal["Evaluations", "Cost", "Fidelities"] = "Evaluations",
    plot_type: Literal["loss", "ranks"] = "loss",
    evaluation_cost: int | float | None = None,
    fidelity_cost: int | float | None = None,
    max_fidelity: int | float | None = None,
    only_best: list[str] = [],
    style_dict: dict[str, Any] = STYLE_DICT_DEFAULT,
) -> matplotlib.axes.Axes | Tuple[matplotlib.axes.Axes, matplotlib.figure.Figure]:
    """
    Plot results from multiple NEPS experiment directories.
    It works in-place on the provided axes and figure or creates new ones if none are provided.
    They will also be returned for further modification or saving.

    Args:
        path: The base path to the experiment directories.
        groups: A list of group names to plot.
        groupstyles: A dictionary mapping group names to their styles.
        seeds: The number of seeds to include in the plot.
        benchmarks: A list of benchmark names to include in the plot (optional).
        ax: The matplotlib axes to plot on (optional).
        fig: The matplotlib figure to plot on (optional).
        x_axis: The x-axis to use for the plot (default: "Evaluations").
        plot_type: The type of the plot (default: "loss").
        evaluation_cost: The cost per full evaluation (default: None).
        fidelity_cost: The cost per maximum fidelity (default: None).
        max_fidelity: The maximum fidelity level (default: None).
        style_dict: A dictionary of style parameters for the plot (optional).
            Includes title, x_label, y_label, x_scale, y_scale, x_limit_left,
            x_limit_right, y_limit_bottom, y_limit_top, title_fontsize,
            x_label_fontsize, y_label_fontsize, axis_ticks, legend_marker_width,
            legend_marker_length, legend_style. See STYLE_DICT_DEFAULT for defaults.
    Returns:
        The ax, if only an ax was given, otherwise the matplotlib figure and axes used for the plot.
    """
    activate_plot_style()
    style_dict = {**STYLE_DICT_DEFAULT, **style_dict}
    if ax is None:
        match style_dict["legend_style"]:
            case "outside":
                fig_size = (6, 4)
            case "both":
                fig_size = (8, 5)
            case "none":
                fig_size = (6, 3.5)
            case _:
                fig_size = (8, 4)
        fig, ax = plt.subplots(figsize=style_dict["figsize"] if style_dict["figsize"] else fig_size)
    path = Path(path)
    all_results = {}
    unique_cost_values = set()
    min_cost_value = 0
    max_cost = float("inf")
    if x_axis == "Evaluations" and evaluation_cost is None:
        evaluation_cost = 1
    for group in groups:
        all_results[group] = {}
        for seed in range(seeds):
            result = None
            if benchmarks is None or len(benchmarks) == 0:
                result = _extract_results(path / f"results/{group}_{seed}", only_best=group in only_best, use_fidelity=max_fidelity is not None)
                all_results[group][f"seed {seed}"] = result
            else:
                assert (
                    isinstance(benchmarks, list) and len(benchmarks) > 0
                ), "benchmarks must be a non-empty list when provided"
                for benchmark in benchmarks:
                    assert (path / f"{benchmark}/results/{group}_{seed}.json").exists(), (
                        f" {path / f'{benchmark}/results/{group}_{seed}'} does not"
                        " exist"
                    )
                    result = _extract_results(path / f"{benchmark}/results/{group}_{seed}", only_best=group in only_best, use_fidelity=max_fidelity is not None)
                    all_results[group][f"{benchmark} - seed {seed}"] = result
            assert result is not None, f"No results extracted for group {group}, seed {seed}"
            unique_cost_values.update(result.index.values)
    unique_cost_values = sorted(unique_cost_values)

    max_cost = max(unique_cost_values)
    min_cost_value = min(unique_cost_values)

    for group, group_df in all_results.items():
        for seed, df in group_df.items():
            if group in only_best:
                if df.index.max() < max_cost:
                    last_value = df.iloc[-1]["Objective to minimize"]
                    df.loc[max_cost] = last_value
                    df = df.sort_index()
                    all_results[group][seed] = df

    plot_data = {}
    all_data = pd.DataFrame(index=unique_cost_values)
    if x_axis != "Cost":
        if x_axis == "Evaluations":
            assert (
                evaluation_cost is not None
            ), "evaluation_cost must be provided when x_axis is 'Evaluations'"
            if max_fidelity is None:
                all_data = all_data.reindex(np.array(all_data.index.values) / evaluation_cost)
            else:
                all_data = all_data.reindex(np.array(all_data.index.values) / max_fidelity)
        elif x_axis == "Fidelities":
            assert fidelity_cost is not None or max_fidelity is not None, (
                "fidelity_cost and max_fidelity must be provided when x_axis is"
                " 'Fidelities'"
            )
            if max_fidelity is not None:
                all_data = all_data.reindex(np.array(all_data.index.values) / max_fidelity)
            else:
                all_data = all_data.reindex(np.array(all_data.index.values) / fidelity_cost)
    for group, group_df in all_results.items():
        for seed, df in group_df.items():
            all_data[str(group) + " - " + str(seed)] = (
                df.reindex(unique_cost_values).ffill(limit_area="inside").values
            )
    if plot_type == "ranks":
        unique_cost_values = [
            v for v in unique_cost_values if max_cost >= v >= min_cost_value
        ]
        for seed in range(seeds):
            seed_cols = sorted(
                [col for col in all_data.columns if col.endswith(f"- seed {seed}")]
            )
            # Assert, that there are an equal amount of seeds for each group
            if not benchmarks:
                assert len(seed_cols) == len(groups), f"Not all groups have a seed {seed}"
            # if there are benchmarks, assert that there are an equal amount of seeds for each benchmark
            if benchmarks is not None and len(benchmarks) > 0:
                for benchmark in benchmarks:
                    benchmark_seed_cols = [
                        col for col in seed_cols if col.split(" - ")[-2] == benchmark
                    ]
                    assert len(benchmark_seed_cols) == len(
                        groups
                    ), f"Not all groups have a seed {seed} for benchmark {benchmark}"
                    all_data[benchmark_seed_cols] = all_data[benchmark_seed_cols].rank(
                        axis=1, method="average"
                    )
            else:
                all_data[seed_cols] = all_data[seed_cols].rank(axis=1, method="average")
    for group in groups:
        group_cols = [col for col in all_data.columns if col.startswith(f"{group} - ")]
        plot_data[group] = {
            "mean": all_data[group_cols].mean(axis=1, skipna=True),
            "sem": all_data[group_cols].sem(axis=1, skipna=True).replace(np.nan, 0.0),
        }

    min_y_value = float("inf")
    max_y_value = float("-inf")
    min_x_value = float("inf")
    max_x_value = float("-inf")
    final_positions = {}  # Track final y-position for each group
    
    for group, data in plot_data.items():
        mean_series = data["mean"].copy()
        sem_series = data["sem"].copy()

        x_limit_left = style_dict["x_limit_left"]
        x_limit_right = style_dict["x_limit_right"]

        # If clipping on the left would remove the segment crossing x_limit_left,
        # insert a boundary point with the last value from the lower-x side.
        if x_limit_left is not None:
            has_lower = (mean_series.index < x_limit_left).any()
            has_upper_or_equal = (mean_series.index >= x_limit_left).any()
            if has_lower and has_upper_or_equal and x_limit_left not in mean_series.index:
                left_source_idx = mean_series.index[mean_series.index < x_limit_left][-1]
                mean_series.loc[x_limit_left] = mean_series.loc[left_source_idx]
                sem_series.loc[x_limit_left] = sem_series.loc[left_source_idx]

        # If clipping on the right would remove the segment crossing x_limit_right,
        # insert a boundary point with the last value from the lower-x side.
        if x_limit_right is not None:
            has_upper = (mean_series.index > x_limit_right).any()
            has_lower_or_equal = (mean_series.index <= x_limit_right).any()
            if has_upper and has_lower_or_equal and x_limit_right not in mean_series.index:
                right_source_idx = mean_series.index[mean_series.index <= x_limit_right][-1]
                mean_series.loc[x_limit_right] = mean_series.loc[right_source_idx]
                sem_series.loc[x_limit_right] = sem_series.loc[right_source_idx]

        mean_series = mean_series.sort_index()
        sem_series = sem_series.sort_index()

        if x_limit_left is not None:
            mean_series = mean_series[mean_series.index >= x_limit_left]
            sem_series = sem_series[sem_series.index >= x_limit_left]
        if x_limit_right is not None:
            mean_series = mean_series[mean_series.index <= x_limit_right]
            sem_series = sem_series[sem_series.index <= x_limit_right]

        # Only plot from the first non-nan value to the last non-nan value
        first_valid_index = mean_series.first_valid_index()
        last_valid_index = mean_series.last_valid_index()
        mean = mean_series.loc[first_valid_index:last_valid_index]
        sem = sem_series.loc[first_valid_index:last_valid_index]

        ax.step(
            mean.index,
            mean.values,
            where="post",
            label=groupstyles[group]["label"],
            color=groupstyles[group]["color"],
            linestyle=groupstyles[group].get("linestyle", "-"),
        )
        ax.fill_between(
            mean.index,
            mean - sem,
            mean + sem,
            alpha=0.1,
            step="post",
            color=groupstyles[group]["color"],
            linestyle=groupstyles[group].get("linestyle", "-"),
        )
        min_y_value = min(min_y_value, (mean - sem).min())
        max_y_value = max(max_y_value, (mean + sem).max())
        min_x_value = min(min_x_value, mean.index.min())
        max_x_value = max(max_x_value, mean.index.max())
        
        # Track final position for right-hand legend
        if len(mean) > 0:
            final_positions[group] = mean.iloc[-1]
            if style_dict["y_limit_bottom"] is not None:
                final_positions[group] = max(style_dict["y_limit_bottom"], final_positions[group])
            if style_dict["y_limit_top"] is not None:
                final_positions[group] = min(style_dict["y_limit_top"], final_positions[group])
    # Set x-axis limits
    x_lim_left = style_dict["x_limit_left"] if style_dict["x_limit_left"] is not None else min_x_value
    x_lim_right = style_dict["x_limit_right"] if style_dict["x_limit_right"] is not None else int(np.ceil(max_x_value))
    
    # Synchronize with shared x-axis siblings
    x_siblings = ax.get_shared_x_axes().get_siblings(ax)
    if len(x_siblings) > 1:
        for sibling_ax in x_siblings:
            if sibling_ax != ax:
                sibling_xlim = sibling_ax.get_xlim()
                x_lim_left = min(x_lim_left, sibling_xlim[0])
                x_lim_right = max(x_lim_right, sibling_xlim[1])
    
    ax.set_xlim(left=x_lim_left, right=x_lim_right)
    
    # Set y-axis limits
    if plot_type == "ranks":
        y_lim_bottom = np.floor(min_y_value) - (0.5 if min_y_value - np.floor(min_y_value) < 0.25 else 0) if style_dict["y_limit_bottom"] is None else style_dict["y_limit_bottom"]
        y_lim_top = np.ceil(max_y_value) + (0.5 if np.ceil(max_y_value) - max_y_value < 0.25 else 0) if style_dict["y_limit_top"] is None else style_dict["y_limit_top"]
    elif plot_type == "loss":
        y_range = max_y_value - min_y_value
        margin = max(y_range * 0.5, 1e-6)
        y_lim_bottom = style_dict["y_limit_bottom"] if style_dict["y_limit_bottom"] is not None else min_y_value - margin
        y_lim_top = style_dict["y_limit_top"] if style_dict["y_limit_top"] is not None else max_y_value + margin
    
    # Synchronize with shared y-axis siblings
    y_siblings = ax.get_shared_y_axes().get_siblings(ax)
    if len(y_siblings) > 1:
        for sibling_ax in y_siblings:
            if sibling_ax != ax:
                sibling_ylim = sibling_ax.get_ylim()
                y_lim_bottom = min(y_lim_bottom, sibling_ylim[0])
                y_lim_top = max(y_lim_top, sibling_ylim[1])
    
    ax.set_ylim(bottom=y_lim_bottom, top=y_lim_top)

    _style_plot(ax, x_axis, style_dict=style_dict, plot_type=plot_type)


    # Handle legend based on legend_style preference
    legend_style = style_dict["legend_style"]
    
    if legend_style in ("right_hand", "both"):
        # Place legend on right side at final positions
        group_colors = {group: groupstyles[group]["color"] for group in groups}
        # Extract line width from the first plotted line
        line_width = 1.5  # default
        lines = ax.get_lines()
        if len(lines) > 0:
            line_width = lines[0].get_linewidth()
        line_style = style_dict["right_hand_legend_line_style"]
        right_hand_legend(ax,
                          final_positions,
                          group_colors,
                          groupstyles,
                          line_width=line_width,
                          line_style=line_style)
    
    if legend_style in ("outside", "both") and fig is not None:
        # Place legend at bottom (traditional combined legend)
        combined_lower_legend(fig,
                              ax,
                              marker_width=style_dict["legend_marker_width"],
                              marker_length=style_dict["legend_marker_length"],
                              n_cols=style_dict["outside_legend_n_cols"]
                              )
    

    if fig is not None:
        fig.tight_layout()
    return ax if fig is None else (fig, ax)
