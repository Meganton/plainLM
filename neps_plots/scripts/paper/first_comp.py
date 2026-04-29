import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from neps_plots.plotting import plot_results, activate_plot_style, combined_lower_legend
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Important: Keep the activate_plot_style, title/label settings and plot_results functions in
# exactly this order to ensure consistent results


# The path to the experiment results directory
path = Path("neps_runs/paper_8")


colors = mpl.colormaps['tab10'](np.linspace(0, 1, 10))
# Define styles for each algorithm
styles = {}
styles["nlines_li_8m"] = {"color": colors[0], "linestyle": "-", "label": r"Ours (Local Inc)"}
styles["nlines_pb_8m"] = {"color": colors[1], "linestyle": "-", "label": r"Ours (PriorBand)"}
styles["nlines_re_8m"] = {"color": colors[2], "linestyle": "-", "label": r"Regularized Evolution"}
styles["nlines_rs_8m"] = {"color": colors[3], "linestyle": "-", "label": r"Random Search"}
styles["nlines_hb_8m"] = {"color": colors[4], "linestyle": "-", "label": r"HyperBand"}

x_axis = "Evaluations"
algorithms = list(styles.keys())
only_best = []#["Adam", "Muon", "AE_Adam_torch", "AE_Adam_base", "8_single_conf_torch_sgd", "8_single_conf_torch_adagrad", "8_single_conf_torch_sgd", "8_single_conf_torch_nadam", "8_single_conf_torch_rmsprop", "8_single_conf_torch_sgd_momentum"]

seeds = 5


# Activate the plot style BEFORE creating the figure and axes
activate_plot_style()
fig, axes = plt.subplots(1, 1, figsize=(5, 2.5))

plot_results(
    fig=fig,
    ax=axes,
    path=path,
    groups=algorithms,
    groupstyles=styles,
    seeds=seeds,
    x_axis=x_axis,
    plot_type="loss",
    # evaluation_cost=3,
    max_fidelity=15,
    style_dict={
        "title": "",
        "y_label": f"Best score",
        "x_label": f"Evaluations",
        "x_limit_right": 800,
        "y_limit_top": 10,
        "y_limit_bottom": 6,
        "legend_style": "right_hand",
        # "outside_legend_n_cols": 2,
    },
    only_best=only_best
)

fig.suptitle(r"NePS Algorithms on 8M Parameter Model", y=1.02)
# Save the figure
fig.savefig(
    "neps_plots/plots/paper/comp_loss.png", bbox_inches="tight", pad_inches=0.1, dpi=300
)

activate_plot_style()
fig, axes = plt.subplots(1, 1, figsize=(5, 2.5))

plot_results(
    fig=fig,
    ax=axes,
    path=path,
    groups=algorithms,
    groupstyles=styles,
    seeds=seeds,
    x_axis=x_axis,
    plot_type="ranks",
    # evaluation_cost=3,
    max_fidelity=15,
    style_dict={
        "title": "",
        "y_label": f"Relative Rank (1=Best)",
        "x_label": f"Evaluations",
        "x_limit_right": 800,
        # "y_limit_top": 3.5,
        "legend_style": "right_hand",
        # "outside_legend_n_cols": 2,
    },
    only_best=only_best
)

fig.suptitle(r"NePS Algorithms on 8M Parameter Model", y=1.02)
# Save the figure
fig.savefig(
    "neps_plots/plots/paper/comp_ranks.png", bbox_inches="tight", pad_inches=0.1, dpi=300
)