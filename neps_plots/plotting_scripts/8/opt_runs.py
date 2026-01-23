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
path = Path("neps_runs/opt_runs")


colors = mpl.colormaps['tab10'](np.linspace(0, 1, 10))
# Define styles for each algorithm
styles = {}
styles["AE_add_LI2_r03"] = {"color": colors[0], "linestyle": "-", "label": "HB w/ Incumbent Mutation"}
# styles["AE_add_LI3_r03"] = {"color": colors[1], "linestyle": "-", "label": "HB w/ Local Prior/Incumbent"}
styles["Adam"] = {"color": colors[5], "linestyle": "--", "label": "Adam (Niccolo)"}
styles["AE_Adam_torch"] = {"color": colors[6], "linestyle": "--", "label": "Adam (Torch)"}
styles["AE_Adam_base"] = {"color": colors[7], "linestyle": "--", "label": "Adam (Base)"}
x_axis = "Evaluations"
algorithms = list(styles.keys())

seeds = 3


# Activate the plot style BEFORE creating the figure and axes
activate_plot_style()
fig, axes = plt.subplots(2, 1, figsize=(7, 5))

plot_results(
    fig=fig,
    ax=axes[0],
    path=path,
    groups=algorithms,
    groupstyles=styles,
    seeds=seeds,
    x_axis=x_axis,
    plot_type="loss",
    # evaluation_cost=12,
    max_fidelity=15,
    style_dict={
        "title": "",
        "y_label": f"Best score",
        "x_label": f"Evaluations",
        "x_limit_right": 2000,
        "y_limit_top": 6,
        "y_limit_bottom": 4,
        "y_scale": "log",
        "legend_style": "right_hand",
    },
    only_best=["Adam", "AE_Adam_torch", "AE_Adam_base"]
)
plot_results(
    fig=fig,
    ax=axes[1],
    path=path,
    groups=algorithms,
    groupstyles=styles,
    seeds=seeds,
    x_axis=x_axis,
    plot_type="ranks",
    # evaluation_cost=12,
    max_fidelity=15,
    style_dict={
        "title": "",
        "y_label": f"Best score",
        "x_label": f"Evaluations",
        "x_limit_right": 2000,
        # "y_limit_top": 9,
        # "y_limit_bottom": 5,
        "legend_style": "right_hand",
    },
    only_best=["Adam", "AE_Adam_torch", "AE_Adam_base"]
)

fig.suptitle("NEPS NOS - Optimal Runs - 8M Parameters", y=1.02)
# Save the figure
fig.savefig(
    "neps_plots/plots/opt_runs.png", bbox_inches="tight", pad_inches=0.1, dpi=300
)