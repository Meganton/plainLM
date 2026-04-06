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
path = Path("neps_runs/LI_space")


colors = mpl.colormaps['tab10'](np.linspace(0, 1, 10))
# Define styles for each algorithm
styles = {}
styles["NLinesU_LI3_r01"] = {"color": colors[0], "linestyle": "-", "label": "HB w/ Local Prior/Incumbent w/ 0.1 random"}
styles["NLinesU_LI3_r03"] = {"color": colors[1], "linestyle": "-", "label": "HB w/ Local Prior/Incumbent w/ 0.3 random"}
styles["NLinesU_LI3_r05"] = {"color": colors[2], "linestyle": "-", "label": "HB w/ Local Prior/Incumbent w/ 0.5 random"}
styles["NLinesU_PB_like"] = {"color": colors[3], "linestyle": "-", "label": "HB w/ Incumbent + Random"}
styles["Adam"] = {"color": colors[4], "linestyle": "--", "label": "Adam (tuned LR)"}
styles["Muon"] = {"color": colors[5], "linestyle": "--", "label": "Muon (tuned LR)"}
x_axis = "Evaluations"
algorithms = list(styles.keys())

seeds = 3


# Activate the plot style BEFORE creating the figure and axes
activate_plot_style()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

plot_results(
    fig=fig,
    ax=axes[0],
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
        # "x_limit_right": 2000 if col != 2 else 200,
        "y_limit_top": 9,
        "y_limit_bottom": 4,
        "legend_style": "none",
    },
    only_best=["Adam", "Muon"]
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
    # evaluation_cost=3,
    max_fidelity=15,
    style_dict={
        "title": "",
        "y_label": f"Relative Rank (1=Best)",
        "x_label": f"Evaluations",
        # "x_limit_right": 890,
        # "y_limit_top": 3.5,
        "legend_style": "outside",
        "outside_legend_n_cols": 2,
    },
    only_best=["Adam", "Muon"]
)

fig.suptitle("NEPS NOS - LI randomizing ablation - 8M Parameters", y=1.02)
# Save the figure
fig.savefig(
    "neps_plots/plots/li_r_ablation.png", bbox_inches="tight", pad_inches=0.1, dpi=300
)