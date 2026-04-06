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
path = Path("neps_runs/plot_run")


colors = mpl.colormaps['tab10'](np.linspace(0, 1, 10))
# Define styles for each algorithm
styles = {}
styles["RE"] =                     {"color": colors[0], "linestyle": "-", "label": "Regularized Evolution"}
styles["RS"] =                     {"color": colors[1], "linestyle": "-", "label": "Random Search"}
x_axis = "Evaluations"
algorithms = list(styles.keys())

seeds = 2


# Activate the plot style BEFORE creating the figure and axes
activate_plot_style()

fig, ax = plot_results(
    path=path,
    groups=algorithms,
    groupstyles=styles,
    seeds=seeds,
    x_axis=x_axis,
    plot_type="loss",
    evaluation_cost=17,
    # max_fidelity=17,
    style_dict={
        "title": "Plottest",
        "y_label": f"Best score",
        "x_label": f"Evaluations",
        # "x_limit_right": 2000 if col != 2 else 200,
        # "y_limit_top": 3.5,
        "legend_style": "outside",
    },
)

# Save the figure
fig.savefig(
    "neps_plots/plots/plot_test.png", bbox_inches="tight", pad_inches=0.1, dpi=300
)