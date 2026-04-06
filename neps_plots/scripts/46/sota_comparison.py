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
path = Path("neps_runs/46_LI_space")


colors = mpl.colormaps['tab10'](np.linspace(0, 1, 10))
# Define styles for SOTA variants
styles = {}
styles["sota_li1_rand_01_ae"] = {"color": colors[0], "linestyle": "-", "label": "SOTA: Random 0.1 + AE"}
styles["sota_li1_rand_01_ae_mul"] = {"color": colors[1], "linestyle": "-", "label": "SOTA: Random 0.1 + AE Mul"}
styles["sota_li1_rand_01_nl"] = {"color": colors[2], "linestyle": "-", "label": "SOTA: Random 0.1 + NOS Lines"}
styles["sota_li1_rand_03_ae"] = {"color": colors[3], "linestyle": "-", "label": "SOTA: Random 0.3 + AE"}
styles["sota_li1_rand_01_r1_ae_add"] = {"color": colors[4], "linestyle": "-", "label": "SOTA: R0.1 + AE (r1)"}
styles["sota_li1_rand_01_r3_ae_add"] = {"color": colors[5], "linestyle": "-", "label": "SOTA: R0.1 + AE (r3)"}
styles["sota_li1_rand_01_r1_ae_add_wd"] = {"color": colors[6], "linestyle": "-", "label": "SOTA: R0.1 + AE WD (r1)"}
# styles["Anton_Adam_torch"] = {"color": colors[8], "linestyle": "--", "label": "Adam (Torch, tuned LR)"}
styles["Anton_Adam_AE"] = {"color": colors[9], "linestyle": "--", "label": "Adam (tuned LR)"}

x_axis = "Evaluations"
algorithms = list(styles.keys())
sota_runs = ["sota_li1_rand_01_ae",  "sota_li1_rand_03_ae", "sota_li1_rand_01_ae_mul", "sota_li1_rand_01_r1_ae_add",  "sota_li1_rand_01_r3_ae_add","sota_li1_rand_01_r1_ae_add_wd"]#, "sota_li1_rand_01_nl"]
baseline = ["Anton_Adam_torch", "Anton_Adam_AE"]

seeds = 3


# Activate the plot style BEFORE creating the figure and axes
activate_plot_style()
fig, axes = plt.subplots(1, 2, figsize=(13, 5))


# Loss comparison
plot_results(
    fig=fig,
    ax=axes[0],
    path=path,
    groups=sota_runs+baseline,
    groupstyles=styles,
    seeds=seeds,
    x_axis=x_axis,
    plot_type="loss",
    max_fidelity=50,
    style_dict={
        "title": "Best Loss",
        "y_label": "Best Score",
        "x_label": "Evaluations",
        # "x_limit_right": 260,
        "y_limit_top": 3.7,
        "y_limit_bottom": 3.5,
        "legend_style": "none",
    },
    only_best=baseline
)

# Rank comparison
plot_results(
    fig=fig,
    ax=axes[1],
    path=path,
    groups=sota_runs+baseline,
    groupstyles=styles,
    seeds=seeds,
    x_axis=x_axis,
    plot_type="ranks",
    max_fidelity=50,
    style_dict={
        "title": "Relative Rank",
        "y_label": "Relative Rank (1=Best)",
        "x_label": "Evaluations",
        # "x_limit_right": 260,
        "legend_style": "none",
    },
    only_best=baseline
)

combined_lower_legend(fig, axes, n_cols=3)

fig.suptitle("NEPS NOS - SOTA Comparison - 46M Parameters", y=1.02)
# Save the figure
fig.savefig(
    "neps_plots/plots/46/sota_comparison.png", bbox_inches="tight", pad_inches=0.1, dpi=300
)
