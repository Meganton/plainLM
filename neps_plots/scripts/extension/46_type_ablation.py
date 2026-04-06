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


colors = mpl.colormaps['tab20'](np.linspace(0, 1, 20))
# Define styles for each algorithm
styles = {}
# styles["li1_ratio_01"] = {"color": colors[0], "linestyle": "-", "label": r"Mutate 10\% of samplings"}
# styles["li1_ratio_03"] = {"color": colors[1], "linestyle": "-", "label": r"Mutate 30\% of samplings"}
styles["li1_ratio_05"] = {"color": colors[2], "linestyle": "-", "label": r"Mutate 50\% of samplings"}
# styles["li1_ratio_08"] = {"color": colors[3], "linestyle": "-", "label": r"Mutate 80\% of samplings"}
styles["li1_rand_01"] = {"color": colors[4], "linestyle": "-", "label": r"Mutate up to 10\% of samplings"}
# styles["li1_rand_03"] = {"color": colors[5], "linestyle": "-", "label": r"Mutate up to 30\% of samplings"}
# styles["li1_rand_05"] = {"color": colors[6], "linestyle": "-", "label": r"Mutate up to 50\% of samplings"}
# styles["li1_rand_08"] = {"color": colors[7], "linestyle": "-", "label": r"Mutate up to 80\% of samplings"}
styles["li1_fixed_1"] = {"color": colors[8], "linestyle": "-", "label": r"Mutate 1 sampling"}
# styles["li1_fixed_3"] = {"color": colors[9], "linestyle": "-", "label": r"Mutate 3 samplings"}
# styles["li1_fixed_5"] = {"color": colors[10], "linestyle": "-", "label": r"Mutate 5 samplings"}
# styles["li1_fixed_8"] = {"color": colors[11], "linestyle": "-", "label": r"Mutate 8 samplings"}
# styles["Anton_Adam_torch"] = {"color": colors[12], "linestyle": "--", "label": "Adam (Torch, tuned LR)"}
# styles["Anton_Adam_AE"] = {"color": colors[18], "linestyle": "--", "label": "Adam (tuned LR)"}

x_axis = "Evaluations"
algorithms = list(styles.keys())
only_best = []#["Anton_Adam_AE"]

seeds = 5


# Activate the plot style BEFORE creating the figure and axes
activate_plot_style()
fig, ax = plt.subplots(1, 1, figsize=(5, 2))

plot_results(
    fig=fig,
    ax=ax,
    path=path,
    groups=algorithms,
    groupstyles=styles,
    seeds=seeds,
    x_axis=x_axis,
    plot_type="loss",
    # evaluation_cost=3,
    max_fidelity=50,
    style_dict={
        "title": "",
        "y_label": f"Best score",
        "x_label": f"Evaluations",
        "x_limit_right": 30,
        "y_limit_top": 10,
        "y_limit_bottom": 6,
        # "y_scale": "log",
        "legend_style": "right_hand",
    },
    only_best=only_best
)




fig.suptitle("Mutation ablation", y=1.02)
# Save the figure
fig.savefig(
    f"neps_plots/plots/extension/type_ablation.png", bbox_inches="tight", pad_inches=0.1, dpi=300
)