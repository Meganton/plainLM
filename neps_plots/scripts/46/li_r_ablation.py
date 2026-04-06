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
styles["li1_ratio_01"] = {"color": colors[0], "linestyle": "-", "label": "HB w/ Incumbent-Mutation w/ 0.1 ratio"}
styles["li1_ratio_03"] = {"color": colors[1], "linestyle": "-", "label": "HB w/ Incumbent-Mutation w/ 0.3 ratio"}
styles["li1_ratio_05"] = {"color": colors[2], "linestyle": "-", "label": "HB w/ Incumbent-Mutation w/ 0.5 ratio"}
styles["li1_ratio_08"] = {"color": colors[3], "linestyle": "-", "label": "HB w/ Incumbent-Mutation w/ 0.8 ratio"}
styles["li1_rand_01"] = {"color": colors[4], "linestyle": "-", "label": "HB w/ Incumbent-Mutation w/ 0.1 random ratio"}
styles["li1_rand_03"] = {"color": colors[5], "linestyle": "-", "label": "HB w/ Incumbent-Mutation w/ 0.3 random ratio"}
styles["li1_rand_05"] = {"color": colors[6], "linestyle": "-", "label": "HB w/ Incumbent-Mutation w/ 0.5 random ratio"}
styles["li1_rand_08"] = {"color": colors[7], "linestyle": "-", "label": "HB w/ Incumbent-Mutation w/ 0.8 random ratio"}
styles["li1_fixed_1"] = {"color": colors[8], "linestyle": "-", "label": "HB w/ Incumbent-Mutation w/ fixed 1"}
styles["li1_fixed_3"] = {"color": colors[9], "linestyle": "-", "label": "HB w/ Incumbent-Mutation w/ fixed 3"}
styles["li1_fixed_5"] = {"color": colors[10], "linestyle": "-", "label": "HB w/ Incumbent-Mutation w/ fixed 5"}
styles["li1_fixed_8"] = {"color": colors[11], "linestyle": "-", "label": "HB w/ Incumbent-Mutation w/ fixed 8"}
styles["Anton_Adam_torch"] = {"color": colors[12], "linestyle": "--", "label": "Adam (Torch, tuned LR)"}
styles["Anton_Adam_AE"] = {"color": colors[13], "linestyle": "--", "label": "Adam (Space, tuned LR)"}

x_axis = "Evaluations"
algorithms = list(styles.keys())
ratios = ["li1_ratio_01", "li1_ratio_03", "li1_ratio_05", "li1_ratio_08"]
randoms = ["li1_rand_01", "li1_rand_03", "li1_rand_05", "li1_rand_08"]
fixeds = ["li1_fixed_1", "li1_fixed_3", "li1_fixed_5", "li1_fixed_8"]
only_best = ["Anton_Adam_torch", "Anton_Adam_AE"]

seeds = 4


# Activate the plot style BEFORE creating the figure and axes
activate_plot_style()
fig, axes = plt.subplots(3, 2, figsize=(12, 12))


for n,group in enumerate([ratios, randoms, fixeds]):

    plot_results(
        fig=fig,
        ax=axes[n, 0],
        path=path,
        groups=group+only_best,
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
            "x_limit_right": 80,
            "y_limit_top": 9,
            "y_limit_bottom": 3.5,
            "legend_style": "none",
        },
        only_best=only_best
    )

    plot_results(
        fig=fig,
        ax=axes[n, 1],
        path=path,
        groups=group+only_best,
        groupstyles=styles,
        seeds=seeds,
        x_axis=x_axis,
        plot_type="ranks",
        # evaluation_cost=3,
        max_fidelity=50,
        style_dict={
            "title": "",
            "y_label": f"Relative Rank (1=Best)",
            "x_label": f"Evaluations",
            "x_limit_right": 80,
            # "y_limit_top": 3.5,
            "legend_style": "none",
            "outside_legend_n_cols": 2,
        },
        only_best=only_best
    )

combined_lower_legend(fig, axes, n_cols=3)

fig.suptitle("NEPS NOS - LI randomizing ablation - 46M Parameters", y=1.02)
# Save the figure
fig.savefig(
    "neps_plots/plots/46/li_r_ablation.png", bbox_inches="tight", pad_inches=0.1, dpi=300
)