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


colors = mpl.colormaps['tab10'](np.linspace(0,1,10))
# Define styles for SOTA variants
styles = {}
# styles["sota_li1_rand_01_ae"] = {"color": colors[0], "linestyle": "-", "label": r"Up to 10 \% ratio w/ no random injections"}
styles["sota_li1_rand_01_ae_mul"] = {"color": colors[0], "linestyle": "-", "label": r"Ours (\texttt{Adam} Multiplication)"}
styles["sota_li1_rand_01_r1_am_add"] = {"color": colors[3], "linestyle": "-", "label": r"\texttt{Adam} More (All)"}
styles["sota_li1_rand_01_r1_am1_add"] = {"color": colors[4], "linestyle": "-", "label": r"\texttt{Adam} More (t)"}
styles["sota_li1_rand_01_r1_am2_add"] = {"color": colors[5], "linestyle": "-", "label": r"\texttt{Adam} More (depth)"}
styles["sota_li1_rand_01_r1_am3_add"] = {"color": colors[6], "linestyle": "-", "label": r"\texttt{Adam} More (attention)"}

# styles["sota_li1_rand_01_r1_ae_add"] = {"color": colors[4], "linestyle": "-", "label": r"Up to 10 \% ratio w/ 10\% random injections"}
styles["sota_li1_rand_01_r3_ae_add"] = {"color": colors[1], "linestyle": "-", "label": r"Ours (\texttt{Adam} Addition)"}
# styles["sota_li1_rand_01_r1_ae_add_wd"] = {"color": colors[6], "linestyle": "-", "label": "SOTA: R0.1 + AE WD (r1)"}
styles["46_single_conf_torch_sgd"] = {"color": colors[2], "linestyle": "--", "label": r"\texttt{SGD} (tuned $\gamma$)"}
styles["Anton_Adam_AE"] = {"color": colors[3], "linestyle": "--", "label": r"\texttt{Adam} (tuned $\gamma$)"}
# styles["46_single_conf_adam_wd"] = {"color": colors[10], "linestyle": "--", "label": r"\texttt{Adam} (tuned $\gamma$ \& $\lambda$)"}

# styles["Anton_Adam_torch"] = {"color": colors[8], "linestyle": "--", "label": r"\texttt{Adam} (Torch, tuned \gamma)"}

only_best = ["Anton_Adam_AE", "46_single_conf_torch_sgd"]#, "46_single_conf_adam_wd"]

x_axis = "Evaluations"
algorithms = list(styles.keys())

seeds = 5


# Activate the plot style BEFORE creating the figure and axes
activate_plot_style()
fig, axes = plt.subplots(1, 1, figsize=(6, 2.5))


# Loss comparison
plot_results(
    fig=fig,
    ax=axes,
    path=path,
    groups=styles.keys(),
    groupstyles=styles,
    seeds=seeds,
    x_axis=x_axis,
    plot_type="loss",
    max_fidelity=50,
    style_dict={
        "title": "",
        "y_label": "Best Score",
        "x_label": "Evaluations",
        "x_limit_right": 400,
        "y_limit_top": 3.62,
        "y_limit_bottom": 3.59,
        "legend_style": "right_hand",
    },
    only_best=only_best
)

# Rank comparison
# plot_results(
#     fig=fig,
#     ax=axes[1],
#     path=path,
#     groups=styles.keys(),
#     groupstyles=styles,
#     seeds=seeds,
#     x_axis=x_axis,
#     plot_type="ranks",
#     max_fidelity=50,
#     style_dict={
#         "title": "Relative Rank",
#         "y_label": "Relative Rank (1=Best)",
#         "x_label": "Evaluations",
#         "x_limit_right": 400,
#         "legend_style": "right_hand",
#     },
#     only_best=only_best
# )

# combined_lower_legend(fig, axes, n_cols=3)

fig.suptitle("'Adam Extension'-Optimizers on 46M Parameter Model", y=1.02)
# Save the figure
fig.savefig(
    "neps_plots/plots/extension/46_adam_more.png", bbox_inches="tight", pad_inches=0.1, dpi=300
)

activate_plot_style()
fig, axes = plt.subplots(1, 1, figsize=(6, 2.5))

plot_results(
    fig=fig,
    ax=axes,
    path=path,
    groups=algorithms,
    groupstyles=styles,
    seeds=seeds,
    x_axis=x_axis,
    plot_type="ranks",
    max_fidelity=50,
    style_dict={
        "title": "",
        "y_label": "Relative Rank (1=Best)",
        "x_label": "Evaluations",
        "x_limit_right": 400,
        "legend_style": "right_hand",
    },
    only_best=only_best
)

fig.suptitle("'Adam Extension'-Optimizers on 46M Parameter Model", y=1.02)
# Save the figure
fig.savefig(
    "neps_plots/plots/extension/46_adam_more_ranks.png", bbox_inches="tight", pad_inches=0.1, dpi=300
)
