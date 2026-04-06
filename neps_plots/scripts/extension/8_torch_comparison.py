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
path = Path("neps_runs/sota_8")


colors = mpl.colormaps['tab20'](np.linspace(0,1,20))
# Define styles for SOTA variants
styles = {}

styles["8_single_conf_adam"] = {"color": colors[0], "linestyle": "-", "label": r"\texttt{Adam (tuned $\gamma$)}"}
styles["AE_Adam_base"] = {"color": colors[1], "linestyle": "-", "label": r"\texttt{Adam base}"}
styles["AE_Adam_torch"] = {"color": colors[2], "linestyle": "-", "label": r"\texttt{Adam torch (tuned $\gamma$)}"}
styles["RealAdamExtend_PB_like"] = {"color": colors[8], "linestyle": "-", "label": r"Ours"}
styles["8_single_conf_torch_adagrad"] = {"color": colors[3], "linestyle": "-", "label": r"\texttt{Adagrad (tuned $\gamma$)}"}
styles["8_single_conf_torch_sgd"] = {"color": colors[4], "linestyle": "-", "label": r"\texttt{SGD (tuned $\gamma$)}"}
styles["8_single_conf_torch_nadam"] = {"color": colors[5], "linestyle": "-", "label": r"\texttt{NAdam (tuned $\gamma$)}"}
styles["8_single_conf_torch_rmsprop"] = {"color": colors[6], "linestyle": "-", "label": r"\texttt{RMSprop (tuned $\gamma$)}"}
styles["8_single_conf_torch_sgd_momentum"] = {"color": colors[7], "linestyle": "-", "label": r"\texttt{SGD w/ Momentum} (tuned $\gamma$)"}

only_best = ["AE_Adam_torch", "AE_Adam_base" ,"8_single_conf_adam", "8_single_conf_torch_adagrad", "8_single_conf_torch_sgd", "8_single_conf_torch_nadam", "8_single_conf_torch_rmsprop", "8_single_conf_torch_sgd_momentum"]

x_axis = "Evaluations"
algorithms = list(styles.keys())

seeds = 3


# Activate the plot style BEFORE creating the figure and axes
activate_plot_style()
fig, axes = plt.subplots(1, 1, figsize=(5, 2.5))


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
    max_fidelity=12,
    style_dict={
        "title": "",
        "y_label": "Best Score",
        "x_label": "Evaluations",
        # "x_limit_right": 400,
        "y_limit_top": 6,
        "y_limit_bottom": 4,
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

fig.suptitle("Torch Optimizers on 8M Parameter Model", y=1.02)
# Save the figure
fig.savefig(
    "neps_plots/plots/extension/8_torch_comparison.png", bbox_inches="tight", pad_inches=0.1, dpi=300
)
