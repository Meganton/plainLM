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
styles["RealAdamExtend_PB_like"] = {"color": colors[1], "linestyle": "-", "label": r"'\texttt{Adam} Addition'-Space"}
styles["AdamExtendMul_PB_like"] = {"color": colors[0], "linestyle": "-", "label": r"'\texttt{Adam} Multiplication'-Space"}
# styles["Real2PremadeModules_PB_like"] = {"color": colors[2], "linestyle": "-", "label": r"'\texttt{Premade-Modules}'-Space"}
styles["NLinesU_PB_like"] = {"color": colors[4], "linestyle": "-", "label": r"'\texttt{NLinesU}'-Space"}
# styles["Adam"] = {"color": colors[5], "linestyle": "--", "label": r"'\texttt{Adam}' (Niccolo)"}
# styles["AE_Adam_torch"] = {"color": colors[6], "linestyle": "--", "label": r"'\texttt{Adam}' (Torch)"}
# styles["AE_Adam_base"] = {"color": colors[7], "linestyle": "--", "label": r"\texttt{Adam} (tuned $\gamma$)"}
# styles["Muon"] = {"color": colors[6], "linestyle": "--", "label": r"'\texttt{Muon}' (tuned LR)"}
x_axis = "Evaluations"
algorithms = list(styles.keys())
only_best = ["Adam", "Muon", "AE_Adam_torch", "AE_Adam_base"]

seeds = 3


# Activate the plot style BEFORE creating the figure and axes
activate_plot_style()
fig, axes = plt.subplots(1, 1, figsize=(5, 2.5))

fig, ax = plot_results(
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
        "y_label": f"Validation Loss",
        "x_label": f"Evaluations",
        "x_limit_right": 30,
        "y_limit_top": 9,
        "y_limit_bottom": 4,
        "legend_style": "right_hand",
    },
    only_best=only_best
)

# plot_results(
#     fig=fig,
#     ax=axes[1],
#     path=path,
#     groups=algorithms,
#     groupstyles=styles,
#     seeds=seeds,
#     x_axis=x_axis,
#     plot_type="ranks",
#     # evaluation_cost=3,
#     max_fidelity=15,
#     style_dict={
#         "title": "",
#         "y_label": f"Relative Rank (1=Best)",
#         "x_label": f"Evaluations",
#         "x_limit_right": 2000,
#         # "y_limit_top": 3.5,
#         "legend_style": "outside",
#         "outside_legend_n_cols": 2,
#     },
#     only_best=["Adam", "Muon"]
# )

fig.suptitle("Search Space Ablation", y=1.02)
# Save the figure
fig.savefig(
    "neps_plots/plots/extension/8_space_ablation.png", bbox_inches="tight", pad_inches=0.1, dpi=300
)