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


colors = mpl.colormaps['tab10'](np.linspace(0, 1, 10))
# Define styles for each algorithm
styles = {}
styles["RealAdamExtend_PB_like"] = {"color": colors[1], "linestyle": "-", "label": r"Ours"}
# styles["AdamExtendMul_PB_like"] = {"color": colors[0], "linestyle": "-", "label": r"Ours (\texttt{Adam} Multiplication)"}
# styles["Real2PremadeModules_PB_like"] = {"color": colors[2], "linestyle": "-", "label": r"'Premade-Modules'-Space"}
# styles["NLinesU_PB_like"] = {"color": colors[4], "linestyle": "-", "label": r"'10 Lines + Update'-Space"}
# styles["Adam"] = {"color": colors[5], "linestyle": "--", "label": r"Adam (Niccolo)"}
# styles["AE_Adam_torch"] = {"color": colors[6], "linestyle": "--", "label": r"Adam (Torch)"}
styles["AE_Adam_base"] = {"color": colors[3], "linestyle": "--", "label": r"\texttt{Adam} (tuned $\gamma$)"}
# styles["8_single_conf_torch_sgd"] = {"color": colors[2], "linestyle": "--", "label": r"\texttt{SGD} (tuned $\gamma$)"}
# styles["sota_8_li1_rand_01_ae_mul"] = {"color": colors[0], "linestyle": "-", "label": r"Ours (\texttt{Adam} Multiplication)"}
# styles["Muon"] = {"color": colors[6], "linestyle": "--", "label": r"Muon (tuned $\gamma$)"}
# styles["8_single_conf_torch_adagrad"] = {"color": colors[3], "linestyle": "--", "label": r"\texttt{Adagrad (tuned $\gamma$)}"}
# styles["8_single_conf_torch_sgd"] = {"color": colors[4], "linestyle": "--", "label": r"\texttt{SGD (tuned $\gamma$)}"}
# styles["8_single_conf_torch_nadam"] = {"color": colors[5], "linestyle": "--", "label": r"\texttt{NAdam (tuned $\gamma$)}"}
# styles["8_single_conf_torch_rmsprop"] = {"color": colors[6], "linestyle": "--", "label": r"\texttt{RMSprop (tuned $\gamma$)}"}
# styles["8_single_conf_torch_sgd_momentum"] = {"color": colors[7], "linestyle": "--", "label": r"\texttt{SGD w/ Momentum} (tuned $\gamma$)"}
x_axis = "Evaluations"
algorithms = list(styles.keys())
only_best = ["Adam", "Muon", "AE_Adam_torch", "AE_Adam_base", "8_single_conf_torch_sgd", "8_single_conf_torch_adagrad", "8_single_conf_torch_sgd", "8_single_conf_torch_nadam", "8_single_conf_torch_rmsprop", "8_single_conf_torch_sgd_momentum"]

seeds = 3


# Activate the plot style BEFORE creating the figure and axes
activate_plot_style()


for i in [0,1]:
    fig, axes = plt.subplots(1, 1, figsize=(5, 2.5))
    if i==1:
        # styles["8_single_conf_torch_rmsprop"] = {"color": colors[6], "linestyle": "--", "label": r"\texttt{RMSprop (tuned $\gamma$)}"}
        styles["8_single_conf_torch_nadam"] = {"color": colors[5], "linestyle": "--", "label": r"\texttt{NAdam (tuned $\gamma$)}"}

        algorithms = list(styles.keys())

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
            "y_label": f"Validation Loss",
            "x_label": f"Evaluations",
            "x_limit_right": 30,
            "y_limit_top": 5,
            "y_limit_bottom": 4.4,
            "legend_style": "right_hand",
            "outside_legend_n_cols": 2,
        },
        only_best=only_best
    )

   
    # fig.suptitle(r"\texttt{Adam} Extension-Optimizers on 8M Parameter Model", y=1.02)
    # Save the figure
    fig.savefig(
        f"neps_plots/plots/extension/8_sota_comparison_{i}.png", bbox_inches="tight", pad_inches=0.1, dpi=300
    )

fig, axes = plt.subplots(1, 1, figsize=(5, 2.5))


styles["RealAdamExtend_PB_like"] = {"color": colors[1], "linestyle": "-", "label": r"Ours"}
# styles["AdamExtendMul_PB_like"] = {"color": colors[0], "linestyle": "-", "label": r"Ours (\texttt{Adam} Multiplication)"}
# styles["Real2PremadeModules_PB_like"] = {"color": colors[2], "linestyle": "-", "label": r"'Premade-Modules'-Space"}
# styles["NLinesU_PB_like"] = {"color": colors[4], "linestyle": "-", "label": r"'10 Lines + Update'-Space"}
# styles["Adam"] = {"color": colors[5], "linestyle": "--", "label": r"Adam (Niccolo)"}
# styles["AE_Adam_torch"] = {"color": colors[6], "linestyle": "--", "label": r"Adam (Torch)"}
styles["AE_Adam_base"] = {"color": colors[3], "linestyle": "--", "label": r"\texttt{Adam} (tuned $\gamma$)"}
styles["8_single_conf_torch_sgd"] = {"color": colors[2], "linestyle": "--", "label": r"\texttt{SGD} (tuned $\gamma$)"}
# styles["sota_8_li1_rand_01_ae_mul"] = {"color": colors[0], "linestyle": "-", "label": r"Ours (\texttt{Adam} Multiplication)"}
# styles["Muon"] = {"color": colors[6], "linestyle": "--", "label": r"Muon (tuned $\gamma$)"}
styles["8_single_conf_torch_adagrad"] = {"color": colors[3], "linestyle": "--", "label": r"\texttt{Adagrad (tuned $\gamma$)}"}
styles["8_single_conf_torch_sgd"] = {"color": colors[4], "linestyle": "--", "label": r"\texttt{SGD (tuned $\gamma$)}"}
styles["8_single_conf_torch_nadam"] = {"color": colors[5], "linestyle": "--", "label": r"\texttt{NAdam (tuned $\gamma$)}"}
styles["8_single_conf_torch_rmsprop"] = {"color": colors[6], "linestyle": "--", "label": r"\texttt{RMSprop (tuned $\gamma$)}"}
styles["8_single_conf_torch_sgd_momentum"] = {"color": colors[7], "linestyle": "--", "label": r"\texttt{SGD w/ Momentum} (tuned $\gamma$)"}
algorithms = list(styles.keys())


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
        "y_label": f"Relative Rank",
        "x_label": f"Evaluations",
        "x_limit_right": 30,
        # "y_limit_top": 3.5,
        "legend_style": "right_hand",
        # "outside_legend_n_cols": 2,
    },
    only_best=only_best
)

# fig.suptitle(r"\texttt{Adam} Extension-Optimizers on 8M Parameter Model", y=1.02)
# Save the figure
fig.savefig(
    "neps_plots/plots/extension/8_sota_comparison_ranks_all.png", bbox_inches="tight", pad_inches=0.1, dpi=300
)