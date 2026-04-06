import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from neps_plots.plotting import plot_results, activate_plot_style, combined_lower_legend, right_hand_legend
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import pandas as pd

results_dir = "neps_runs/test_runs/fid_best_config/results"
algos = ["NLinesU_LI0_r03", "RealAdamExtend_PB_like", "Real2PremadeModules_PB_like"]
results={}
colors = mpl.colormaps['plasma'](np.linspace(0, 1, 10))
styles = {}
styles["NLinesU_LI0_r03_seed_0"] = {"color": colors[0], "linestyle": "--", "label": "NlinesU Seed 0"}
styles["NLinesU_LI0_r03_seed_1"] = {"color": colors[1], "linestyle": "--", "label": "NlinesU Seed 1"}
styles["NLinesU_LI0_r03_seed_2"] = {"color": colors[2], "linestyle": "--", "label": "NlinesU Seed 2"}

styles["RealAdamExtend_PB_like_seed_0"] = {"color": colors[3], "linestyle": "-", "label": "AdamExtend Seed 0"}
styles["RealAdamExtend_PB_like_seed_1"] = {"color": colors[4], "linestyle": "-", "label": "AdamExtend Seed 1"}
styles["RealAdamExtend_PB_like_seed_2"] = {"color": colors[5], "linestyle": "-", "label": "AdamExtend Seed 2"}
styles["Real2PremadeModules_PB_like_seed_0"] = {"color": colors[6], "linestyle": "-.", "label": "PremadeModules Seed 0"}
styles["Real2PremadeModules_PB_like_seed_1"] = {"color": colors[7], "linestyle": "-.", "label": "PremadeModules Seed 1"}
styles["Real2PremadeModules_PB_like_seed_2"] = {"color": colors[8], "linestyle": "-.", "label": "PremadeModules Seed 2"}
activate_plot_style()


small_model_data = pd.DataFrame()
large_model_data = pd.DataFrame()

for algo in algos:
    results[algo] = {}
    for seed in range(3):
        summary_file = os.path.join(results_dir, f"{algo}_{seed}.json")
        if os.path.exists(summary_file):
            import json
            with open(summary_file, "r") as f:
                data = json.load(f)
                results[algo][seed] = data
        else:
            print(f"Warning: Summary file for {algo} seed {seed} not found.")
        # print(results[algo][seed])

        small_model_losses = results[algo][seed]["small_loss_history"]
        large_model_losses = results[algo][seed]["large_loss_history"]
        small_model_steps = results[algo][seed]["small_fidelities"]
        large_model_steps = results[algo][seed]["large_fidelities"]


        df_small = pd.DataFrame({"step": small_model_steps, f"{algo}_seed_{seed}": small_model_losses}).set_index("step").sort_index()
        small_model_data = small_model_data.join(df_small, how="outer")
        df_large = pd.DataFrame({"step": large_model_steps, f"{algo}_seed_{seed}": large_model_losses}).set_index("step").sort_index()
        large_model_data = large_model_data.join(df_large, how="outer")

small_model_data = small_model_data.loc[small_model_data.index<=13]
small_model_ranked_data = small_model_data.rank(axis=1, method='average')
large_model_ranked_data = large_model_data.rank(axis=1, method='average')



fig, axes = plt.subplots(2,2, figsize=(14, 12), sharey='row')
final_positions = {"small_loss": {}, "large_loss": {}, "small_rank": {}, "large_rank": {}}
for algo in small_model_data.columns:
    axes[0,0].plot(small_model_ranked_data.index, small_model_ranked_data[algo], color=styles[algo]["color"], linestyle=styles[algo]["linestyle"], alpha=1, label=styles[algo]["label"])
    final_positions["small_rank"][styles[algo]["label"]] = small_model_ranked_data[algo].values[-1]
    axes[0,1].plot(large_model_ranked_data.index, large_model_ranked_data[algo], color=styles[algo]["color"], linestyle=styles[algo]["linestyle"], alpha=1, label=styles[algo]["label"])
    final_positions["large_rank"][styles[algo]["label"]] = large_model_ranked_data[algo].values[-1]
    axes[1,0].plot(small_model_data.index, small_model_data[algo], color=styles[algo]["color"], linestyle=styles[algo]["linestyle"], alpha=1, label=styles[algo]["label"])
    final_positions["small_loss"][styles[algo]["label"]] = small_model_data[algo].values[-1]
    axes[1,1].plot(large_model_data.index, large_model_data[algo], color=styles[algo]["color"], linestyle=styles[algo]["linestyle"], alpha=1, label=styles[algo]["label"])
    final_positions["large_loss"][styles[algo]["label"]] = large_model_data[algo].values[-1]
axes[1,0].set_ylim(3.5,9)
fig.subplots_adjust(wspace=0.7, hspace=0.3)
axes[0,0].set_title("Fidelity Comparison Across Seeds (Small Model)")
axes[0,0].set_xlabel("Training Steps")
axes[0,0].set_ylabel("Rank (Lower is Better)")
axes[0,1].set_title("Fidelity Comparison Across Seeds (Large Model)")
axes[0,1].set_xlabel("Training Steps")
axes[1,0].set_title("Loss Comparison Across Seeds (Small Model)")
axes[1,0].set_xlabel("Training Steps")
axes[1,0].set_ylabel("Loss")
axes[1,1].set_title("Loss Comparison Across Seeds (Large Model)")
axes[1,1].set_xlabel("Training Steps")


right_hand_legend(axes[0,0], final_positions=final_positions["small_rank"], group_colors={styles[algo]["label"]: styles[algo]["color"] for algo in small_model_data.columns})
right_hand_legend(axes[0,1], final_positions=final_positions["large_rank"], group_colors={styles[algo]["label"]: styles[algo]["color"] for algo in large_model_data.columns})
right_hand_legend(axes[1,0], final_positions=final_positions["small_loss"], group_colors={styles[algo]["label"]: styles[algo]["color"] for algo in small_model_data.columns})
right_hand_legend(axes[1,1], final_positions=final_positions["large_loss"], group_colors={styles[algo]["label"]: styles[algo]["color"] for algo in large_model_data.columns})
fig.savefig("neps_plots/plots/fidelity_ranks.png", bbox_inches="tight", pad_inches=0.1, dpi=300)

