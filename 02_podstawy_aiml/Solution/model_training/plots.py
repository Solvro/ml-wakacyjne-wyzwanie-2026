import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def plot_hyperparameter_tuning_results(cv_results):
    results = pd.DataFrame(cv_results)

    max_depths = sorted(results["param_max_depth"].unique())
    n_estimators_list = sorted(results["param_n_estimators"].unique())

    fig, axes = plt.subplots(
        len(n_estimators_list), 
        len(max_depths), 
        figsize=(24, 6 * len(n_estimators_list)),
        sharex=True, 
        sharey=True
    )

    if len(n_estimators_list) == 1 and len(max_depths) == 1:
        axes = np.array([[axes]])
    elif len(n_estimators_list) == 1:
        axes = axes[np.newaxis, :]
    elif len(max_depths) == 1:
        axes = axes[:, np.newaxis]

    for i, n_est in enumerate(n_estimators_list):
        for j, depth in enumerate(max_depths):
            ax = axes[i, j]
            
            data = results[
                (results["param_max_depth"] == depth) & 
                (results["param_n_estimators"] == n_est)
            ]
            
            if data.empty:
                ax.set_visible(False)
                continue

            heatmap_data = data.pivot(
                index="param_min_child_weight",
                columns="param_learning_rate",
                values="mean_test_score",
            )

            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=".3f",
                ax=ax,
                cmap="Reds",    
                vmin=0.79,       
                vmax=0.845, 
                cbar=(j == len(max_depths) - 1),
            )

            ax.set_title(f"n_est = {n_est} | max_depth = {depth}", fontsize=12, pad=10)
            
            if i == len(n_estimators_list) - 1:
                ax.set_xlabel("Learning rate")
            else:
                ax.set_xlabel("")
                
            if j == 0:
                ax.set_ylabel("min_child_weight")
            else:
                ax.set_ylabel("")

    plt.suptitle("XGBoost: Hyperparameter Tuning", y=1.02, fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()