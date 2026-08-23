import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_validate
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"}

def run_hyperparameter_tuning(model, param_grid, X_train, y_train, folds):
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=folds,
        scoring="accuracy",
        n_jobs=1
    )
    grid_search.fit(X_train, y_train)
    
    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.best_score_,
        grid_search.cv_results_
    )

def evaluate_model(model, X_train, y_train, folds):
    scores = cross_validate(model, X_train, y_train, cv=folds, scoring=SCORING, n_jobs=1)
    return pd.DataFrame({metric: scores[f"test_{metric}"] for metric in SCORING})

def print_evaluation_results(scores):
    summary = scores.agg(["mean", "std"]).T
    summary = summary.rename(columns={"mean": "CV mean", "std": "CV std"}).round(3)

    display(summary)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        summary,
        annot=True,
        fmt=".3f",
        cmap="Blues"
    )
    plt.title("Cross-validation results")
    plt.tight_layout()
    plt.show()