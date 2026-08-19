from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from IPython.display import display

def run_hyperparameter_tuning(model, param_grid, X_train, y_train, folds):
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=folds,
        scoring="accuracy",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.best_score_,
        grid_search.cv_results_
    )

def print_evaluation_results(scores):
    print(f"Mean accuracy: {sum(scores) / len(scores):.4f}")
    print(f"Standard deviation: {pd.Series(scores).std():.4f}")

def test_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)

def compare_candidates(candidates, X_train, y_train, X_test, y_test):
    results = []

    for item in candidates:
        params = item.copy()
        name = params.pop("Candidate")

        model = XGBClassifier(
            **params,
            random_state=67,
            objective="binary:logistic",
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        results.append(
            {
                "Candidate": name,
                **params,
                "Train Score": round(train_acc, 4),
                "Test Score": round(test_acc, 4),
                "Gap (Train - Test)": round(train_acc - test_acc, 4),
            }
        )

    df_comparison = pd.DataFrame(results)
    display(df_comparison.sort_values(by="Test Score", ascending=False))