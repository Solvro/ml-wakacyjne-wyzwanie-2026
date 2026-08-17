from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score

def run_hyperparameter_tuning(model, param_grid, X_train, y_train, folds):
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=folds,
        scoring="accuracy"
    )
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def print_evaluation_results(scores):
    print(f"Mean accuracy: {sum(scores) / len(scores):.4f}")
    print(f"Standard deviation: {pd.Series(scores).std():.4f}")

def test_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)