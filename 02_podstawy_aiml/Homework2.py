import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingRandomSearchCV, train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import ParameterGrid
import warnings

warnings.filterwarnings('ignore', category=ConvergenceWarning)

def main():
    tytanik = pd.read_csv('02_podstawy_aiml/titanic_ready.csv')

    X = tytanik[['Pclass', 'Male', 'Age', 'SibSp', 'Parch', 'Fare', 'Has_Cabin', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
    # posiadane dane
    y = tytanik['Survived']
    # obliczany wynik

    zakres_parametrow = list(range(1, 31))
    zakres_cech = list(range(1, 11))
    # zakres cech

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # trenowanie modelu

    baseline_model = DummyClassifier(strategy='most_frequent')
    # moim baselinem będzię DummyClassifier

    baseline_cv_scores = cross_val_score(baseline_model, X_train, y_train, cv=5)
    print(f"--- Baseline (DummyClassifier): {baseline_cv_scores.mean():.3f} ---\n", flush=True)

    #'flush=True' jest dlaczego aby konsola nie wariowała, bo czasami duplikuje tekst

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("--- Optymalizacja wielu modeli (HalvingRandomSearchCV) ---", flush=True)
    # tworze dużą liste algroytów oraz parametrów jakie one bedą urzywać
    modele_do_sprawdzenia = {
        'Drzewo Decyzyjne': {
            'model': Pipeline([
                ('skb', SelectKBest(score_func=f_classif)), 
                ('dt', DecisionTreeClassifier(random_state=42))
            ]),
            'parametry': {
                'skb__k': zakres_cech,
                'dt__max_depth': zakres_parametrow,
                'dt__criterion': ['gini', 'entropy', 'log_loss'],
                'dt__min_samples_leaf': zakres_parametrow,
                'dt__min_samples_split': zakres_parametrow[1:], 
                'dt__max_features': ['sqrt', 'log2', None],
                'dt__splitter': ['best', 'random']
            }
        },
        'KNN': {
        # piplinie skaluje dane, bo niektóre dane mają różny zakres
            'model': Pipeline([
                ('skaler', StandardScaler()),
                ('skb', SelectKBest(score_func=f_classif)),
                ('knn', KNeighborsClassifier())
            ]),
            'parametry': {
                'skb__k': zakres_cech,
                'knn__n_neighbors': zakres_parametrow[::2],
                    # tylko nie parzyste liczby
         
                'knn__weights': ['uniform', 'distance'],
                'knn__metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
                'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'knn__p': [1, 2, 3]
            }
        },
        'SVM': {
            'model': Pipeline([
                ('skaler', StandardScaler()),
                ('skb', SelectKBest(score_func=f_classif)),
                ('svm', SVC(random_state=42))
            ]),
            'parametry': {
                'skb__k': zakres_cech,
                'svm__C': loguniform(1e-3, 1e2),
                'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'svm__degree': zakres_parametrow[:6] 
            }
        },
        'Sieć Neuronowa (MLP)': {
            'model': Pipeline([
                ('skaler', StandardScaler()),
                ('skb', SelectKBest(score_func=f_classif)),
                ('mlp', MLPClassifier(max_iter=500, random_state=42))
            ]),
            'parametry': {
                'skb__k': zakres_cech,
                'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                'mlp__activation': ['relu', 'tanh', 'logistic', 'identity'],
                'mlp__solver': ['adam', 'lbfgs', 'sgd'],
                'mlp__alpha': [0.0001, 0.001, 0.01, 0.1]
            }
        },
        'XGBoost': {  
            'model': Pipeline([
                ('skb', SelectKBest(score_func=f_classif)),
                ('xgb', XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0, n_jobs=1))
            ]),
            'parametry': {
                'skb__k': zakres_cech,
                'xgb__n_estimators': [50, 100, 200, 300],
                'xgb__max_depth': randint(1, 11),
                'xgb__min_child_weight': zakres_parametrow[:5],
                'xgb__learning_rate': loguniform(1e-3, 3e-1),
                'xgb__gamma': [0, 0.1, 0.2], 
                'xgb__subsample': [0.7, 0.8, 1.0]
            }
        }
    }
    # testujemy różne algorytmy z różnymi parametrami
    # dzieki temu nie musze na początku wybierać algorytmu
    # po dopasowaniu najlepszych (znalezionych) parametrów moge wybrać algorytm

    najlepsze_modele = {}

    for nazwa, konfiguracja in modele_do_sprawdzenia.items():
        print(f"Optymalizacja dla: {nazwa}...", flush=True)

        ilosc_iteracji = 300  # <--- ZMIENIASZ ILOŚĆ ITERACJI W TYM MIEJSCU
        try:
            l_kombinacji = len(ParameterGrid(konfiguracja['parametry']))
            n_cand = min(ilosc_iteracji, l_kombinacji)
        except TypeError:
            n_cand = ilosc_iteracji

        szukaj = HalvingRandomSearchCV(
            estimator=konfiguracja['model'],
            param_distributions=konfiguracja['parametry'],
            n_candidates=n_cand,
            # ilość kombinacji do przetestowania

            min_resources=40,
            cv=skf,
            scoring='accuracy',
            random_state=42,
            n_jobs=1
        )
        szukaj.fit(X_train, y_train)
        najlepsze_modele[nazwa] = szukaj 
        
        print(f"Najlepszy wynik (CV): {szukaj.best_score_:.3f}", flush=True)
        print(f"Najlepsze parametry: {szukaj.best_params_}\n", flush=True)

    print("--- Ostateczne starcie na zbiorze testowym ---", flush=True)

    for nazwa, zoptymalizowany_model in najlepsze_modele.items():
        print(f"\n===== Wyniki dla: {nazwa} =====", flush=True)
        
        y_pred = zoptymalizowany_model.predict(X_test)
        
        print(classification_report(y_test, y_pred), flush=True)
        print("Macierz pomyłek:", flush=True)
        print(confusion_matrix(y_test, y_pred), flush=True)

    # zapisuje najlepsze algorytmy z pareamtry do pliku

    sciezka_parametry = Path('02_podstawy_aiml/najlepsze_parametry.json')
    wszystkie_parametry = {}
    for nazwa, szukaj in najlepsze_modele.items():
        czyste_parametry = {}
        for k, v in szukaj.best_params_.items():
            if hasattr(v, 'item'):
                czyste_parametry[k] = v.item()
            else:
                czyste_parametry[k] = v
        wszystkie_parametry[nazwa] = czyste_parametry

    with open(sciezka_parametry, 'w', encoding='utf-8') as plik:
        json.dump(wszystkie_parametry, plik, indent=4, ensure_ascii=False)

    print(f"\nZapisano parametry wszystkich modeli do: {sciezka_parametry}", flush=True)
 
    print("\n--- Generowanie wykresu Feature Importance (XGBoost) ---", flush=True)

    najlepszy_potok_xgb = najlepsze_modele['XGBoost'].best_estimator_
    selektor_cech = najlepszy_potok_xgb.named_steps['skb']
    model_xgb = najlepszy_potok_xgb.named_steps['xgb']

    wybrane_maska = selektor_cech.get_support()
    nazwy_wybranych_cech = X.columns[wybrane_maska]
    wagi = model_xgb.feature_importances_

    df_wagi = pd.DataFrame({
        'Cecha': nazwy_wybranych_cech,
        'Ważność': wagi
    }).sort_values(by='Ważność', ascending=False)

    print("\nWażność cech (XGBoost):", flush=True)
    print(df_wagi.to_string(index=False), flush=True)

    plt.figure(figsize=(10, 6))
    plt.barh(df_wagi['Cecha'][::-1], df_wagi['Ważność'][::-1], color='teal')
    plt.title('Zależność decyzyjna modelu (Feature Importance) - XGBoost')
    plt.xlabel('Wpływ na predykcję (skala 0.0 - 1.0)')
    plt.ylabel('Cecha wejściowa')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()


    plt.show()

if __name__ == '__main__':
    main()

