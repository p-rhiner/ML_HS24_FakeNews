import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import optuna

# Function so save best parameters
def save_best_params(params, filename="best_params_gb.json"):
    with open(filename, "w") as f:
        json.dump(params, f)
    print(f"Best parameters saved to {filename}.")

# Function to load best parameters
def load_best_params(filename="best_params_gb.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            params = json.load(f)
        print(f"Best parameters loaded from {filename}.")
        return params
    else:
        print(f"{filename} not found. Parameters need to be calculated anew.")
        return None

# Main function (model training)
def gradient_boosting(X, y, use_saved_params=False, param_file="best_params_gb.json"):
    if use_saved_params:
        best_params = load_best_params(param_file)
        if best_params:
            print("Applying saved parameters for training of model.")
            model = XGBClassifier(**best_params)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return {
                "model_name": "Gradient Boosting",
                "accuracy": accuracy_score(y_test, y_pred),
                "report": classification_report(y_test, y_pred, output_dict=True),
                "best_params": best_params,
            }

    # Parameter-Tuning with optuna
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 200]),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.8, 1.0),
            'tree_method': 'hist'
        }
        model = XGBClassifier(**params)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return f1_score(y_test, y_pred)

    # Executing optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # Returning best parameters
    best_params = study.best_params
    print("Best parameters found.", best_params)
    save_best_params(best_params, param_file)
    
    # Training with best parameters
    model = XGBClassifier(**best_params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluating results
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Returning results
    return {
        "model_name": "Gradient Boosting",
        "accuracy": accuracy,
        "report": report,
        "best_params": best_params
    }

