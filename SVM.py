from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import json
import os
import pandas as pd
import numpy as np

# Functions for saving/loading parameters
def save_best_params(params, filename="best_params_svm.json"):
    with open(filename, "w") as f:
        json.dump(params, f)
    print(f"Best parameters saved to {filename}.")

def load_best_params(filename="best_params_svm.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            params = json.load(f)
        print(f"Best parameters loaded from {filename}.")
        return params
    else:
        print(f"{filename} not found. Parameters will be tuned.")
        return None

# Main function for SVM with GridSearchCV
def svm_model(X, y, use_saved_params=False, param_file="best_params_svm.json"):
    # Load best parameters if requested
    if use_saved_params:
        best_params = load_best_params(param_file)
        if best_params:
            model = SVC(**best_params, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return {
                "model_name": "Support Vector Machine",
                "accuracy": accuracy_score(y_test, y_pred),
                "report": classification_report(y_test, y_pred, output_dict=True),
                "best_params": best_params,
            }

    # Parameter Tuning with GridSearchCV
    print("Starting parameter tuning using GridSearchCV...")
    param_grid = {
        "C": [0.1, 1, 10],  # Regularization parameter
        "kernel": ["linear", "rbf"],  # Kernel type
        "gamma": ["scale", "auto"],  # Kernel coefficient
    }
    model = SVC(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring="f1_macro", cv=3, verbose=2, n_jobs=-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    grid_search.fit(X_train, y_train)

    # Retrieve best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")

    # Save best parameters
    save_best_params(best_params, param_file)

    # Train the model with best parameters
    model = SVC(**best_params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Return results
    return {
        "model_name": "Support Vector Machine",
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "best_params": best_params,
    }

