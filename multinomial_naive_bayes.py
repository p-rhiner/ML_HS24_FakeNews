from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import json
import os
import pandas as pd
import numpy as np

# Functions for saving/loading parameters
def save_best_params(params, filename="best_params_nb.json"):
    with open(filename, "w") as f:
        json.dump(params, f)
    print(f"Best parameters saved to {filename}.")

def load_best_params(filename="best_params_nb.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            params = json.load(f)
        print(f"Best parameters loaded from {filename}.")
        return params
    else:
        print(f"{filename} not found. Parameters will be tuned.")
        return None

# Main function for Multinomial Naive Bayes
def multinomial_nb(X, y, use_saved_params=False, param_file="best_params_nb.json"):
    # Load best parameters if requested
    if use_saved_params:
        best_params = load_best_params(param_file)
        if best_params:
            print("Applying saved parameters for training of model.")
            model = MultinomialNB(**best_params)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return {
                "model_name": "Multinomial Naive Bayes",
                "accuracy": accuracy_score(y_test, y_pred),
                "report": classification_report(y_test, y_pred, output_dict=True),
                "best_params": best_params,
            }

    # Parameter Tuning 
    print("Starting parameter tuning using GridSearchCV...")
    param_grid = {"alpha": [0.1, 0.5, 1.0, 2.0, 5.0]}  # Grid of alpha values
    model = MultinomialNB()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring="f1_macro", cv=3, verbose=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    grid_search.fit(X_train, y_train)

    # Retrieve best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")

    # Train the model
    model = MultinomialNB(**best_params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Save best parameters
    save_best_params(best_params, param_file)

    # Return results
    return {
        "model_name": "Multinomial Naive Bayes",
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "best_params": best_params,
    }
