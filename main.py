import pandas as pd
import re
import nltk
import numpy as np
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from gradient_boosting import gradient_boosting
from multinomial_naive_bayes import multinomial_nb
from logistic_regression import logistic_regression
from random_forest import random_forest
from svm import svm_model

# Cleaning Data function
def clean_text(text, stop_words):
    if isinstance(text, str):
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphanumeric characters
        text = text.lower()  # Convert to lowercase
        text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
        return text
    else:
        return ''

# Main program
def main():
    # Activate progress bar (visual enhancement)
    tqdm.pandas()

    # Setting directory of project
    project_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths
    csv_file_path = os.path.join(project_dir, "WELFake_Dataset.csv")
    pkl_file_path = os.path.join(project_dir, "cleaned_welfake_dataset.pkl")

    # Loading stop-words
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.discard('not')
    stop_words.discard('no')

    # Load data file and clean-up
    print("Loading dataset...")
    if os.path.exists(pkl_file_path):
        print(f"Loading cleaned dataset from {pkl_file_path}.")
        df = pd.read_pickle(pkl_file_path)
    else:
        print(f"Cleaning and processing dataset from {csv_file_path}.")
        df = pd.read_csv(csv_file_path).drop('Unnamed: 0', axis=1)
        df['text'] = df['text'].astype(str)
        df['hashtags'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word.startswith('#')]))
        df['text_without_hashtags'] = df['text'].apply(
            lambda x: ' '.join([word for word in x.split() if not word.startswith('#')]))
        df['cleaned_text'] = df['text_without_hashtags'].progress_apply(lambda x: clean_text(x, stop_words))
        df.to_pickle(pkl_file_path)
        print(f"Cleaned dataset saved as {pkl_file_path}.")

    # TF-IDF Vectorization
    print("Vectorizing data...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_cleaned_text = vectorizer.fit_transform(df['cleaned_text']).toarray()

    hashtag_vectorizer = TfidfVectorizer(max_features=1000)
    X_hashtags = hashtag_vectorizer.fit_transform(df['hashtags']).toarray()

    X = np.concatenate((X_cleaned_text, X_hashtags), axis=1)
    y = df['label']

    # Prompt user for parameter tuning choice
    use_saved_params = input("Do you want to use saved parameters? (yes/no): ").strip().lower() == 'yes'

    # Initialize list to store all results
    all_results = []

    # Execute Gradient Boosting
    print("Starting Gradient Boosting...")
    results_gb = gradient_boosting(X, y, use_saved_params=use_saved_params)
    print(f"Model Name: {results_gb['model_name']}")
    print(f"Model Accuracy: {results_gb['accuracy']:.2f}")
    print("Classification Report:")
    print(results_gb['report'])
    all_results.append({
        "Model": results_gb['model_name'],
        "Accuracy": results_gb['accuracy'],
        "Best Parameters": results_gb.get("best_params", "N/A"),
        "Report": results_gb['report']
    })

    # Execute Multinomial Naive Bayes
    print("Starting Multinomial Naive Bayes...")
    results_mnb = multinomial_nb(X, y, use_saved_params=use_saved_params)
    print(f"Model Name: {results_mnb['model_name']}")
    print(f"Model Accuracy: {results_mnb['accuracy']:.2f}")
    print("Classification Report:")
    print(results_mnb['report'])
    all_results.append({
        "Model": results_mnb['model_name'],
        "Accuracy": results_mnb['accuracy'],
        "Best Parameters": results_mnb.get("best_params", "N/A"),
        "Report": results_mnb['report']
    })

    # Execute Logistic Regression
    print("Starting Logistic Regression...")
    results_lr = logistic_regression(X, y, use_saved_params=use_saved_params)
    print(f"Model Name: {results_lr['model_name']}")
    print(f"Model Accuracy: {results_lr['accuracy']:.2f}")
    print("Classification Report:")
    print(results_lr['report'])
    all_results.append({
        "Model": results_lr['model_name'],
        "Accuracy": results_lr['accuracy'],
        "Best Parameters": results_lr.get("best_params", "N/A"),
        "Report": results_lr['report']
    })

    # Execute Random Forest
    print("Starting Random Forest...")
    results_rf = random_forest(X, y, use_saved_params=use_saved_params)
    print(f"Model Name: {results_rf['model_name']}")
    print(f"Model Accuracy: {results_rf['accuracy']:.2f}")
    print("Classification Report:")
    print(results_rf['report'])
    all_results.append({
        "Model": results_rf['model_name'],
        "Accuracy": results_rf['accuracy'],
        "Best Parameters": results_rf.get("best_params", "N/A"),
        "Report": results_rf['report']
    })

    # Save results to CSV
    output_path = os.path.join(project_dir, "results_summary.csv")
    results_df = pd.DataFrame([{
        "Model": result["Model"],
        "Accuracy": result["Accuracy"],
        "Best Parameters": result["Best Parameters"],
        "Classification Report": json.dumps(result["Report"])
    } for result in all_results])

    results_df.to_csv(output_path, index=False)    
    print(f"Results summary with classification report saved to {output_path}")

    # Display overview of all models
    print("\nOverview of all models:")
    for result in all_results:
        print("-" * 40)
        print(f"Model: {result['Model']}")
        print(f"Accuracy: {result['Accuracy']:.2f}")
        print("Classification Report:")
        for label, metrics in result['Report'].items():
            if isinstance(metrics, dict):  # Ignore overall metrics like "accuracy"
                print(f"  {label}:")
                for metric, value in metrics.items():
                    print(f"    {metric}: {value:.2f}")
        print(f"Best Parameters: {result['Best Parameters']}")
        print("-" * 40)


    # # Execute Support Vector Machine
    # print("Starting Support Vector Machine...")
    # results_svm = svm_model(X, y, use_saved_params=use_saved_params)
    # print(f"Model Name: {results_svm['model_name']}")
    # print(f"Model Accuracy: {results_svm['accuracy']:.2f}")
    # print("Classification Report:")
    # print(results_svm['report'])

if __name__ == "__main__":
    main()
