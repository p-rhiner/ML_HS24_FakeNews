import pandas as pd
import re
import nltk
import numpy as np
import os
import optuna
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm

#downloading of stop words which will later be removed from news articles except "not", since I thought it could be of help to the machine learning
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.discard('not')
stop_words.discard('no')

# Initialising tqdm to see a progress bar while working (only for my own joy, please ignore)
tqdm.pandas()

# Base directory (actual folder of script)
project_dir = os.path.dirname(os.path.abspath(__file__))

# Paths for files relative to project folder
csv_file_path = os.path.join(project_dir, "WELFake_Dataset.csv")
pkl_file_path = os.path.join(project_dir, "cleaned_welfake_dataset.pkl")

# funktion for cleaning text
def clean_text(text):
    if isinstance(text, str):
        # Removing URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Removing all non-alphanumerical characters except spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Transforming everything to lower case.
        text = text.lower()

        # Removing stop words 
        text = ' '.join([word for word in text.split() if word not in stop_words])

        return text
    else:
        return ''

#read original data file (csv)
df = pd.read_csv(csv_file_path)

# Remove unnecessary columns ('Unnamed')
df = df.drop('Unnamed: 0', axis=1)

# making sure to save the text column as string
df['text'] = df['text'].astype(str)

# extract hashtags and saving to a new column:
df['hashtags'] = df['text'].apply(lambda x: ' '.join([word for word in str(x).split() if word.startswith('#')]))

# Saving text without hashtags
df['text_without_hashtags'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if not word.startswith('#')]) if isinstance(x, str) else '')

# executing function (executed after extracting hashtags to not lose hashtags in advance)
df['cleaned_text'] = df['text_without_hashtags'].progress_apply(clean_text)

# Show columns 'text', 'hashtags' and 'cleaned_text' to check the results:
print("Example of cleaned Data File:\n")
print(df[['text', 'hashtags', 'cleaned_text']].head())

# save pandas dataframe as pickle file (binary format, which is quicker to work with than csv)
df.to_pickle(pkl_file_path)
print("File saved as pickle.\n")

# Bereinigten DataFrame aus der Pickle-Datei laden
df = pd.read_pickle('cleaned_welfake_dataset.pkl')
if not df.empty:
    print("Pickle file loaded successfully.\n")
else:
    print("Error: DataFrame is empty.\n")

    # Making sure the data file has been loaded correctly
print("df.columns after loading of file\n")
print(df.columns)

# Creating and configuring TF-IDF Vectorizer 
vectorizer = TfidfVectorizer(max_features=5000)

# Transforming cleaned texts (cleaned_text) to numerical vectors
X_cleaned_text = vectorizer.fit_transform(df['cleaned_text']).toarray()

# Creating a second TF-IDF Vectorizer since I wanted to use them as a feature too
hashtag_vectorizer = TfidfVectorizer(max_features=1000)
X_hashtags = hashtag_vectorizer.fit_transform(df['hashtags']).toarray()

# Combining of the two afore-mentioned TF-IDF Vectorizers in one as feaure 'X':
X = np.concatenate((X_cleaned_text, X_hashtags), axis=1)

# Setting labels as feature 'y' (fake=1/real=0)
y = df['label']

# Printing examples of both classes
print("Fake News Beispiel:")
print(df[df['label'] == 0]['cleaned_text'].iloc[0])
print("Real News Beispiel:")
print(df[df['label'] == 1]['cleaned_text'].iloc[0])

# Splitting data into training and test set (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test of data split
print("Train Label Distribution:")
print(y_train.value_counts())
print("Test Label Distribution:")
print(y_test.value_counts())

# Parameter-Tuning 
def objective(trial):
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 200]),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),
        'tree_method': 'hist'
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Training best model
print("Beste Parameter:", study.best_params)
best_params = study.best_params
best_model = XGBClassifier(**best_params)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Print results
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy with best parameters: {accuracy:.2f}")
print("Classification report:\n")
print(classification_report(y_test, y_pred))