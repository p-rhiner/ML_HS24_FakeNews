import pandas as pd
import re
import nltk
import numpy as np
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Stoppwörter herunterladen
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.discard('not')
stop_words.discard('no')

# Fortschrittsanzeige einrichten
tqdm.pandas()

# Basisverzeichnis
project_dir = os.path.dirname(os.path.abspath(__file__))

# Pfade für Dateien
csv_file_path = os.path.join(project_dir, "WELFake_Dataset.csv")

# CSV-Datei lesen
df = pd.read_csv(csv_file_path)
df = df.drop('Unnamed: 0', axis=1)
df['text'] = df['text'].astype(str)

# Textbereinigung
def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['cleaned_text'] = df['text'].progress_apply(clean_text)

# TF-IDF-Vektorisierung
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text']).toarray()
y = df['label']

# Daten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost-Modell erstellen und trainieren
model = XGBClassifier(tree_method="hist", n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# Vorhersagen und Bewertung
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
