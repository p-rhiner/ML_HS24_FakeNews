
import sys
print("Aktiver Python-Interpeter: ", sys.executable)
print("Python-Version: ", sys.version)

print("I am a Naive-Bayes.")

import pandas as pd
import re
import nltk
import numpy as np
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

#downloading of stop words which will later be removed from news articles except "not", since I thought it could be of help to the machine learning
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.discard('not')
stop_words.discard('no')

# setting up tqdm to see a progress bar while working (only for my own joy, please ignore)
tqdm.pandas()

# Base directory (actual folder of script)
project_dir = os.path.dirname(os.path.abspath(__file__))

# Paths for files relative to project folder
csv_file_path = os.path.join(project_dir, "WELFake_Dataset.csv")
pkl_file_path = os.path.join(project_dir, "cleaned_welfake_dataset.pkl")

# funktion for cleaning text
def clean_text(text):
    if isinstance(text, str):
        # Entfernen von URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Entfernen von allen nicht-alphanumerischen Zeichen außer Leerzeichen
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Alles in Kleinbuchstaben umwandeln
        text = text.lower()

        # Entfernen von Stoppwörtern
        text = ' '.join([word for word in text.split() if word not in stop_words])

        return text
    else:
        return ''

#read original data file (csv)
df = pd.read_csv(csv_file_path)

# print(df.columns)
# print(df.head())


# Remove unnecessary columns ('Unnamed')
df = df.drop('Unnamed: 0', axis=1)

# # Has the column been removed successfully?
# print(df.columns)
# print(df.head())

# show 'text' column 
print("Showing column 'text' (for reference purpose): \n")
print(df['text'].head())

# making sure to save the text column as string
df['text'] = df['text'].astype(str)

# show label count (fake = 1 /real = 0)
print(f"Label count: \n{df['label'].value_counts()}")

# extract hashtags and saving to a new column:
df['hashtags'] = df['text'].apply(lambda x: ' '.join([word for word in str(x).split() if word.startswith('#')]))


# Saving text without hashtags
df['text_without_hashtags'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if not word.startswith('#')]) if isinstance(x, str) else '')

print("Printing df.head for reference purposes: \n")
print(df.head)


# executing function (executed after extracting hashtags to not lose hashtags in advance)
df['cleaned_text'] = df['text_without_hashtags'].progress_apply(clean_text)

print("Printing df.columns")
print(df.columns)

# Show columns 'text', 'hashtags' and 'cleaned_text' to check the results:
print(df[['text', 'hashtags', 'cleaned_text']].head())

#Exemplary query from database (for my understanding):
print("Exemplary query from database (showing first instance's hashtags):\n" + df['hashtags'].iloc[0])

# save pandas dataframe as pickle file (binary format, which is quicker than csv)
df.to_pickle(pkl_file_path)
print("File saved as pickle.")

# Bereinigten DataFrame aus der Pickle-Datei laden
df = pd.read_pickle('cleaned_welfake_dataset.pkl')
if not df.empty:
    print("Pickle file loaded successfully.")
else:
    print("Error: DataFrame is empty.")



# Making sure the data file has been loaded correctly
print("df.columns after loading of file")
print(df.columns)

# Creating and configuring TF-IDF Vectorizer 
vectorizer = TfidfVectorizer(max_features=5000)

# Transforming cleaned texts (cleaned_text) to numerical vectors
X_cleaned_text = vectorizer.fit_transform(df['cleaned_text']).toarray()

# Creating a second TF-IDF Vectorizer since I wanted to use them as a feature too
hashtag_vectorizer = TfidfVectorizer(max_features=1000)
X_hashtags = hashtag_vectorizer.fit_transform(df['hashtags']).toarray()

# Combining of the two afore-mentioned TF-IDF Vectorizers in one:
X = np.concatenate((X_cleaned_text, X_hashtags), axis=1)


# Checking dimensions of created vectors
print("Printing X.shape to check dimensions of created vectors: \n")
print(X.shape)

# Labels (fake/real)
y = df['label']

# Splitting data into training and test set (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Outputting size of trainings and test data set
print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")

# Creating model and training
model = MultinomialNB(alpha = 0.1)
model.fit(X_train, y_train)

# Output to confirm the model has been trained.
print("Model successfully trained.")

# Forecast on test data
y_pred = model.predict(X_test)

# Evaluating model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Output of classification report (Precision, Recall, F1-Score)
print("Classification report:\n")
print(classification_report(y_test, y_pred))