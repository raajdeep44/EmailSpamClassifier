import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

# Keep only the necessary columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Map labels to binary values
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Extract messages and labels
corpus = data['message'].tolist()
labels = data['label'].tolist()

# Create and fit the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(corpus)

# Create and train the MultinomialNB model
model = MultinomialNB()
model.fit(X, labels)

# Save the fitted TfidfVectorizer and trained model
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("TF-IDF Vectorizer and model have been created and saved successfully.")
