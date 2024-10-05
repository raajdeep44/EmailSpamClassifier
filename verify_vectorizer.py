import pickle

try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    print("TF-IDF Vectorizer loaded successfully.")
except Exception as e:
    print(f"An error occurred while loading the vectorizer.pkl file: {e}")
