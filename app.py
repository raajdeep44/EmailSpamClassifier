import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Initialize tfidf and model to None
tfidf = None
model = None

# Load the pre-trained TF-IDF vectorizer and model
try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    st.write("TF-IDF Vectorizer loaded successfully.")
except FileNotFoundError:
    st.error("The vectorizer.pkl file was not found. Please ensure it is in the correct directory.")
except Exception as e:
    st.error(f"An error occurred while loading the vectorizer.pkl file: {e}")

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    st.write("Model loaded successfully.")
except FileNotFoundError:
    st.error("The model.pkl file was not found. Please ensure it is in the correct directory.")
except Exception as e:
    st.error(f"An error occurred while loading the model.pkl file: {e}")

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if tfidf is None or model is None:
        st.error("The model or vectorizer is not properly loaded.")
    elif not input_sms.strip():
        st.error("Please enter a message to classify.")
    else:
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        try:
            vector_input = tfidf.transform([transformed_sms])
            # 3. predict
            result = model.predict(vector_input)[0]
            # 4. Display
            if result == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")