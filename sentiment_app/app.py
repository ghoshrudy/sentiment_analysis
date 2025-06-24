# %%
import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from bs4 import BeautifulSoup
import nltk.classify.util
from nltk.probability import FreqDist
from nltk.classify import SklearnClassifier
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import spacy
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
# Download the spaCy model if not already present
import sys
import subprocess
try:
	spacy.load("en_core_web_sm")
except OSError:
	subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
nlp = spacy.load("en_core_web_sm")



def clean_text(text):
    if text is None or not isinstance(text, str):
        return ''
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Load the model
@st.cache_resource
def load_model():
    with open("/Users/r0g0aci/Documents/Personal/Python/sentiment_analysis/archive/sentiment_app/sentiment_classifier.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("🧠 Sentiment Classifier")
st.write("Enter a sentence to classify:")

text_input = st.text_area("Text Input")

if st.button("Classify"):
    if text_input:
        # Preprocess the text
        cleaned_text = clean_text(text_input)
        prediction = model.predict([cleaned_text])[0]
        # Map prediction to sentiment
        sentiment_map = {0: "Negative", 1: "Positive"}
        st.success(f"Predicted Sentiment: **{sentiment_map[prediction]}**")
        # Provide confidence score
        proba = np.round(model.predict_proba([cleaned_text]), 2)
        if sentiment_map[prediction]=="Positive":
           proba=proba[0][1]
        else:
           proba=proba[0][0]
        st.success(f"Confidence Scores: **{proba}**")
    else:
        st.warning("Please enter text.")


