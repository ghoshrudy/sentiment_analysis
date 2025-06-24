# %%
import pandas as pd
import re
import string
#import emoji
#from pycontractions import Contractions
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

#Load data

#Train data
twitter_data_train=pd.read_csv("/Users/r0g0aci/Documents/Personal/Python/sentiment_analysis/archive/Test.csv")
corpus= twitter_data_train['text'].tolist()
labels= twitter_data_train['label'].tolist()
lemmatizer=WordNetLemmatizer()
#Cleaning text 
def clean_text(text):
    if text is None or not isinstance(text, str):
        return ''
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    #text = emoji.demojize(text)
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    #contractions = Contractions()
    #contractions.load_models()
    #text = contractions.expand_texts([text])[0]  # Expand contractions
    return text

corpus = [clean_text(text) for text in corpus]
# #lemmatization
lemmatizer=WordNetLemmatizer()
def word_lemmatize(text):
    doc=nlp(text)
    return [token.lemma_ for token in doc]
tokenizer=word_lemmatize

# Create feature sets for training and testing
count_vectorizer = feature_extraction.text.CountVectorizer(stop_words='english',ngram_range=(1,2), max_features=500,tokenizer=word_lemmatize)

#Model - Naive Bayes Classifier
model=make_pipeline(count_vectorizer,MultinomialNB())
model.fit(corpus,labels)
predicted=model.predict(corpus)
print ("Predictions:", predicted[:10])
print ("Classification report:\n", classification_report(labels,predicted))

# Save the model
import pickle
with open('/Users/r0g0aci/Documents/Personal/Python/sentiment_analysis/archive/sentiment_app/sentiment_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

# %%



