# main.py

import zipfile
import os

# Extract model files from zip if models/ doesn't exist (e.g., in Streamlit Cloud)
if not os.path.exists("models"):
    with zipfile.ZipFile("models.zip", 'r') as zip_ref:
        zip_ref.extractall(".")


import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models.ldamodel import LdaModel
from gensim import corpora
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline
from PyPDF2 import PdfReader
from textblob import TextBlob
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.sparse import coo_matrix

# Download required NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("vader_lexicon")

# Load models
cv = pickle.load(open("models/count_vectorizer_file.pkl", "rb"))
feature_names = pickle.load(open("models/feature_names_file.pkl", "rb"))
tfidf_transformer = pickle.load(open("models/tfidf_transformer_file.pkl", "rb"))
lda_model = LdaModel.load("models/lda_model.model")
dictionary = pickle.load(open("models/dictionary.pkl", "rb"))
tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6", cache_dir="./hf_cache")
bart_model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6", cache_dir="./hf_cache")
sentiment_pipeline = pipeline("sentiment-analysis")
sia = SentimentIntensityAnalyzer()


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english")).union({
    "fig", "figure", "image", "sample", "using", "show", "result",
    "also", "one", "two", "three", "four", "five", "seven", "eight", "nine"
})

def preprocessing_text(txt):
    txt = re.sub(r"<.*?>", " ", txt.lower())
    txt = re.sub(r"[^a-zA-Z\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    tokens = nltk.word_tokenize(txt)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 3]
    return " ".join(tokens)

def get_keywords(doc, topN=10):
    transformed = tfidf_transformer.transform(cv.transform([doc]))
    transformed_coo = transformed.tocoo()  # convert to COO format
    tuples = zip(transformed_coo.col, transformed_coo.data)
    sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)[:topN]
    return {feature_names[idx]: round(score, 3) for idx, score in sorted_items}


def summarize_text(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = bart_model.generate(inputs, max_length=150, min_length=50, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def extract_text_from_pdf(file):
    return "".join([page.extract_text() for page in PdfReader(file).pages])

def lda_topics(text):
    bow_vector = dictionary.doc2bow(text.split())
    return lda_model.get_document_topics(bow_vector)

# Streamlit UI
st.title("NLP Toolkit")

task = st.sidebar.radio("Choose Task", ["Keyword Extraction", "Topic Modeling", "Summarization & Sentiment"])

uploaded_file = st.file_uploader("Upload a text or PDF file", type=["txt", "pdf"])
text_input = st.text_area("Or paste your text here")

if uploaded_file or text_input:
    if uploaded_file and uploaded_file.name.endswith(".pdf"):
        raw_text = extract_text_from_pdf(uploaded_file)
    else:
        raw_text = uploaded_file.read().decode("utf-8") if uploaded_file else text_input

    cleaned = preprocessing_text(raw_text)

    if task == "Keyword Extraction":
        top_n = st.slider("Number of Keywords", 5, 30, 10)
        if st.button("Extract Keywords"):
            keywords = get_keywords(cleaned, top_n)

            # Convert to DataFrame for table display
            df_keywords = pd.DataFrame(list(keywords.items()), columns=["Keyword", "Score"])

            st.subheader("Extracted Keywords")

            st.markdown(
                "<h3 style='text-align: center; color: #2c3e50;'>Extracted Keywords</h3>",
                unsafe_allow_html=True
            )

            st.table(df_keywords.style.format({"Score": "{:.3f}"}))

    elif task == "Topic Modeling":

        if st.button("Extract Topics"):

            topics = lda_topics(cleaned)

            if topics:

                # Convert to DataFrame

                df_topics = pd.DataFrame(topics, columns=["Topic Number", "Probability"])

                # Add a styled header

                st.markdown(

                    "<h3 style='text-align: center; color: #2c3e50;'>Extracted Topics</h3>",

                    unsafe_allow_html=True

                )

                # Display the styled table

                st.table(df_topics.style.format({"Probability": "{:.4f}"}))

            else:

                st.warning("No topics extracted. Please upload a file first.")


    elif task == "Summarization & Sentiment":
        if st.button("Summarize & Analyze"):
            summary = summarize_text(raw_text)
            vader_scores = sia.polarity_scores(raw_text)
            textblob_sentiment = TextBlob(raw_text).sentiment

            # Styled headers
            st.markdown(
                "<h3 style='text-align: center; color: #2c3e50;'>Summarized Text</h3>",
                unsafe_allow_html=True
            )
            st.success(summary)

            # Sentiment section
            st.markdown(
                "<h3 style='text-align: center; color: #2c3e50;'>Sentiment Analysis</h3>",
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### VADER Scores")
                st.metric(label="Positive", value=f"{vader_scores['pos']:.3f}")
                st.metric(label="Neutral", value=f"{vader_scores['neu']:.3f}")
                st.metric(label="Negative", value=f"{vader_scores['neg']:.3f}")
                st.metric(label="Compound", value=f"{vader_scores['compound']:.3f}")

            with col2:
                st.markdown("#### TextBlob Sentiment")
                st.metric(label="Polarity", value=f"{textblob_sentiment.polarity:.3f}")
                st.metric(label="Subjectivity", value=f"{textblob_sentiment.subjectivity:.3f}")


else:
    st.info("Upload a file or paste text to begin.")
