import numpy as np
from flask import Flask, request, render_template, session, jsonify
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
from gensim.models.ldamodel import LdaModel
from gensim import corpora
import os
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # For lemmatizer language models

from transformers import BartForConditionalGeneration, BartTokenizer, pipeline
from PyPDF2 import PdfReader
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


# Check current working directory (debugging purposes)
print("Current working directory:", os.getcwd())

# Flask app initialization
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # For session management

# Loading pre-trained models
cv = pickle.load(open('../models/count_vectorizer_file.pkl', 'rb'))  # CountVectorizer model
feature_names = pickle.load(open('../models/feature_names_file.pkl', 'rb'))  # Feature names list
tfidf_transformer = pickle.load(open('../models/tfidf_transformer_file.pkl', 'rb'))  # TF-IDF transformer

# Load the saved LDA model and dictionary for topic extraction
lda_model = LdaModel.load('../models/lda_model.model')
with open('../models/dictionary.pkl', 'rb') as f:
    dictionary = pickle.load(f)

# Set of stopwords with additional custom words to filter out irrelevant text
stop_words = set(stopwords.words('english'))
custom_stopwords = ["fig", "figure", "image", "sample", "using", "show", "result", "large",
                    "also", "one", "two", "three", "four", "five", "seven", "eight", "nine"]
stop_words = stop_words.union(custom_stopwords)


# Preprocessing function for text cleaning and stemming
# Define the stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocessing_text(txt):
    # Convert to lowercase
    txt = txt.lower()

    # Remove HTML tags
    txt = re.sub(r'<.*?>', ' ', txt)

    # Remove non-alphabetical characters and extra spaces
    txt = re.sub(r'[^a-zA-Z\s]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()

    # Tokenize the text
    txt = nltk.word_tokenize(txt)

    # Remove stopwords and words with less than 3 characters
    txt = [word for word in txt if word not in stop_words and len(word) > 3]

    # Apply lemmatization
    txt = [lemmatizer.lemmatize(word) for word in txt]

    # Join the words back into a string
    return " ".join(txt)

# Function to extract keywords using TF-IDF and feature names
def get_keywords(docs, topN=10):
    docs_words_count = tfidf_transformer.transform(cv.transform([docs]))
    docs_words_count = docs_words_count.tocoo()  # Convert sparse matrix to COO format

    # Sort words by TF-IDF score
    tuples = zip(docs_words_count.col, docs_words_count.data)
    sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
    sorted_items = sorted_items[:topN]  # Get top N keywords

    # Extract feature names and scores
    score_vals = []
    feature_vals = []
    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    results = {feature_vals[i]: score_vals[i] for i in range(len(feature_vals))}
    return results

# Route for keyword extraction from an uploaded file
@app.route('/extract_keywords', methods=['POST'])
def extract_keywords():
    file = request.files.get('file')
    text_input = request.form.get('text_input')
    num_keywords = request.form.get('num_keywords', default=20, type=int)

    # Check if either a file or text input is provided
    if not file and not text_input:
        return render_template('Keyword_Extraction.html', error='Please upload a file or enter text.')

    cleaned_file = None

    if file and file.filename != '':
        # Handle file upload
        file_content = file.read().decode('utf-8', errors='ignore')
        cleaned_file = preprocessing_text(file_content)

    elif text_input and text_input.strip() != '':
        # Handle text input
        cleaned_file = preprocessing_text(text_input)

    # Ensure cleaned_file is not None
    if cleaned_file is None:
        return render_template('Keyword_Extraction.html', error='Invalid file or text input.')

    # Extract keywords using the cleaned text
    keywords = get_keywords(cleaned_file, num_keywords)

    # Store keywords in session for later search
    session['extracted_keywords'] = list(keywords.keys())

    return render_template('Keyword_Extraction_Results.html', keywords=keywords)


# Route to search within extracted keywords
@app.route('/search_keywords', methods=['POST', 'GET'])
def search_keywords():
    search_query = request.form.get('search')
    if search_query:
        extracted_keywords = session.get('extracted_keywords', [])  # Extract stored keywords

        # Search for matching keywords
        matching_keywords = [keyword for keyword in extracted_keywords if search_query.lower() in keyword.lower()]

        return render_template('Keyword_Search_Results.html', keywords=matching_keywords)
    return render_template('Keyword_Extraction.html')


# Route for viewing topics (after extracting them)
@app.route('/topics', methods=['GET'])
def view_topics():
    topics = session.get('extracted_topics', [])
    if topics:
        return render_template('topics_Extracted.html', topics=topics)
    return render_template('Keyword_Extraction.html', error='No topics extracted. Please upload a file first.')


# The convert_to_float function is designed to recursively convert numpy float
# types to regular Python float types, making data easier to work with and
# compatible with JSON serialization.

def convert_to_float(data):
    if isinstance(data, dict):
        return {key: convert_to_float(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_float(item) for item in data]
    elif isinstance(data, np.float32):
        return float(data)  # Convert numpy.float32 to regular float
    else:
        return data


@app.route('/extract_topics', methods=['POST'])
def extract_topics():
    file = request.files.get('file')
    text_input = request.form.get('text_input')

    # Validate that the user provided either a file or text input
    if not file and not text_input:
        return render_template('topic_modeling.html', error='Please upload a file or enter text.')

    if file and file.filename != '':
        # Handle file upload
        file_content = file.read().decode('utf-8', errors='ignore')
        cleaned_file = preprocessing_text(file_content)

    elif text_input and text_input.strip() != '':
        # Handle text input
        cleaned_file = preprocessing_text(text_input)

    else:
        return render_template('topic_modeling.html', error='Invalid file or text input.')

    # Convert text to bag of words and predict topics
    bow_vector = dictionary.doc2bow(cleaned_file.split())
    topics = lda_model.get_document_topics(bow_vector)

    # Convert topics to standard Python types (floats)
    topics = convert_to_float(topics)

    # Convert topics to a JSON serializable format (tuple to list)
    topics = [(int(topic_id), float(score)) for topic_id, score in topics]

    # Store the topics in session for later retrieval
    session['extracted_topics'] = topics

    return render_template('topics_Extracted.html', topics=topics)


# Flask route to render the main page
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/topic_modeling')
def topic_modeling():
    return render_template('topic_modeling.html')

@app.route('/keyword_extraction')
def keyword_extraction():
    return render_template('Keyword_Extraction.html')

@app.route('/index2')
def index2_view():
    return render_template('index2.html')



# Load pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Initialize sentiment analysis tools
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
sia = SentimentIntensityAnalyzer()

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

# Function to summarize text
def summarize_text(text, max_length=150, min_length=50):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@app.route('/summarize', methods=['GET', 'POST'])
def summarize():
    if request.method == 'POST':
        text = request.form.get('text')
        file = request.files.get('file')

        # Check if the user has provided neither input or both inputs
        if not text and not file:
            return jsonify({"error": "Please provide either a text input or upload a PDF file."})

        if text and file:
            return jsonify({"error": "Please provide either a text input or upload a PDF file, not both."})

        if file and file.filename.endswith('.pdf'):
            text = extract_text_from_pdf(file)
        elif text:
            pass
        else:
            return jsonify({"error": "Please upload a PDF file."})

        # Summarize the text
        summary = summarize_text(text)

        # Perform sentiment analysis using NLTK's SentimentIntensityAnalyzer
        nltk_sentiment = sia.polarity_scores(text)

        # Using TextBlob for further sentiment polarity (optional)
        text_blob_analysis = TextBlob(text)
        text_blob_sentiment = text_blob_analysis.sentiment.polarity

        # Send both summary and sentiment results to the template
        return render_template('results2.html', summary=summary, sentiment=nltk_sentiment, blob_sentiment=text_blob_sentiment)

    return render_template('index2.html')




# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5001)

