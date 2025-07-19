from flask import Flask, request, render_template, jsonify
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline
import torch
from PyPDF2 import PdfReader
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Initialize sentiment analysis tools
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
sia = SentimentIntensityAnalyzer()

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

# Route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
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
            # Use the provided text
            pass
        else:
            return jsonify({"error": "Please upload a PDF file."})

        # Summarize the text
        summary = summarize_text(text)

        # Perform sentiment analysis using NLTK's SentimentIntensityAnalyzer
        nltk_sentiment = sia.polarity_scores(text)

        # Optionally, you can use Hugging Face's pipeline sentiment analysis
        # sentiment = sentiment_pipeline(text)[0]

        # Using TextBlob for further sentiment polarity (optional)
        text_blob_analysis = TextBlob(text)
        text_blob_sentiment = text_blob_analysis.sentiment.polarity

        # Send both summary and sentiment results to the template
        return render_template('results2.html', summary=summary, sentiment=nltk_sentiment, blob_sentiment=text_blob_sentiment)

    return render_template('index2.html')


# Flask route to render the home page
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/index2')
def topic_modeling():
    return render_template('index2.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
