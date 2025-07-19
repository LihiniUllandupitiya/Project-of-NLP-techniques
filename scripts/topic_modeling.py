import numpy as np
from flask import Flask, request, render_template, session
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
from gensim.models.ldamodel import LdaModel
from gensim import corpora
import os

# Check current working directory (debugging purposes)
print("Current working directory:", os.getcwd())

# Flask app initialization
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # For session management

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
def preprocessing_text(txt):
    txt = txt.lower()  # Convert to lowercase
    txt = re.sub(r'<.*?>', ' ', txt)  # Remove HTML tags
    txt = re.sub(r'[^a-zA-Z]', ' ', txt)  # Remove non-alphabetical characters
    txt = nltk.word_tokenize(txt)  # Tokenize the text
    txt = [word for word in txt if word not in stop_words]  # Remove stopwords
    txt = [word for word in txt if len(word) > 3]  # Remove short words
    stemming = PorterStemmer()
    txt = [stemming.stem(word) for word in txt]  # Stem the words

    print("Processed text:", txt)  # Debugging: print the processed text
    return " ".join(txt)  # Return the processed text as a single string


# Flask route to render the main page
@app.route('/')
def index():
    return render_template('topic_modeling.html')

# Route for viewing topics (after extracting them)
@app.route('/topics', methods=['GET'])
def view_topics():
    topics = session.get('extracted_topics', [])
    if topics:
        return render_template('topics_Extracted.html', topics=topics)
    return render_template('topic_modeling.html', error='No topics extracted. Please upload a file first.')


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



# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5001)  # or any other port that's free

