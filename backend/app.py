from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load and prepare data
df = pd.read_csv('nic_2008.csv')
df['full_description'] = df['Description'] + ' ' + df['Division'] + ' ' + df['Section']

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['processed_description'] = df['full_description'].apply(preprocess_text)

vectorizer = TfidfVectorizer()
nic_vectors = vectorizer.fit_transform(df['processed_description'])

@app.route('/get_nic_codes', methods=['POST'])
def get_nic_codes():
    user_input = request.json['input']
    processed_input = preprocess_text(user_input)
    input_vector = vectorizer.transform([processed_input])
    
    similarities = cosine_similarity(input_vector, nic_vectors)
    top_3_indices = similarities.argsort()[0][-3:][::-1]
    
    results = []
    for idx in top_3_indices:
        nic_code = int(df.iloc[idx]['Sub Class'])
        description = df.iloc[idx]['Description']
        results.append({'nic_code': nic_code, 'description': description})
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)