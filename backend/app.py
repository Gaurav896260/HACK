from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spellchecker import SpellChecker
import json
import time
import logging
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Only download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)

# Load spaCy model for grammar checking and entity recognition
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Loaded spaCy model successfully")
except:
    logger.warning("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
    logger.info("Downloaded and loaded spaCy model")

# Initialize spell checker
spell = SpellChecker()

app = Flask(__name__)
CORS(app)

# Enhanced NIC Code Processor with grammar and vague language handling
class NICCodeProcessor:
    def __init__(self, csv_path, industry_dict_path=None):
        logger.info("Initializing NIC Code Processor")
        # Load stopwords once
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Load CSV file with NIC codes (using more efficient dtype specifications)
        start_time = time.time()
        logger.info(f"Loading CSV from {csv_path}")
        try:
            # Specify dtypes for more efficient memory usage
            dtypes = {
                'Sub Class': 'int32',
                'Description': 'str',
                'Division': 'str',
                'Section': 'str'
            }
            self.df = pd.read_csv(csv_path, dtype=dtypes)
            logger.info(f"CSV loaded successfully in {time.time() - start_time:.2f} seconds with {len(self.df)} rows")
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise
        
        # Load or create industry dictionary
        if industry_dict_path:
            try:
                with open(industry_dict_path, 'r') as f:
                    self.industry_synonyms = json.load(f)
                logger.info(f"Loaded industry dictionary from {industry_dict_path}")
            except Exception as e:
                logger.warning(f"Could not load industry dictionary: {str(e)}. Creating default.")
                self.create_industry_synonyms()
        else:
            self.create_industry_synonyms()
        
        # Flatten synonyms list for faster lookups
        self.flat_synonyms = {}
        for industry, synonyms in self.industry_synonyms.items():
            for synonym in synonyms:
                self.flat_synonyms[synonym] = industry
            self.flat_synonyms[industry] = industry
        
        # Load common industry examples for vague queries
        self.load_common_examples()
        
        # Track recent user queries for quick repetitive lookups
        self.query_cache = {}
        self.max_cache_size = 100
        
        # Preprocess and vectorize data
        self.preprocess_data()
    
    def create_industry_synonyms(self):
        logger.info("Creating industry synonyms dictionary")
        # Keep the existing industry_synonyms dictionary
        self.industry_synonyms = {
            'software': ['tech', 'technology', 'programming', 'coding', 'development', 'application', 'app', 'digital', 'it', 'information technology', 'software engineering', 'web', 'mobile', 'computer', 'saas', 'cloud', 'api'],
            'manufacturing': ['production', 'fabrication', 'assembly', 'factory', 'industry', 'make', 'construct', 'build', 'industrial', 'goods', 'products', 'fabricate', 'manufacture', 'manufacturing unit'],
            'consulting': ['advisory', 'guidance', 'counseling', 'consulting', 'consultant', 'advice', 'strategy', 'mentor', 'coach', 'expertise', 'specialist', 'professional service', 'business advice'],
            'retail': ['shop', 'store', 'sales', 'merchant', 'marketplace', 'selling', 'outlet', 'mall', 'e-commerce', 'commerce', 'trade', 'consumer', 'distribution', 'supermarket', 'hypermarket'],
            'agriculture': ['farming', 'crops', 'livestock', 'plantation', 'cultivation', 'farm', 'agrarian', 'agribusiness', 'horticulture', 'dairy', 'agricultural', 'poultry', 'fishery', 'organic farming'],
            'healthcare': ['medical', 'health', 'hospital', 'clinic', 'wellness', 'pharma', 'pharmaceutical', 'doctor', 'nursing', 'therapy', 'diagnostic', 'medicine', 'healthcare services', 'telemedicine'],
            'education': ['teaching', 'school', 'academic', 'training', 'learning', 'university', 'college', 'education', 'instruction', 'academy', 'tutoring', 'coaching', 'e-learning', 'edtech'],
            'finance': ['banking', 'investment', 'financial', 'money', 'capital', 'loan', 'credit', 'fund', 'insurance', 'fintech', 'wealth', 'asset', 'trading', 'stock', 'mortgage', 'microfinance'],
            'construction': ['building', 'infrastructure', 'contractor', 'architecture', 'civil', 'construction', 'real estate', 'property', 'housing', 'development', 'project', 'engineering', 'structural'],
            'transport': ['logistics', 'shipping', 'transportation', 'freight', 'delivery', 'cargo', 'courier', 'haulage', 'fleet', 'vehicle', 'transit', 'trucking', 'supply chain', 'distribution'],
            'food': ['restaurant', 'catering', 'eatery', 'cafe', 'bakery', 'food processing', 'culinary', 'kitchen', 'meal', 'dining', 'snack', 'beverage', 'grocery', 'food products'],
            'hospitality': ['hotel', 'inn', 'lodging', 'accommodation', 'resort', 'tourism', 'travel', 'motel', 'guest house', 'hospitality', 'vacation', 'leisure', 'bnb', 'homestay'],
            'entertainment': ['media', 'film', 'music', 'game', 'entertainment', 'arts', 'performance', 'production', 'studio', 'broadcast', 'streaming', 'show', 'event', 'theatre'],
            'mining': ['extraction', 'quarry', 'mineral', 'coal', 'ore', 'mining', 'drill', 'excavation', 'dig', 'resource', 'metallurgy', 'exploration', 'mine'],
            'energy': ['power', 'electricity', 'gas', 'oil', 'fuel', 'renewable', 'solar', 'wind', 'hydro', 'energy', 'utility', 'generation', 'grid', 'petroleum', 'refinery'],
            'textile': ['fabric', 'cloth', 'garment', 'apparel', 'fashion', 'textile', 'weaving', 'knitting', 'clothing', 'wear', 'mill', 'fiber', 'yarn', 'cotton'],
            'telecom': ['communication', 'network', 'telecom', 'cellular', 'phone', 'mobile', 'broadband', 'internet', 'wireless', 'telecommunication', 'connectivity', 'isp'],
            'waste': ['recycling', 'waste', 'garbage', 'disposal', 'sanitation', 'trash', 'scrap', 'junk', 'environmental', 'treatment', 'pollution', 'circular economy'],
            'biotech': ['biotechnology', 'research', 'science', 'laboratory', 'biotech', 'life science', 'biopharma', 'genomics', 'clinical', 'biological'],
            'cybersecurity': ['security', 'cyber', 'protection', 'defense', 'encryption', 'firewall', 'privacy', 'compliance', 'data protection', 'information security']
        }
    
    def load_common_examples(self):
        # Examples for vague queries that map to specific industries
        self.common_examples = {
            'start a business': ['retail shop', 'consulting service', 'online store'],
            'open a shop': ['retail', 'grocery store', 'clothing shop'],
            'make things': ['manufacturing', 'production', 'crafts'],
            'build stuff': ['construction', 'carpentry', 'fabrication'],
            'sell products': ['retail', 'e-commerce', 'wholesale'],
            'grow food': ['agriculture', 'farming', 'crop cultivation'],
            'help people': ['healthcare', 'consulting', 'social services'],
            'teach': ['education', 'training', 'coaching'],
            'provide service': ['consulting', 'professional services', 'maintenance'],
            'fix things': ['repair', 'maintenance', 'technical service'],
            'handle money': ['finance', 'accounting', 'banking'],
            'transport goods': ['logistics', 'shipping', 'delivery'],
            'feed people': ['restaurant', 'catering', 'food service'],
            'create software': ['software development', 'programming', 'tech solutions'],
            'process data': ['data analytics', 'information processing', 'data management'],
            'online business': ['e-commerce', 'digital service', 'online platform']
        }
        
        # Create compiled regex patterns for vague query matching (faster than substring search)
        self.vague_patterns = {}
        for vague_query in self.common_examples:
            pattern = r'\b' + r'\b|\b'.join(vague_query.split()) + r'\b'
            self.vague_patterns[vague_query] = re.compile(pattern, re.IGNORECASE)
    
    @lru_cache(maxsize=1000)
    def correct_spelling(self, word):
        """Cache-enabled spell checker for individual words"""
        if word.lower() in self.flat_synonyms:
            return word  # Don't correct industry terms
        return spell.correction(word) or word
    
    def correct_grammar(self, text):
        """Fix grammar issues in the input text"""
        # Basic spell checking with caching
        words = text.split()
        corrected_words = [self.correct_spelling(word) for word in words]
        
        corrected_text = ' '.join(corrected_words)
        was_corrected = corrected_text != text
        
        # Simplified grammar checking
        # Skip full spaCy processing for basic corrections
        return corrected_text, was_corrected
    
    @lru_cache(maxsize=5000)
    def lemmatize_word(self, word):
        """Cache lemmatization results for better performance"""
        return self.lemmatizer.lemmatize(word)
    
    def basic_preprocess(self, text):
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove special characters and digits (using more efficient regex)
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Split by whitespace (faster than word_tokenize for simple cases)
        tokens = text.split()
        
        # Remove stopwords and lemmatize with caching
        tokens = [self.lemmatize_word(word) for word in tokens 
                 if word not in self.stopwords and len(word) > 2]
        
        return ' '.join(tokens)
    
    def preprocess_data(self):
        logger.info("Starting data preprocessing")
        start_time = time.time()
        
        # Combine relevant columns for better matching (more memory efficient)
        self.df['full_description'] = (
            self.df['Description'] + ' ' + 
            self.df['Division'] + ' ' + 
            self.df['Section']
        )
        
        # Process descriptions in chunks to reduce memory usage
        chunk_size = 500
        processed_texts = []
        
        for i in range(0, len(self.df), chunk_size):
            chunk = self.df['full_description'].iloc[i:i+chunk_size]
            processed_chunk = [self.basic_preprocess(text) for text in chunk]
            processed_texts.extend(processed_chunk)
            
            # Log progress for larger datasets
            if i % 5000 == 0 and i > 0:
                logger.info(f"Processed {i}/{len(self.df)} descriptions")
        
        self.df['processed_description'] = processed_texts
        
        # Vectorization with optimized max_features
        logger.info("Vectorizing descriptions")
        # Using fewer features to reduce memory footprint without losing too much accuracy
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.nic_vectors = self.vectorizer.fit_transform(self.df['processed_description'])
        
        # Clean up memory
        self.df.drop('full_description', axis=1, inplace=True)
        
        logger.info(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
    
    def handle_vague_query(self, query):
        """Handle vague business descriptions by suggesting more specific alternatives"""
        query_lower = query.lower()
        
        # Check if the query matches any common vague examples using regex (faster)
        for vague_query, pattern in self.vague_patterns.items():
            if pattern.search(query_lower):
                return True, self.common_examples[vague_query]
        
        # Check if the query is too short or general
        words = query_lower.split()
        if len(words) < 3 and not any(word in self.flat_synonyms for word in words):
            # Return a smaller set of suggestions for very vague queries
            return True, ["retail", "manufacturing", "services", "technology"]
        
        return False, []
    
    def expand_query(self, query):
        # More efficient query expansion
        tokens = set(self.basic_preprocess(query).split())
        expanded_tokens = set(tokens)  # Use sets for faster operations
        
        # Add industry synonyms
        for token in tokens:
            if token in self.flat_synonyms:
                industry = self.flat_synonyms[token]
                # Add the industry term and a subset of relevant synonyms
                expanded_tokens.add(industry)
                expanded_tokens.update(self.industry_synonyms[industry][:5])  # Limit to 5 synonyms
        
        # More lightweight entity extraction
        doc = nlp(query, disable=['parser'])  # Disable parser for speed
        
        # Only extract the most relevant entities
        for entity in doc.ents:
            if entity.label_ in ['ORG', 'PRODUCT']:
                expanded_tokens.add(entity.text.lower())
        
        return ' '.join(expanded_tokens)
    
    def find_nic_codes(self, query, top_n=5):
        logger.info(f"Processing query: {query}")
        start_time = time.time()
        
        # Check cache first
        cache_key = query.lower().strip()
        if cache_key in self.query_cache:
            logger.info("Using cached result")
            return self.query_cache[cache_key]
        
        # Check for grammar issues and correct
        corrected_query, was_corrected = self.correct_grammar(query)
        
        # Check if the query is vague
        is_vague, suggestions = self.handle_vague_query(corrected_query)
        
        # Expand and preprocess query
        expanded_query = self.expand_query(corrected_query)
        
        # Vectorize query
        query_vector = self.vectorizer.transform([expanded_query])
        
        # Compute similarities (using cosine_similarity for sparse matrices)
        similarities = cosine_similarity(query_vector, self.nic_vectors)[0]
        
        # Use faster numpy operations for finding top matches
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        results = []
        min_threshold = 0.05
        for idx in top_indices:
            if similarities[idx] > min_threshold:
                results.append({
                    'nic_code': int(self.df.iloc[idx]['Sub Class']),
                    'description': self.df.iloc[idx]['Description'],
                    'division': self.df.iloc[idx]['Division'],
                    'section': self.df.iloc[idx]['Section'],
                    'similarity_score': float(similarities[idx])
                })
        
        response = {
            'results': results,
            'corrected_query': corrected_query if was_corrected else None,
            'is_vague': is_vague,
            'vague_suggestions': suggestions if is_vague else [],
            'expanded_terms': list(set(expanded_query.split()))  # Deduplicated list
        }
        
        # Update cache (maintain size limit)
        if len(self.query_cache) >= self.max_cache_size:
            # Remove oldest item (simple approach)
            self.query_cache.pop(next(iter(self.query_cache)))
        self.query_cache[cache_key] = response
        
        logger.info(f"Found {len(results)} results in {time.time() - start_time:.2f} seconds")
        return response

# Use lazy initialization to prevent blocking app startup
nic_processor = None

def get_processor():
    global nic_processor
    if nic_processor is None:
        try:
            logger.info("Initializing NIC processor")
            nic_processor = NICCodeProcessor('nic_2008.csv', 'industry_dictionary.json')
            logger.info("NIC processor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing NIC processor: {str(e)}")
            # Return None to indicate initialization failure
    return nic_processor

@app.route('/get_nic_codes', methods=['POST'])
def get_nic_codes():
    try:
        # Lazy initialization of processor
        processor = get_processor()
        if processor is None:
            return jsonify({
                'error': 'NIC Code Processor failed to initialize. Check server logs.',
                'results': []
            }), 500
            
        # Get the enhanced input (from Gemini)
        user_input = request.json['input']
        
        # Get the original input if provided
        original_input = request.json.get('original_input', user_input)
        
        logger.info(f"Received request for: {user_input}")
        logger.info(f"Original input was: {original_input}")
        
        # Find NIC codes with enhanced processing
        result_data = processor.find_nic_codes(user_input)
        
        # Store original input in the response
        return jsonify({
            'results': result_data['results'],
            'suggestions': result_data['expanded_terms'],
            'original_input': original_input,
            'enhanced_input': user_input if user_input != original_input else None,
            'corrected_input': result_data['corrected_query'],
            'is_vague': result_data['is_vague'],
            'vague_suggestions': result_data['vague_suggestions']
        })
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            'error': str(e),
            'results': []
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'processor_initialized': get_processor() is not None})

@app.route('/feedback', methods=['POST'])
def collect_feedback():
    """Collect user feedback on NIC code matches for future improvements"""
    try:
        feedback_data = request.json
        logger.info(f"Received feedback: {feedback_data}")
        
        # Here you would store this feedback for model improvement
        # This could update weights, add to a training set, etc.
        
        return jsonify({'status': 'feedback received'})
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application")
    # Pre-initialize processor in a background thread to avoid blocking startup
    import threading
    threading.Thread(target=get_processor, daemon=True).start()
    app.run(debug=True)
