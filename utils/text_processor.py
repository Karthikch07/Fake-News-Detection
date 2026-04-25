import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import streamlit as st

class TextProcessor:
    def __init__(self):
        """Initialize text processor with NLTK components"""
        self.setup_nltk()
        self.lemmatizer = WordNetLemmatizer()
        
    def setup_nltk(self):
        """Download required NLTK data"""
        try:
            # Try to download NLTK data silently
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            nltk_downloads = [
                'punkt', 'stopwords', 'wordnet', 'omw-1.4', 
                'averaged_perceptron_tagger'
            ]
            
            for item in nltk_downloads:
                try:
                    nltk.download(item, quiet=True)
                except Exception as e:
                    st.warning(f"Could not download NLTK data '{item}': {e}")
                    
        except Exception as e:
            st.warning(f"NLTK setup warning: {e}")
    
    def preprocess_text(self, text):
        """Minimal text preprocessing for ML—keep most content intact."""
        if not text or not isinstance(text, str):
            return ""
        
        # Minimal preprocessing: lowercase + simple whitespace cleanup
        # Let TF-IDF handle tokenization/stemming
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text if text else ""
    
    def basic_cleaning(self, text):
        """Basic text cleaning operations"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation for sentence structure
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', '', text)
        
        return text.strip()
    
    def tokenize(self, text):
        """Tokenize text into words"""
        try:
            tokens = word_tokenize(text)
            return [token for token in tokens if token.isalpha()]  # Keep only alphabetic tokens
        except Exception:
            # Fallback to simple split if NLTK tokenizer fails
            words = text.split()
            return [re.sub(r'[^\w]', '', word) for word in words if re.sub(r'[^\w]', '', word)]
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from tokens"""
        try:
            stop_words = set(stopwords.words('english'))
            return [token for token in tokens if token not in stop_words and len(token) > 1]
        except Exception:
            # Fallback stopwords list
            basic_stopwords = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'over', 'under', 'again', 'further',
                'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'is', 'are',
                'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did'
            }
            return [token for token in tokens if token not in basic_stopwords and len(token) > 1]
    
    def lemmatize_tokens(self, tokens):
        """Lemmatize tokens to their root form"""
        try:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        except Exception:
            # Return tokens as-is if lemmatization fails
            return tokens
    
    def extract_sentences(self, text):
        """Extract sentences from text"""
        try:
            return sent_tokenize(text)
        except Exception:
            # Fallback to simple split
            return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    def extract_keywords(self, text, top_n=10):
        """Extract key terms from text"""
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        # Simple frequency-based keyword extraction
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Only consider words longer than 3 characters
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, freq in sorted_words[:top_n]]
    
    def clean_for_display(self, text, max_length=500):
        """Clean text for display purposes"""
        if not text:
            return ""
        
        # Basic cleaning without aggressive preprocessing
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text
