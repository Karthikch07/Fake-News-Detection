import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.text_processor import TextProcessor
import pickle
import os
import joblib

class MLPredictor:
    def __init__(self):
        """Initialize ML predictor with trained model"""
        self.text_processor = TextProcessor()
        self.vectorizer = None
        self.model = None
        self.model_loaded = False
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize or load the ML model"""
        try:
            # Try to load pre-trained model
            self.load_model()
        except Exception as e:
            st.warning("Pre-trained model not found. Creating baseline model...")
            self.create_baseline_model()
    
    def create_baseline_model(self):
        """Create a baseline model with synthetic training data"""
        # Create synthetic training data for demonstration
        # In production, this would use a real labeled dataset
        fake_news_patterns = [
            "BREAKING: Scientists discover aliens",
            "You won't believe what happened next",
            "This one weird trick doctors hate",
            "SHOCKING revelation about celebrities",
            "Government doesn't want you to know this",
            "Miracle cure discovered by local person",
            "Billionaire's secret method revealed",
            "What they don't tell you about",
            "Explosive evidence uncovered",
            "Insider reveals shocking truth"
        ]
        
        real_news_patterns = [
            "According to recent studies published in",
            "Government officials announced today",
            "Research from university shows",
            "Economic indicators suggest that",
            "Medical professionals recommend",
            "Data from national statistics indicates",
            "Official spokesperson confirmed",
            "Peer-reviewed study published in",
            "Analysis of market trends reveals",
            "Expert panel concludes that"
        ]
        
        # Create training dataset
        X_train = fake_news_patterns + real_news_patterns
        y_train = ['Fake'] * len(fake_news_patterns) + ['Real'] * len(real_news_patterns)
        
        # Process text
        X_processed = [self.text_processor.preprocess_text(text) for text in X_train]
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        # Fit and transform text
        X_vectorized = self.vectorizer.fit_transform(X_processed)
        
        # Train logistic regression model
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_vectorized, y_train)
        
        self.model_loaded = True
        st.success("✅ Baseline model created successfully!")
    
    def load_model(self):
        """Load pre-trained model from file"""
        model_path = 'models/fake_news_model.pkl'
        vectorizer_path = 'models/tfidf_vectorizer.pkl'
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.model_loaded = True
            st.success("✅ Pre-trained model loaded successfully!")
        else:
            raise FileNotFoundError("Model files not found")
    
    def save_model(self):
        """Save trained model to file"""
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/fake_news_model.pkl')
        joblib.dump(self.vectorizer, 'models/tfidf_vectorizer.pkl')
    
    def predict(self, text):
        """Predict if news is fake or real"""
        if not self.model_loaded:
            raise Exception("Model not loaded. Cannot make predictions.")
        
        try:
            # Preprocess text
            processed_text = self.text_processor.preprocess_text(text)
            
            if not processed_text.strip():
                raise Exception("No valid text content for prediction")
            
            # Vectorize text
            text_vectorized = self.vectorizer.transform([processed_text])
            
            # Make prediction
            prediction = self.model.predict(text_vectorized)[0]
            prediction_proba = self.model.predict_proba(text_vectorized)[0]
            
            # Get confidence score
            confidence = max(prediction_proba) * 100
            
            # Get feature importance for explanation
            feature_names = self.vectorizer.get_feature_names_out()
            feature_scores = self.model.coef_[0]
            
            # Get top features that influenced the decision
            text_features = self.vectorizer.transform([processed_text]).toarray()[0]
            top_features = []
            
            for i, score in enumerate(text_features):
                if score > 0:
                    feature_importance = abs(feature_scores[i] * score)
                    top_features.append((feature_names[i], feature_importance))
            
            top_features = sorted(top_features, key=lambda x: x[1], reverse=True)[:5]
            
            result = {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': {
                    'Fake': prediction_proba[0] * 100,
                    'Real': prediction_proba[1] * 100
                },
                'top_features': top_features,
                'model_used': 'Logistic Regression + TF-IDF'
            }
            
            return result
            
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")
    
    def get_prediction_explanation(self, prediction_result):
        """Generate explanation for the prediction"""
        explanation = []
        
        explanation.append(f"The model predicts this news is **{prediction_result['prediction']}** with {prediction_result['confidence']:.1f}% confidence.")
        
        if prediction_result['top_features']:
            explanation.append("\n**Key factors in this decision:**")
            for feature, importance in prediction_result['top_features']:
                explanation.append(f"• '{feature}' (importance: {importance:.3f})")
        
        return "\n".join(explanation)
