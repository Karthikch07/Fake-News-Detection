# Fake News Detection System

## Overview

A comprehensive Streamlit-based web application for detecting and verifying fake news across multiple media types. The system combines machine learning prediction, fact verification through credible sources, and community notes generation to provide users with comprehensive analysis of news content. It supports text, URL, image (OCR), and video (speech-to-text) inputs, making it a versatile tool for combating misinformation.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Framework**: Web-based interface with multi-column layouts and sidebar navigation
- **Input Flexibility**: Supports four input types - direct text, URL extraction, image OCR, and video transcription
- **Caching Strategy**: Uses `@st.cache_resource` for component initialization to improve performance
- **Progressive Disclosure**: Two-column layout separating input processing from results display

### Backend Architecture
- **Modular Component Design**: Four main processing components with clear separation of concerns
  - `TextExtractor`: Handles content extraction from various media types
  - `MLPredictor`: Manages machine learning-based fake news classification
  - `FactVerifier`: Performs cross-referencing with credible news sources
  - `CommunityNotesGenerator`: Creates detailed analysis reports with contradictions and confirmations
- **Pipeline Processing**: Sequential processing through extraction → prediction → verification → notes generation
- **Utility Layer**: Shared components for text processing and API communication

### Machine Learning Components
- **Text Classification**: TF-IDF vectorization with Logistic Regression for baseline fake news detection
- **Semantic Analysis**: Sentence transformers (all-MiniLM-L6-v2) for similarity matching and claim verification
- **Fallback Mechanisms**: Synthetic training data generation when pre-trained models are unavailable
- **Model Persistence**: Pickle/joblib serialization for trained model storage and loading

### Media Processing Pipeline
- **OCR Integration**: Tesseract with OpenCV preprocessing for image text extraction
- **Speech Recognition**: Whisper model integration for video/audio transcription
- **Web Scraping**: Trafilatura for clean text extraction from news URLs
- **Content Validation**: Multi-layer validation for extracted content quality

### Fact Verification System
- **Multi-Source Strategy**: Integration with Google Custom Search and Bing Search APIs
- **Trusted Domain Filtering**: Curated list of 16+ credible news sources for verification
- **Semantic Matching**: Cosine similarity comparison between claims and source content
- **Evidence Aggregation**: Contradiction and confirmation tracking across multiple sources

## External Dependencies

### Core ML/NLP Libraries
- **Streamlit**: Web application framework and UI components
- **Sentence Transformers**: Semantic similarity analysis and embedding generation
- **Scikit-learn**: Machine learning algorithms and text vectorization
- **NLTK**: Natural language processing, tokenization, and text preprocessing
- **Whisper**: OpenAI's speech recognition model for audio transcription

### Media Processing
- **Tesseract/Pytesseract**: Optical Character Recognition for image text extraction
- **OpenCV (cv2)**: Image preprocessing and computer vision operations
- **PIL (Pillow)**: Image handling and format conversion
- **Trafilatura**: Web content extraction and HTML parsing

### Search APIs
- **Google Custom Search API**: Primary source for credible news verification
- **Bing Search API**: Fallback search engine for fact verification
- **Environment Variables**: `GOOGLE_API_KEY`, `GOOGLE_CX`, `BING_API_KEY` for API authentication

### Utility Libraries
- **Requests**: HTTP client for API calls and web scraping
- **NumPy**: Numerical computing for similarity calculations
- **Pandas**: Data manipulation for ML model training
- **Joblib/Pickle**: Model serialization and caching

### Rate Limiting and Performance
- **Built-in Rate Limiting**: Custom rate limiting for API calls to prevent quota exhaustion
- **Session Management**: Persistent HTTP sessions with proper headers
- **Error Handling**: Comprehensive exception handling with graceful degradation
- **Resource Caching**: Streamlit caching for expensive model loading operations