import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import trafilatura
import requests
from io import BytesIO

# Optional imports for advanced features
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    st.warning("Whisper not available. Video transcription will be limited.")

class TextExtractor:
    def __init__(self):
        """Initialize text extraction components"""
        if WHISPER_AVAILABLE:
            try:
                # Try to load Whisper model
                self.whisper_model = whisper.load_model("base")
            except Exception as e:
                st.warning(f"Whisper model not available: {e}")
                self.whisper_model = None
        else:
            self.whisper_model = None
    
    def extract_from_url(self, url):
        """Extract text content from URL using trafilatura"""
        try:
            # Validate URL format
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Fetch and extract content
            downloaded = trafilatura.fetch_url(url)
            if downloaded is None:
                raise Exception("Failed to download content from URL")
            
            text = trafilatura.extract(downloaded)
            if not text:
                raise Exception("No text content found in the webpage")
            
            return text.strip()
            
        except Exception as e:
            raise Exception(f"URL extraction failed: {str(e)}")
    
    def extract_from_image(self, uploaded_file):
        """Extract text from image using OCR"""
        try:
            # Read image file
            image = Image.open(uploaded_file)
            
            # Convert PIL image to OpenCV format for preprocessing
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocess image for better OCR results
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get better contrast
            _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(threshold)
            
            # Extract text using pytesseract
            extracted_text = pytesseract.image_to_string(processed_image, lang='eng')
            
            if not extracted_text.strip():
                raise Exception("No text detected in the image")
            
            return extracted_text.strip()
            
        except Exception as e:
            raise Exception(f"Image OCR failed: {str(e)}")
    
    def extract_from_video(self, uploaded_file):
        """Extract text from video using audio transcription"""
        try:
            if not WHISPER_AVAILABLE or self.whisper_model is None:
                # Fallback message for video transcription
                return "Video transcription is currently not available. Please enable Whisper model or provide text input directly."
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name
            
            try:
                # Transcribe audio using Whisper
                result = self.whisper_model.transcribe(temp_path)
                transcribed_text = result["text"]
                
                if not transcribed_text.strip():
                    raise Exception("No speech detected in the video")
                
                return transcribed_text.strip()
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            raise Exception(f"Video transcription failed: {str(e)}")
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep punctuation for sentence structure
        import re
        text = re.sub(r'[^\w\s\.\!\?\,\;\:]', '', text)
        
        return text.strip()
