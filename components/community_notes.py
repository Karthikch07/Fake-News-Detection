import streamlit as st
import numpy as np
import re

# Optional imports for advanced features
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    st.warning("Sentence transformers not available for Community Notes. Using fallback methods.")

class CommunityNotesGenerator:
    def __init__(self):
        """Initialize Community Notes generator"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                st.warning(f"Sentence transformer not available for Community Notes: {e}")
                self.sentence_model = None
        else:
            self.sentence_model = None
    
    def generate_notes(self, original_text, prediction_result, verification_result):
        """Generate Community Notes-style contrast report"""
        try:
            # Extract claims from original text
            original_claims = self.extract_claims(original_text)
            
            # Compare with sources
            contradictions = []
            confirmations = []
            references = []
            
            for source in verification_result.get('sources', []):
                source_analysis = self.analyze_source_against_claims(
                    original_claims, source
                )
                
                if source_analysis['contradictions']:
                    contradictions.extend(source_analysis['contradictions'])
                
                if source_analysis['confirmations']:
                    confirmations.extend(source_analysis['confirmations'])
                
                # Add reference
                references.append({
                    'source': source['domain'],
                    'url': source['url'],
                    'title': source['title']
                })
            
            # Generate summary based on prediction and sources
            summary = self.generate_summary(
                original_text, 
                prediction_result, 
                contradictions, 
                confirmations
            )
            
            return {
                'contradictions': list(set(contradictions)),  # Remove duplicates
                'confirmations': list(set(confirmations)),
                'summary': summary,
                'references': references,
                'original_claims': original_claims
            }
            
        except Exception as e:
            st.error(f"Error generating Community Notes: {str(e)}")
            return {
                'contradictions': [],
                'confirmations': [],
                'summary': "Unable to generate Community Notes due to an error.",
                'references': [],
                'original_claims': []
            }
    
    def extract_claims(self, text):
        """Extract key claims from the original text"""
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) > 5:  # Meaningful sentences
                # Look for factual claims (contains numbers, dates, names)
                if (any(word.isdigit() for word in sentence.split()) or 
                    any(word[0].isupper() for word in sentence.split() if word) or
                    any(keyword in sentence.lower() for keyword in [
                        'said', 'reported', 'announced', 'confirmed', 'revealed',
                        'according', 'study', 'research', 'data', 'percent'
                    ])):
                    claims.append(sentence)
        
        return claims[:5]  # Return top 5 claims
    
    def analyze_source_against_claims(self, claims, source):
        """Analyze a source against original claims"""
        source_text = f"{source.get('title', '')} {source.get('summary', '')}"
        contradictions = []
        confirmations = []
        
        if not self.sentence_model:
            # Simple keyword-based analysis when sentence model is not available
            return self.simple_claim_analysis(claims, source_text, source)
        
        try:
            # Use semantic similarity for claim analysis
            source_embedding = self.sentence_model.encode([source_text])
            
            for claim in claims:
                claim_embedding = self.sentence_model.encode([claim])
                similarity = cosine_similarity(claim_embedding, source_embedding)[0][0]
                
                # High similarity suggests confirmation or contradiction
                if similarity > 0.6:
                    # Check for contradiction indicators
                    if self.check_contradiction_indicators(claim, source_text):
                        contradictions.append(f"Source contradicts: '{claim[:100]}...'")
                    else:
                        confirmations.append(f"Source supports: '{claim[:100]}...'")
        
        except Exception as e:
            st.warning(f"Error in semantic analysis: {str(e)}")
            return self.simple_claim_analysis(claims, source_text, source)
        
        return {
            'contradictions': contradictions,
            'confirmations': confirmations
        }
    
    def simple_claim_analysis(self, claims, source_text, source):
        """Simple keyword-based claim analysis fallback"""
        contradictions = []
        confirmations = []
        
        contradiction_keywords = [
            'denies', 'refutes', 'false', 'incorrect', 'untrue', 'debunked',
            'misleading', 'no evidence', 'not supported', 'contradicts'
        ]
        
        confirmation_keywords = [
            'confirms', 'supports', 'verifies', 'true', 'correct', 'accurate',
            'evidence shows', 'data indicates', 'research suggests'
        ]
        
        source_lower = source_text.lower()
        
        for claim in claims:
            claim_words = set(claim.lower().split())
            source_words = set(source_lower.split())
            
            # Check for word overlap
            overlap = len(claim_words.intersection(source_words))
            if overlap > 2:  # Some overlap suggests relevance
                
                # Check for contradiction indicators
                if any(keyword in source_lower for keyword in contradiction_keywords):
                    contradictions.append(
                        f"'{source['domain']}' appears to contradict the claim: '{claim[:100]}...'"
                    )
                
                # Check for confirmation indicators
                elif any(keyword in source_lower for keyword in confirmation_keywords):
                    confirmations.append(
                        f"'{source['domain']}' appears to support the claim: '{claim[:100]}...'"
                    )
        
        return {
            'contradictions': contradictions,
            'confirmations': confirmations
        }
    
    def check_contradiction_indicators(self, claim, source_text):
        """Check if source text contradicts the claim"""
        contradiction_patterns = [
            'no evidence', 'false', 'incorrect', 'untrue', 'denies',
            'refutes', 'contradicts', 'disputes', 'debunked', 'misleading'
        ]
        
        source_lower = source_text.lower()
        return any(pattern in source_lower for pattern in contradiction_patterns)
    
    def generate_summary(self, original_text, prediction_result, contradictions, confirmations):
        """Generate a summary for the Community Notes"""
        prediction = prediction_result['prediction']
        confidence = prediction_result['confidence']
        
        summary_parts = []
        
        # Start with prediction
        summary_parts.append(
            f"AI analysis suggests this content is **{prediction}** "
            f"({confidence:.0f}% confidence)."
        )
        
        # Add contradiction information
        if contradictions:
            summary_parts.append(
                f"Multiple credible sources contradict key claims in this content. "
                f"({len(contradictions)} contradiction(s) found)"
            )
        
        # Add confirmation information
        if confirmations:
            summary_parts.append(
                f"Some claims appear to be supported by credible sources. "
                f"({len(confirmations)} confirmation(s) found)"
            )
        
        # Overall assessment
        if contradictions and not confirmations:
            summary_parts.append(
                "**Community Note:** This content appears to contain misinformation "
                "based on fact-checking against credible sources."
            )
        elif confirmations and not contradictions:
            summary_parts.append(
                "**Community Note:** This content appears to be largely supported "
                "by credible sources."
            )
        elif contradictions and confirmations:
            summary_parts.append(
                "**Community Note:** This content contains mixed information - "
                "some claims are supported while others are contradicted by credible sources."
            )
        else:
            summary_parts.append(
                "**Community Note:** Unable to fully verify this content against "
                "available credible sources. Further verification recommended."
            )
        
        return " ".join(summary_parts)
