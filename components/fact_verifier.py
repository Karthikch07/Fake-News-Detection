import streamlit as st
import requests
import json
import os
from urllib.parse import quote
import trafilatura
import numpy as np

# Optional imports for advanced features
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class FactVerifier:
    def __init__(self):
        """Initialize fact verification component"""
        # Try to load sentence transformer for similarity matching
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                st.warning(f"Sentence transformer not available: {e}")
                self.sentence_model = None
        else:
            self.sentence_model = None
        
        # Trusted news domains for verification
        self.trusted_domains = [
            'reuters.com', 'bbc.com', 'cnn.com', 'npr.org', 'apnews.com',
            'nytimes.com', 'washingtonpost.com', 'theguardian.com',
            'abcnews.go.com', 'cbsnews.com', 'nbcnews.com', 'foxnews.com',
            'usatoday.com', 'wsj.com', 'bloomberg.com', 'economist.com'
        ]
        
        # API keys from environment variables
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "")
        self.google_cx = os.getenv("GOOGLE_CX", "")
        self.bing_api_key = os.getenv("BING_API_KEY", "")
        self._fallback_notice_shown = False
    
    def verify_claims(self, text):
        """Verify claims by searching credible sources"""
        try:
            # Extract key claims/entities from text
            search_queries = self.extract_search_queries(text)

            has_google = bool(self.google_api_key and self.google_cx)
            has_bing = bool(self.bing_api_key)

            # If no external API is configured, return a deterministic offline fallback.
            if not has_google and not has_bing:
                sources = self.fallback_search(search_queries)
                ranked_sources = self.rank_sources(text, self.deduplicate_sources(sources))
                return {
                    'sources': ranked_sources[:5],
                    'search_queries': search_queries
                }

            sources = []
            for query in search_queries:
                # Try Google Search first, then Bing if available
                if has_google:
                    google_results = self.search_google(query)
                    sources.extend(google_results)
                elif has_bing:
                    bing_results = self.search_bing(query)
                    sources.extend(bing_results)
            
            # Remove duplicates and rank by relevance
            unique_sources = self.deduplicate_sources(sources)
            ranked_sources = self.rank_sources(text, unique_sources)
            
            return {
                'sources': ranked_sources[:5],  # Return top 5 sources
                'search_queries': search_queries
            }
            
        except Exception as e:
            st.error(f"Error in fact verification: {str(e)}")
            return {'sources': [], 'search_queries': []}
    
    def extract_search_queries(self, text):
        """Extract relevant search queries from the news text"""
        # Simple approach: split text into chunks and create search queries
        import re
        
        # Remove common stop words and extract potential entities/claims
        words = text.split()
        
        # Look for potential entities (capitalized words, numbers, dates)
        entities = []
        for word in words:
            word = re.sub(r'[^\w\s]', '', word)
            if word and (word[0].isupper() or word.isdigit()):
                entities.append(word)
        
        # Create search queries
        queries = []
        
        # Use the first 50 words as main query
        main_query = ' '.join(words[:50])
        queries.append(main_query)
        
        # Create entity-based queries
        if entities:
            entity_query = ' '.join(entities[:10])
            queries.append(entity_query)
        
        # Look for specific claims or statements
        sentences = text.split('.')
        for sentence in sentences[:3]:  # First 3 sentences
            sentence = sentence.strip()
            if len(sentence.split()) > 5:  # Meaningful sentences
                queries.append(sentence)
        
        return queries[:3]  # Return top 3 queries
    
    def search_google(self, query):
        """Search using Google Custom Search API"""
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.google_api_key,
                'cx': self.google_cx,
                'q': query,
                'num': 5,
                'dateRestrict': 'y1',  # Last year
                'safe': 'active'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            sources = []
            
            if 'items' in data:
                for item in data['items']:
                    url = item.get('link', '')
                    domain = self.extract_domain(url)
                    
                    if any(trusted in domain for trusted in self.trusted_domains):
                        source = {
                            'title': item.get('title', ''),
                            'url': url,
                            'summary': item.get('snippet', ''),
                            'domain': domain,
                            'source_type': 'google_search'
                        }
                        sources.append(source)
            
            return sources
            
        except Exception as e:
            st.warning(f"Google search failed: {str(e)}")
            return []
    
    def search_bing(self, query):
        """Search using Bing Search API"""
        try:
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {'Ocp-Apim-Subscription-Key': self.bing_api_key}
            params = {
                'q': query,
                'count': 5,
                'freshness': 'Month',
                'safeSearch': 'Moderate'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            sources = []
            
            if 'webPages' in data and 'value' in data['webPages']:
                for item in data['webPages']['value']:
                    url = item.get('url', '')
                    domain = self.extract_domain(url)
                    
                    if any(trusted in domain for trusted in self.trusted_domains):
                        source = {
                            'title': item.get('name', ''),
                            'url': url,
                            'summary': item.get('snippet', ''),
                            'domain': domain,
                            'source_type': 'bing_search'
                        }
                        sources.append(source)
            
            return sources
            
        except Exception as e:
            st.warning(f"Bing search failed: {str(e)}")
            return []
    
    def fallback_search(self, queries):
        """Offline fallback when APIs are not available."""
        if isinstance(queries, str):
            query_list = [queries]
        else:
            query_list = [q for q in (queries or []) if q and q.strip()]

        if not query_list:
            query_list = ["submitted claim"]

        fallback_sources = []
        source_catalog = [
            "reuters.com",
            "apnews.com",
            "bbc.com",
            "npr.org",
            "theguardian.com",
        ]

        for i, query in enumerate(query_list[:3]):
            domain = source_catalog[i % len(source_catalog)]
            # Use a stable, public search URL filtered to each trusted domain.
            search_query = f"site:{domain} {query[:80]}"
            encoded_query = quote(search_query)
            fallback_sources.append(
                {
                    'title': f"Offline verification note: {query[:70]}",
                    'url': f"https://duckduckgo.com/?q={encoded_query}",
                    'summary': (
                        "API keys are not configured, so this result was generated using "
                        "offline fallback guidance for further manual verification."
                    ),
                    'domain': domain,
                    'source_type': 'offline_fallback'
                }
            )

        return fallback_sources
    
    def extract_domain(self, url):
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except:
            return ""
    
    def deduplicate_sources(self, sources):
        """Remove duplicate sources"""
        seen_urls = set()
        unique_sources = []
        
        for source in sources:
            url = source.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append(source)
        
        return unique_sources
    
    def rank_sources(self, original_text, sources):
        """Rank sources by relevance to original text"""
        if not sources or not self.sentence_model:
            # Simple ranking by domain trust if no sentence model
            domain_scores = {}
            for domain in self.trusted_domains:
                domain_scores[domain] = len(self.trusted_domains) - self.trusted_domains.index(domain)
            
            for source in sources:
                domain = source.get('domain', '')
                score = 50  # Default score
                for trusted_domain in domain_scores:
                    if trusted_domain in domain:
                        score = domain_scores[trusted_domain] * 10
                        break
                source['relevance_score'] = score
            
            return sorted(sources, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        try:
            # Use sentence similarity for ranking
            original_embedding = self.sentence_model.encode([original_text])
            
            for source in sources:
                source_text = f"{source.get('title', '')} {source.get('summary', '')}"
                source_embedding = self.sentence_model.encode([source_text])
                
                # Calculate cosine similarity
                similarity = cosine_similarity(original_embedding, source_embedding)[0][0]
                source['relevance_score'] = float(similarity * 100)
            
            # Sort by relevance score
            return sorted(sources, key=lambda x: x.get('relevance_score', 0), reverse=True)
            
        except Exception as e:
            st.warning(f"Error in ranking sources: {str(e)}")
            return sources
    
    def extract_source_content(self, url):
        """Extract full content from source URL"""
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                return trafilatura.extract(downloaded)
            return None
        except Exception as e:
            st.warning(f"Failed to extract content from {url}: {str(e)}")
            return None
