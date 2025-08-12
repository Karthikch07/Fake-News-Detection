import requests
import os
import time
import streamlit as st
from typing import Dict, List, Optional

class APIClient:
    def __init__(self):
        """Initialize API client with rate limiting"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FakeNewsDetector/1.0 (Educational Purpose)'
        })
        
        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = 1.0  # Minimum seconds between requests
        
        # API configurations
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "")
        self.google_cx = os.getenv("GOOGLE_CX", "")
        self.bing_api_key = os.getenv("BING_API_KEY", "")
    
    def rate_limit_check(self, api_name: str):
        """Check rate limiting for API calls"""
        current_time = time.time()
        if api_name in self.last_request_time:
            time_diff = current_time - self.last_request_time[api_name]
            if time_diff < self.min_request_interval:
                sleep_time = self.min_request_interval - time_diff
                time.sleep(sleep_time)
        
        self.last_request_time[api_name] = current_time
    
    def make_request(self, url: str, params: Optional[Dict] = None, 
                    headers: Optional[Dict] = None, timeout: int = 10) -> Optional[Dict]:
        """Make HTTP request with error handling"""
        try:
            self.rate_limit_check("general")
            
            request_headers = self.session.headers.copy()
            if headers:
                request_headers.update(headers)
            
            response = self.session.get(
                url, 
                params=params, 
                headers=request_headers, 
                timeout=timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again.")
            return None
        except requests.exceptions.ConnectionError:
            st.error("Connection error. Please check your internet connection.")
            return None
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP error: {e}")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Request error: {e}")
            return None
        except ValueError as e:
            st.error(f"Invalid JSON response: {e}")
            return None
    
    def test_apis(self) -> Dict[str, bool]:
        """Test availability of configured APIs"""
        api_status = {
            'google_search': False,
            'bing_search': False
        }
        
        # Test Google Search API
        if self.google_api_key and self.google_cx:
            test_url = "https://www.googleapis.com/customsearch/v1"
            test_params = {
                'key': self.google_api_key,
                'cx': self.google_cx,
                'q': 'test',
                'num': 1
            }
            
            response = self.make_request(test_url, test_params)
            if response and 'items' in response:
                api_status['google_search'] = True
        
        # Test Bing Search API
        if self.bing_api_key:
            test_url = "https://api.bing.microsoft.com/v7.0/search"
            test_headers = {'Ocp-Apim-Subscription-Key': self.bing_api_key}
            test_params = {'q': 'test', 'count': 1}
            
            response = self.make_request(test_url, test_params, test_headers)
            if response and 'webPages' in response:
                api_status['bing_search'] = True
        
        return api_status
    
    def get_api_status_message(self) -> str:
        """Get human-readable API status message"""
        status = self.test_apis()
        
        messages = []
        if status['google_search']:
            messages.append("✅ Google Search API: Available")
        else:
            messages.append("❌ Google Search API: Not configured")
        
        if status['bing_search']:
            messages.append("✅ Bing Search API: Available")
        else:
            messages.append("❌ Bing Search API: Not configured")
        
        if not any(status.values()):
            messages.append("⚠️ No search APIs configured. Using fallback methods.")
        
        return "\n".join(messages)
    
    def search_web(self, query: str, num_results: int = 5) -> List[Dict]:
        """Generic web search function"""
        results = []
        
        # Try Google first
        if self.google_api_key and self.google_cx:
            google_results = self.google_search(query, num_results)
            results.extend(google_results)
        
        # Try Bing if Google didn't return enough results
        if len(results) < num_results and self.bing_api_key:
            remaining = num_results - len(results)
            bing_results = self.bing_search(query, remaining)
            results.extend(bing_results)
        
        return results[:num_results]
    
    def google_search(self, query: str, num_results: int = 5) -> List[Dict]:
        """Perform Google search"""
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': self.google_api_key,
            'cx': self.google_cx,
            'q': query,
            'num': min(num_results, 10),  # Google API max is 10
            'safe': 'active'
        }
        
        response = self.make_request(url, params)
        results = []
        
        if response and 'items' in response:
            for item in response['items']:
                result = {
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'source': 'google'
                }
                results.append(result)
        
        return results
    
    def bing_search(self, query: str, num_results: int = 5) -> List[Dict]:
        """Perform Bing search"""
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {'Ocp-Apim-Subscription-Key': self.bing_api_key}
        params = {
            'q': query,
            'count': min(num_results, 50),  # Bing API max is 50
            'safeSearch': 'Moderate'
        }
        
        response = self.make_request(url, params, headers)
        results = []
        
        if response and 'webPages' in response and 'value' in response['webPages']:
            for item in response['webPages']['value']:
                result = {
                    'title': item.get('name', ''),
                    'url': item.get('url', ''),
                    'snippet': item.get('snippet', ''),
                    'source': 'bing'
                }
                results.append(result)
        
        return results
    
    def check_url_accessibility(self, url: str) -> bool:
        """Check if a URL is accessible"""
        try:
            response = self.session.head(url, timeout=5, allow_redirects=True)
            return response.status_code == 200
        except:
            return False
    
    def get_url_content(self, url: str) -> Optional[str]:
        """Get content from URL"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            st.warning(f"Failed to get content from {url}: {str(e)}")
            return None
