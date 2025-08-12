import streamlit as st
import os
from components.text_extractor import TextExtractor
from components.ml_predictor import MLPredictor
from components.fact_verifier import FactVerifier
from components.community_notes import CommunityNotesGenerator

# Configure page
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all system components"""
    return {
        'extractor': TextExtractor(),
        'predictor': MLPredictor(),
        'verifier': FactVerifier(),
        'notes_generator': CommunityNotesGenerator()
    }

def main():
    st.title("🔍 Fake News Detection System")
    st.markdown("### Confronting Misinformation - Detecting and Verifying Fake News")
    
    # Initialize components
    components = initialize_components()
    
    # Sidebar for input type selection
    st.sidebar.title("Input Options")
    input_type = st.sidebar.selectbox(
        "Select Input Type:",
        ["Text", "URL", "Image", "Video"]
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📝 {input_type} Input")
        
        extracted_text = None
        
        # Handle different input types
        if input_type == "Text":
            user_text = st.text_area(
                "Enter news text or headline:",
                height=150,
                placeholder="Enter the news content you want to verify..."
            )
            if user_text.strip():
                extracted_text = user_text.strip()
                
        elif input_type == "URL":
            url = st.text_input(
                "Enter news article URL:",
                placeholder="https://example.com/news-article"
            )
            if url.strip():
                with st.spinner("Extracting content from URL..."):
                    try:
                        extracted_text = components['extractor'].extract_from_url(url.strip())
                        if extracted_text:
                            st.success("✅ Content extracted successfully!")
                            with st.expander("View extracted content"):
                                st.text_area("Extracted text:", extracted_text, height=100, disabled=True)
                        else:
                            st.error("❌ Failed to extract content from URL")
                    except Exception as e:
                        st.error(f"❌ Error processing URL: {str(e)}")
                        
        elif input_type == "Image":
            uploaded_image = st.file_uploader(
                "Upload an image:",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                help="Upload a screenshot or image containing news text"
            )
            if uploaded_image is not None:
                st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
                with st.spinner("Extracting text from image..."):
                    try:
                        extracted_text = components['extractor'].extract_from_image(uploaded_image)
                        if extracted_text:
                            st.success("✅ Text extracted successfully!")
                            with st.expander("View extracted text"):
                                st.text_area("Extracted text:", extracted_text, height=100, disabled=True)
                        else:
                            st.error("❌ No text found in image")
                    except Exception as e:
                        st.error(f"❌ Error processing image: {str(e)}")
                        
        elif input_type == "Video":
            uploaded_video = st.file_uploader(
                "Upload a video:",
                type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
                help="Upload a news video clip for transcription"
            )
            if uploaded_video is not None:
                st.video(uploaded_video)
                with st.spinner("Transcribing audio from video..."):
                    try:
                        extracted_text = components['extractor'].extract_from_video(uploaded_video)
                        if extracted_text:
                            st.success("✅ Audio transcribed successfully!")
                            with st.expander("View transcription"):
                                st.text_area("Transcribed text:", extracted_text, height=100, disabled=True)
                        else:
                            st.error("❌ No audio transcription available")
                    except Exception as e:
                        st.error(f"❌ Error processing video: {str(e)}")
        
        # Process the extracted text
        if extracted_text:
            st.markdown("---")
            if st.button("🔍 Analyze News", type="primary", use_container_width=True):
                analyze_news(extracted_text, components)
    
    with col2:
        st.subheader("ℹ️ How it works")
        st.markdown("""
        **Step 1:** Choose your input type
        - Text: Direct text input
        - URL: Extract from news websites
        - Image: OCR text extraction
        - Video: Audio transcription
        
        **Step 2:** AI Analysis
        - Text preprocessing
        - Feature extraction
        - ML prediction
        
        **Step 3:** Fact Verification
        - Search credible sources
        - Compare claims
        - Generate Community Notes
        """)

def analyze_news(text, components):
    """Analyze the extracted text for fake news detection"""
    
    # Create results container
    results_container = st.container()
    
    with results_container:
        st.markdown("## 📊 Analysis Results")
        
        # Step 1: ML Prediction
        st.markdown("### 🤖 AI Prediction")
        with st.spinner("Analyzing content..."):
            try:
                prediction_result = components['predictor'].predict(text)
                
                # Display prediction
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction_result['prediction'] == 'Real':
                        st.success(f"**Prediction:** {prediction_result['prediction']}")
                    else:
                        st.error(f"**Prediction:** {prediction_result['prediction']}")
                
                with col2:
                    confidence = prediction_result['confidence']
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                with col3:
                    st.info(f"**Model:** {prediction_result['model_used']}")
                
                # Confidence visualization
                st.progress(confidence / 100.0)
                
            except Exception as e:
                st.error(f"❌ Error in prediction: {str(e)}")
                return
        
        # Step 2: Fact Verification
        st.markdown("### 🔍 Fact Verification")
        with st.spinner("Searching credible sources..."):
            try:
                verification_result = components['verifier'].verify_claims(text)
                
                if verification_result['sources']:
                    st.success(f"Found {len(verification_result['sources'])} credible sources")
                    
                    # Display sources
                    st.markdown("**Credible Sources Found:**")
                    for i, source in enumerate(verification_result['sources'], 1):
                        with st.expander(f"📰 Source {i}: {source['title']}"):
                            st.markdown(f"**URL:** {source['url']}")
                            st.markdown(f"**Summary:** {source['summary']}")
                            if source.get('relevance_score'):
                                st.progress(source['relevance_score'] / 100.0)
                                st.caption(f"Relevance: {source['relevance_score']:.1f}%")
                else:
                    st.warning("⚠️ No credible sources found for verification")
                    
            except Exception as e:
                st.error(f"❌ Error in fact verification: {str(e)}")
                verification_result = {'sources': []}
        
        # Step 3: Community Notes
        st.markdown("### 📝 Community Notes")
        with st.spinner("Generating contrast report..."):
            try:
                if verification_result['sources']:
                    community_notes = components['notes_generator'].generate_notes(
                        text, 
                        prediction_result, 
                        verification_result
                    )
                    
                    # Display Community Notes
                    if community_notes['contradictions']:
                        st.error("**❌ Contradictions Found:**")
                        for contradiction in community_notes['contradictions']:
                            st.markdown(f"• {contradiction}")
                    
                    if community_notes['confirmations']:
                        st.success("**✅ Confirmed Facts:**")
                        for confirmation in community_notes['confirmations']:
                            st.markdown(f"• {confirmation}")
                    
                    if community_notes['summary']:
                        st.info("**📋 Summary:**")
                        st.markdown(community_notes['summary'])
                    
                    # References
                    if community_notes['references']:
                        st.markdown("**🔗 References:**")
                        for ref in community_notes['references']:
                            st.markdown(f"- [{ref['source']}]({ref['url']})")
                else:
                    st.info("📝 Unable to generate Community Notes without credible sources.")
                    
            except Exception as e:
                st.error(f"❌ Error generating Community Notes: {str(e)}")

if __name__ == "__main__":
    main()
