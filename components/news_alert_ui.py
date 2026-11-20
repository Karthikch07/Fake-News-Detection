import streamlit as st


def render_news_alert_header() -> None:
    """Plain masthead: title, one-line description, short red rule."""
    st.markdown(
        """
        <div class="masthead">
          <div class="kicker">Claim check</div>
          <h1>Fake News Detector</h1>
          <p>Paste a headline, article, link, screenshot or video clip. The text is scored by a classifier
          trained on about 45,000 labelled news articles, then checked against news sources where possible.</p>
        </div>
        <div class="ticker-wrap">
          <div class="ticker">
            <span>BREAKING:</span> Verify claims against credible sources •
            <span>ALERT:</span> Watch for sensational language &amp; missing citations •
            <span>TIP:</span> Use URL extraction for full context •
            <span>NOTE:</span> OCR requires Tesseract on Windows •
            <span>STATUS:</span> Uncertain verdicts mean: check the sources yourself •
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_status_block(*, title: str, lines: list[str]) -> None:
    st.sidebar.markdown(f"### {title}")
    for line in lines:
        st.sidebar.write(line)

