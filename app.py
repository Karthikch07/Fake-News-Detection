import streamlit as st
from dotenv import load_dotenv
from components.text_extractor import TextExtractor
from components.ml_predictor import MLPredictor
from components.fact_verifier import FactVerifier
from components.community_notes import CommunityNotesGenerator
from components.ui_3d import inject_glass_css
from components.news_alert_ui import render_news_alert_header
from components.sidebar_console import render_console, pop_actions

# Load environment variables from a local .env file when present.
load_dotenv(override=False)

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


def _init_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "active_result" not in st.session_state:
        st.session_state.active_result = None
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = "SVM (TF-IDF)"


def main():
    _init_state()
    inject_glass_css()
    
    # Initialize components
    components = initialize_components()

    render_news_alert_header()

    st.markdown("---")

    console = render_console(
        extractor=components["extractor"],
        predictor=components["predictor"],
        verifier=components["verifier"],
        history=st.session_state.history,
    )
    actions = pop_actions()

    # Immediate analyze
    if actions["analyze_now"] and console.input_result.extracted_text and not console.input_result.errors:
        st.session_state.active_result = analyze_news(
            console.input_result.extracted_text,
            components,
            input_type=console.input_type,
            model_choice=console.model_choice,
        )
        st.session_state.history = [st.session_state.active_result] + st.session_state.history[:25]

    # Batch run
    if actions["run_queue"] and st.session_state.queue:
        progress = st.progress(0.0)
        status = st.empty()
        total = len(st.session_state.queue)
        results: list[dict] = []

        for i, item in enumerate(list(st.session_state.queue), 1):
            status.markdown(f"**Batch running** {i}/{total}: `{item.get('input_type')}` • `{item.get('model_choice')}`")
            try:
                r = analyze_news(
                    item.get("text", ""),
                    components,
                    input_type=item.get("input_type", "Text"),
                    model_choice=item.get("model_choice", st.session_state.model_choice),
                )
                results.append(r)
                st.session_state.active_result = r
                st.session_state.history = [r] + st.session_state.history
                if r.get("errors") and st.session_state.batch_stop_on_error:
                    break
            except Exception as e:
                results.append({"errors": [f"Batch item failed: {e}"], "input_type": item.get("input_type", "Text")})
                if st.session_state.batch_stop_on_error:
                    break

            progress.progress(i / total)

        # Remove items that were run (keep any remaining if we stopped early)
        st.session_state.queue = st.session_state.queue[len(results) :]
        status.success(f"Batch complete. Ran {len(results)} item(s).")
        progress.progress(1.0)

    # Quick input area for users who skip sidebar
    col1, col2 = st.columns([3, 1])
    with col1:
        quick_input = st.text_area(
            "Quick analysis - paste text or headline here:",
            placeholder="Enter text to analyze...",
            height=120,
            key="quick_input_main"
        )
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        quick_analyze_btn = st.button(
            "🔍 Analyze",
            type="primary",
            use_container_width=True,
            key="quick_analyze_btn"
        )

    # Handle direct main-page analysis
    if quick_analyze_btn and quick_input and quick_input.strip():
        st.session_state.active_result = analyze_news(
            quick_input.strip(),
            components,
            input_type="Text",
            model_choice=st.session_state.model_choice,
        )
        st.session_state.history = [st.session_state.active_result] + st.session_state.history[:25]
        st.rerun()

    st.markdown("### Results")

    if st.session_state.active_result:
        render_results(st.session_state.active_result)
    else:
        st.markdown(
            """
            <div class="muted">
              <b>How it works</b><br/>
              1) Extract text from your chosen input (main area above or sidebar)<br/>
              2) Run a classifier (SVM/BERT/LSTM/GAN depending on availability)<br/>
              3) Search credible sources (API-backed if configured)<br/>
              4) Generate Community Notes-style contradictions/confirmations
            </div>
            """,
            unsafe_allow_html=True,
        )

def analyze_news(text: str, components: dict, *, input_type: str, model_choice: str) -> dict:
    """Run the full pipeline and return a serializable result."""
    import datetime as _dt

    result: dict = {
        "timestamp": _dt.datetime.now().strftime("%H:%M:%S"),
        "input_type": input_type,
        "model_choice": model_choice,
        "snippet": (" ".join(text.split())[:72] + ("…" if len(text) > 72 else "")),
        "text": text,
        "prediction": None,
        "verification": None,
        "notes": None,
        "errors": [],
    }

    try:
        with st.spinner("Predicting (ML)…"):
            result["prediction"] = components["predictor"].predict(text, model_choice=model_choice)
    except Exception as e:
        result["errors"].append(f"Prediction error: {e}")
        return result

    try:
        with st.spinner("Verifying (credible sources)…"):
            result["verification"] = components["verifier"].verify_claims(text)
    except Exception as e:
        result["errors"].append(f"Verification error: {e}")
        result["verification"] = {"sources": [], "search_queries": []}

    try:
        with st.spinner("Generating Community Notes…"):
            if result["verification"] and result["verification"].get("sources"):
                result["notes"] = components["notes_generator"].generate_notes(
                    text,
                    result["prediction"],
                    result["verification"],
                )
            else:
                result["notes"] = {
                    "contradictions": [],
                    "confirmations": [],
                    "summary": "No credible sources were found/configured, so Community Notes could not be fully generated.",
                    "references": [],
                    "original_claims": [],
                }
    except Exception as e:
        result["errors"].append(f"Community Notes error: {e}")

    return result


def render_results(result: dict) -> None:
    def _normalize_source_url(source: dict) -> str:
        """Return a robust URL for source links, including legacy offline entries."""
        raw_url = str(source.get("url", "")).strip()
        if raw_url and not raw_url.startswith(("http://", "https://")):
            raw_url = f"https://{raw_url.lstrip('/')}"

        # Backfill older cached offline URLs to stable search links.
        if source.get("source_type") == "offline_fallback":
            summary = str(source.get("summary", ""))
            legacy_offline = (
                "offline fallback guidance" in summary.lower()
                and (
                    "reuters.com/site-search" in raw_url
                    or "bbc.co.uk/search" in raw_url
                    or "apnews.com/search" in raw_url
                )
            )
            if legacy_offline:
                from urllib.parse import quote

                domain = str(source.get("domain", "")).strip() or "reuters.com"
                title = str(source.get("title", "")).replace("Offline verification note:", "").strip()
                search_query = f"site:{domain} {title[:80]}"
                return f"https://duckduckgo.com/?q={quote(search_query)}"

        return raw_url

    # Only show critical errors; hide routine prediction fallback messages
    errors = result.get("errors") or []
    critical_errors = [e for e in errors if "Community Notes" in e or "model not loaded" in e.lower()]
    
    if critical_errors:
        with st.expander("⚠️ Notes", expanded=False):
            for e in critical_errors:
                st.caption(e)

    pred = result.get("prediction") or {}
    ver = result.get("verification") or {"sources": []}
    notes = result.get("notes") or {}

    st.caption(f"Run: {result.get('timestamp','')} • Input: {result.get('input_type','')} • Model: {result.get('model_choice','')}")

    # Prediction block
    st.markdown("#### AI Prediction")
    if pred:
        c1, c2, c3 = st.columns(3)
        verdict = pred.get("prediction_label")
        if verdict is None:
            verdict = "True" if bool(pred.get("prediction_bool", False)) else "False"

        with c1:
            if verdict == "True":
                st.success(f"**Verdict: {verdict}**")
            elif verdict == "False":
                st.error(f"**Verdict: {verdict}**")
            else:
                st.warning(f"**Verdict: {verdict}**")
        with c2:
            accuracy = pred.get("accuracy")
            st.metric("Accuracy", f"{accuracy:.1f}%" if isinstance(accuracy, (int, float)) else "N/A")
        with c3:
            st.info(pred.get("model_used", "Model"))

        st.caption(f"Raw class: {pred.get('raw_prediction', pred.get('prediction', 'Unknown'))} • Boolean: {pred.get('prediction_bool', verdict == 'True')}")
        accuracy_pct = pred.get("accuracy") if isinstance(pred.get("accuracy"), (int, float)) else 0.0
        accuracy_progress = max(0.0, min(float(accuracy_pct) / 100.0, 1.0))
        st.progress(accuracy_progress)

        top_features = pred.get("top_features") or []
        if top_features:
            with st.expander("Why the model decided this", expanded=False):
                for feat, score in top_features:
                    st.write(f"- `{feat}` (impact: {score:.3f})")

    # Verification block
    st.markdown("#### Fact Verification")
    sources = ver.get("sources") or []
    if sources:
        st.success(f"Credible sources: {len(sources)}")

        # Always show a compact clickable source list.
        st.markdown("**Sources**")
        for i, source in enumerate(sources, 1):
            title = source.get("title", "Untitled")
            url = _normalize_source_url(source)
            domain = source.get("domain", "source")
            if url:
                st.link_button(
                    f"{i}. {title} ({domain})",
                    url,
                    use_container_width=True,
                    key=f"source_link_{result.get('timestamp','')}_{i}",
                )
            else:
                st.markdown(f"{i}. {title} ({domain})")

        for i, source in enumerate(sources, 1):
            with st.expander(f"Source {i}: {source.get('title', 'Untitled')}"):
                source_url = _normalize_source_url(source)
                if source_url:
                    st.markdown(f"**URL:** [{source_url}]({source_url})")
                else:
                    st.markdown("**URL:** unavailable")
                st.markdown(f"**Summary:** {source.get('summary', '')}")
                if source.get("relevance_score") is not None:
                    relevance_pct = float(source["relevance_score"])
                    relevance_progress = max(0.0, min(relevance_pct / 100.0, 1.0))
                    st.progress(relevance_progress)
                    st.caption(f"Relevance: {relevance_pct:.1f}%")
    else:
        st.warning("No credible sources found (or APIs not configured).")

    # Community Notes block
    st.markdown("#### Community Notes")
    if notes.get("summary"):
        st.info(notes["summary"])
    contradictions = notes.get("contradictions") or []
    confirmations = notes.get("confirmations") or []
    if contradictions:
        st.error("Contradictions")
        for c in contradictions:
            st.write(f"- {c}")
    if confirmations:
        st.success("Confirmations")
        for c in confirmations:
            st.write(f"- {c}")

    refs = notes.get("references") or []
    if refs:
        with st.expander("References", expanded=False):
            for ref in refs:
                st.markdown(f"- [{ref.get('source','source')}]({ref.get('url','')})")

if __name__ == "__main__":
    main()
