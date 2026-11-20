import os

import streamlit as st
from dotenv import load_dotenv
from components.text_extractor import TextExtractor
from components.ml_predictor import MLPredictor, model_needs_training
from components.fact_verifier import FactVerifier
from components.community_notes import CommunityNotesGenerator, final_assessment
from components.accuracy_tracker import feedback_rows, graded_ids, live_stats, record_feedback
from components.history_store import load_history, save_history
from components import source_reputation as rep
from components.ui_3d import inject_glass_css
from components.news_alert_ui import render_news_alert_header
from components.sidebar_console import render_console, pop_actions

# Load environment variables from a local .env file when present.
load_dotenv(override=False)

# Configure page
st.set_page_config(
    page_title="Fake News Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

INPUT_META = {
    "Text": {"sub": "Paste the article text, or a headline with its main claims."},
    "URL": {"sub": "Paste a link and the article text is pulled from the page."},
    "Image": {"sub": "Upload a screenshot or photo. The text is read with OCR."},
    "Video": {"sub": "Upload a clip. The speech is transcribed to text."},
}

def _card_header(title: str, subtitle: str, tag: str) -> str:
    from html import escape

    return (
        f'<div class="card-head">'
        f'<div><div class="card-title">{escape(title)}</div>'
        f'<div class="card-sub">{escape(subtitle)}</div></div>'
        + (f'<div class="card-tag">{escape(tag)}</div>' if tag else "")
        + '</div>'
    )


def _components_code_version() -> str:
    """Changes whenever a component's code changes, so cached objects are rebuilt
    instead of old instances being used with new code."""
    import glob

    files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "components", "*.py")))
    return "|".join(f"{os.path.basename(f)}:{os.path.getmtime(f):.0f}" for f in files)


@st.cache_resource(show_spinner="Loading models…", max_entries=1)
def _build_components(code_version: str):
    """Initialize all system components (code_version only keys the cache)."""
    return {
        'extractor': TextExtractor(),
        'predictor': MLPredictor(),
        'verifier': FactVerifier(),
        'notes_generator': CommunityNotesGenerator()
    }


def initialize_components():
    return _build_components(_components_code_version())


initialize_components.clear = _build_components.clear


def _init_state():
    if "history" not in st.session_state:
        st.session_state.history = load_history()
    if "active_result" not in st.session_state:
        st.session_state.active_result = None


def _add_to_history(result: dict) -> None:
    st.session_state.history = [result] + [h for h in st.session_state.history if h.get("id") != result.get("id")][:49]
    save_history(st.session_state.history)


def _retrain() -> None:
    from components.ml_predictor import train_model

    rows = feedback_rows()
    with st.status(f"Retraining with {len(rows)} graded item(s). This takes about 3 minutes…", expanded=True) as status:
        try:
            bundle = train_model(progress=st.write, feedback=rows)
        except Exception as e:
            status.update(label=f"Retraining failed: {e}", state="error")
            return
        m = bundle["metrics"]
        status.update(
            label=f"Retrained. Test accuracy {m['accuracy']:.1f}%, real-world headlines "
                  f"{(m.get('realworld_accuracy') or 0):.1f}%.",
            state="complete", expanded=False,
        )
    initialize_components.clear()
    st.rerun()


def main():
    _init_state()
    inject_glass_css()

    # First launch (or after a model format change): download the dataset and train,
    # and download the evidence-checking models (~110 MB, once).
    if model_needs_training():
        with st.status(
            "Setting up: downloading the training data and evidence models, then training the classifier. "
            "This takes about 3–4 minutes, once.",
            expanded=True,
        ) as status:
            st.write("Training data: pushpdeep/fake_news_combined (~45k labelled news articles, ~70 MB)")
            st.write("Evidence models: MiniLM sentence embeddings and DeBERTa-v3 NLI (ONNX, ~110 MB)")
            components = initialize_components()
            acc = components["predictor"].metrics.get("accuracy") or 0
            status.update(label=f"Ready. Classifier test accuracy {acc:.1f}%", state="complete", expanded=False)
    else:
        components = initialize_components()

    render_news_alert_header()

    # Input intake card: the sidebar console renders its input panel in here.
    with st.container(key="intake_card"):
        intake_head = st.empty()
        console = render_console(
            extractor=components["extractor"],
            predictor=components["predictor"],
            verifier=components["verifier"],
            history=st.session_state.history,
        )
        itype = console.input_type
        ires = console.input_result
        intake_head.markdown(
            _card_header(
                {"Text": "Check text", "URL": "Check a link", "Image": "Check an image", "Video": "Check a video"}[itype],
                INPUT_META[itype]["sub"],
                console.model_choice,
            ),
            unsafe_allow_html=True,
        )
        if st.session_state.console_step not in ("1) Input", "3) Actions"):
            st.markdown(
                '<div class="card-empty">Select <b>Input</b> in the sidebar to add content here.</div>',
                unsafe_allow_html=True,
            )
        else:
            for err in ires.errors:
                st.error(err)
            for warn in ires.warnings:
                st.caption(warn)
            meta = ires.meta or {}
            c1, c2 = st.columns([3, 1], vertical_alignment="center")
            with c1:
                st.markdown(
                    f'<div class="counter"><b>{meta.get("words", 0)}</b> words · '
                    f'<b>{meta.get("chars", 0)}</b> characters</div>',
                    unsafe_allow_html=True,
                )
            with c2:
                intake_analyze = st.button(
                    "Analyze",
                    type="primary",
                    use_container_width=True,
                    disabled=not ires.extracted_text or bool(ires.errors),
                    key="intake_analyze_btn",
                )
            if intake_analyze:
                st.session_state._analyze_now = True
    actions = pop_actions()
    if actions["retrain"]:
        _retrain()

    # Immediate analyze
    if actions["analyze_now"] and console.input_result.extracted_text and not console.input_result.errors:
        st.session_state.active_result = analyze_news(
            console.input_result.extracted_text,
            components,
            input_type=console.input_type,
            model_choice=console.model_choice,
            page_url=(console.input_result.meta or {}).get("url"),
            page_meta=(console.input_result.meta or {}).get("page"),
        )
        _add_to_history(st.session_state.active_result)

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
                _add_to_history(r)
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

    st.markdown(
        '<div class="section-title"><h3>Result</h3><div class="line"></div></div>',
        unsafe_allow_html=True,
    )

    if st.session_state.active_result:
        render_results(st.session_state.active_result)
    else:
        st.markdown(
            """
<div class="steps">
  <div class="step"><div class="num">01</div><div class="t">Extract</div>
    <div class="d">Text is taken from what you paste, a link, an image (OCR) or a video (speech-to-text).</div></div>
  <div class="step"><div class="num">02</div><div class="t">Find claims</div>
    <div class="d">The factual statements in the text are picked out to be checked one by one.</div></div>
  <div class="step"><div class="num">03</div><div class="t">Check sources</div>
    <div class="d">Each claim is compared with news coverage, fact-checkers and Wikipedia to see if they support or contradict it.</div></div>
  <div class="step"><div class="num">04</div><div class="t">Verdict</div>
    <div class="d">Evidence decides the verdict. The writing-style classifier is the fallback when no source covers it.</div></div>
</div>
            """,
            unsafe_allow_html=True,
        )

    render_live_accuracy(components["predictor"])


def analyze_news(text: str, components: dict, *, input_type: str, model_choice: str,
                 page_url: str | None = None, page_meta: dict | None = None) -> dict:
    """Run the full pipeline and return a serializable result."""
    import datetime as _dt
    import uuid

    result: dict = {
        "id": uuid.uuid4().hex[:12],
        "timestamp": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "input_type": input_type,
        "model_choice": model_choice,
        "snippet": (" ".join(text.split())[:72] + ("…" if len(text) > 72 else "")),
        "text": text,
        "page_url": page_url,
        "page": page_meta,
        "prediction": None,
        "verification": None,
        "notes": None,
        "final": None,
        "errors": [],
    }

    try:
        with st.spinner("Scoring the writing style…"):
            result["prediction"] = components["predictor"].predict(text, model_choice=model_choice)
    except Exception as e:
        result["errors"].append(f"Prediction error: {e}")
        return result

    try:
        with st.spinner("Checking each claim against news, fact-checkers and Wikipedia…"):
            result["verification"] = components["verifier"].verify_claims(text, page_url=page_url)
    except Exception as e:
        result["errors"].append(f"Verification error: {e}")
        result["verification"] = {"claims": [], "overall": {"label": "Unverified"}, "manual_check_links": [],
                                  "fact_checks": [], "sources": [], "errors": []}

    try:
        result["notes"] = components["notes_generator"].generate_notes(text, result["prediction"], result["verification"])
    except Exception as e:
        result["errors"].append(f"Community Notes error: {e}")

    result["final"] = final_assessment(result["prediction"], result["verification"])
    return result


VERDICT_DISPLAY = {
    "False": ("Likely fake", "fake"),
    "True": ("Likely real", "real"),
    "Uncertain": ("Uncertain", "unsure"),
}

CLAIM_BADGE = {
    "Supported": "real",
    "Contradicted": "fake",
    "Disputed": "unsure",
    "Fact-checked": "unsure",
    "Not enough evidence": "none",
}
STANCE_LABEL = {"supports": ("Supports", "real"), "contradicts": ("Contradicts", "fake"), "neutral": ("Related", "none")}


def _pct(v) -> str:
    return f"{v:.1f}%" if isinstance(v, (int, float)) else "—"


def _md_bold(text: str) -> str:
    """Escape text for HTML, keeping **bold** markdown."""
    import re
    from html import escape

    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(text))


def _esc(text) -> str:
    from html import escape

    return escape(str(text or ""))


def render_live_accuracy(predictor) -> None:
    """Model performance: held-out test metrics + live accuracy from user grades."""
    m = getattr(predictor, "metrics", {}) or {}
    live = live_stats()
    st.markdown(
        '<div class="section-title"><h3>Style classifier accuracy</h3><div class="line"></div></div>',
        unsafe_allow_html=True,
    )

    def _frac(prefix):
        c, w = m.get(f"{prefix}_correct"), m.get(f"{prefix}_wrong")
        return f"{c}/{c + w} right · {m.get(prefix + '_uncertain')} unsure" if c is not None else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Test-set accuracy", _pct(m.get("accuracy")),
        delta=f"{m.get('n_test', 0):,} held-out articles", delta_color="off",
        help="In-distribution: same sources and era as the training data, so this is the optimistic number.",
    )
    c2.metric(
        "Real-world headlines", _pct(m.get("realworld_accuracy")),
        delta=_frac("realworld"), delta_color="off",
        help="Hand-written realistic headlines never used for training. Accuracy of confident calls.",
    )
    c3.metric(
        "Style-swapped cases", _pct(m.get("hard_accuracy")),
        delta=_frac("hard"), delta_color="off",
        help="True facts written casually and false claims written like wire news. The classifier alone can't "
             "handle these; that's what the source check is for.",
    )
    c4.metric(
        "Live accuracy (your grades)", _pct(live.accuracy),
        delta=(f"{live.graded} graded · last {live.recent_n}: {live.recent_accuracy:.0f}%" if live.recent_n
               else f"{live.graded} graded"),
        delta_color="off",
        help="Updated as you grade results with “Was this right?”. Uncertain verdicts are excluded.",
    )
    st.caption(
        m.get("description", "")
        + (
            f" Live fake-news precision {_pct(live.fake_precision)}, recall {_pct(live.fake_recall)}."
            if live.graded
            else " Grade results with “Was this right?” to measure live accuracy on what you actually check."
        )
    )


def render_feedback(result: dict) -> None:
    pred = result.get("prediction") or {}
    rid = result.get("id")
    if not rid or not pred:
        return
    already = graded_ids().get(rid)
    st.markdown("**Was this right?** Your answer tracks live accuracy and can be used to retrain the classifier.")
    c1, c2, c3 = st.columns([1, 1, 3])
    label = pred.get("prediction_label", "Uncertain")
    with c1:
        if st.button("It's fake", key=f"fb_fake_{rid}", use_container_width=True,
                     type="primary" if already == "Fake" else "secondary"):
            record_feedback(rid, label, "Fake", result.get("text", ""), pred.get("confidence"))
            st.rerun()
    with c2:
        if st.button("It's real", key=f"fb_real_{rid}", use_container_width=True,
                     type="primary" if already == "Real" else "secondary"):
            record_feedback(rid, label, "Real", result.get("text", ""), pred.get("confidence"))
            st.rerun()
    with c3:
        if already:
            if label == "Uncertain":
                st.caption(f"Recorded as {already.lower()} (uncertain style verdicts don't count toward accuracy).")
            else:
                right = (label == "False") == (already == "Fake")
                st.caption(f"Recorded as {already.lower()}: the style classifier was {'right' if right else 'wrong'}.")


def _evidence_row(e: dict) -> str:
    stance, cls = STANCE_LABEL.get(e.get("stance", "neutral"), ("Related", "none"))
    info = rep.RATING_INFO.get(e.get("rating", "unrated"), rep.RATING_INFO["unrated"])
    outlet = e.get("outlet") or e.get("domain") or "source"
    title = e.get("text") if e.get("kind") == "reference" else e.get("title")
    meta = [f'<span class="rep rep-{_esc(e.get("rating"))}" title="{_esc(info[1])}">{_esc(info[0])}</span>']
    if e.get("date"):
        meta.append(_esc(e["date"]))
    return (
        f'<div class="ev"><span class="tag tag-{cls}">{stance}</span>'
        f'<div class="ev-body"><a href="{_esc(e.get("url"))}" target="_blank" rel="noopener">{_esc(title)}</a>'
        f'<div class="ev-meta"><b>{_esc(outlet)}</b> · {" · ".join(meta)}</div></div></div>'
    )


def render_claims(ver: dict) -> None:
    claims = ver.get("claims") or []
    if ver.get("mode") == "unavailable":
        for err in ver.get("errors") or []:
            st.warning(err)
        return
    if not claims:
        st.markdown('<div class="note">No checkable factual claim was found. Try pasting the full headline or a few sentences.</div>',
                    unsafe_allow_html=True)
        return
    for i, c in enumerate(claims, 1):
        badge = CLAIM_BADGE.get(c["verdict"], "none")
        decisive = [e for e in c.get("evidence", []) if e.get("stance") != "neutral"]
        related = [e for e in c.get("evidence", []) if e.get("stance") == "neutral"]
        rows = "".join(_evidence_row(e) for e in decisive[:5])
        if not decisive:
            rows = ('<div class="ev-empty">No source clearly supports or contradicts this claim.'
                    + (" Related coverage is listed below." if related else "") + "</div>")
        st.markdown(
            f'<div class="claim"><div class="claim-head"><span class="claim-n">Claim {i}</span>'
            f'<span class="badge badge-{badge}">{_esc(c["verdict"])}</span></div>'
            f'<div class="claim-text">“{_esc(c["claim"])}”</div>{rows}</div>',
            unsafe_allow_html=True,
        )
        if related:
            with st.expander(f"Related coverage for claim {i} ({len(related[:6])})", expanded=False):
                st.markdown("".join(_evidence_row(e) for e in related[:6]), unsafe_allow_html=True)
    soft = [e for e in ver.get("errors") or [] if e]
    if soft:
        st.caption("Some sources couldn't be searched: " + "; ".join(soft) + ".")


def _highlight(text: str, pred: dict) -> str:
    """Escape the text and mark the phrases the style classifier flagged."""
    import re

    from components.ml_predictor import style_signals

    s = style_signals(text)
    terms = sorted({*s.get("sensational", []), *s.get("caps_words", [])}, key=len, reverse=True)
    html = _esc(text)
    if terms:
        rx = re.compile(r"(?<![\w])(" + "|".join(re.escape(_esc(t)) for t in terms) + r")(?![\w])", re.IGNORECASE)
        html = rx.sub(r"<mark>\1</mark>", html)
    return html.replace("\n", "<br>")


def render_results(result: dict) -> None:
    st.caption(
        f"Run {result.get('timestamp', '')} • Input: {result.get('input_type', '')} • “{result.get('snippet', '')}”"
    )
    for e in result.get("errors") or []:
        st.error(e)

    pred = result.get("prediction") or {}
    ver = result.get("verification") or {}
    notes = result.get("notes") or {}
    final = result.get("final") or final_assessment(pred, ver)
    overall = ver.get("overall") or {}

    # ------------------------------------------------------------- verdict
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown(
            f'<div class="verdict {final["kind"]}"><div class="k">Verdict · based on {_esc(final["basis"])}</div>'
            f'<div class="v">{_esc(final["label"])}</div><div class="r">{_esc(final["reason"])}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        n = overall.get("checked", 0)
        st.metric("Claims checked", f"{n}",
                  delta=f"{overall.get('supported', 0)} supported · {overall.get('contradicted', 0)} contradicted",
                  delta_color="off", help="Factual claims found in the text and checked against sources.")
    with c3:
        p_fake = float((pred.get("probabilities") or {}).get("False", 0))
        st.metric("Style: fake-looking", f"{p_fake:.0f}%",
                  help="How much the wording resembles fake news (writing-style classifier, calibrated). "
                       "It doesn't check facts.")

    page = ver.get("page_source")
    if page:
        st.markdown(
            f'<div class="page-info">Source: <b>{_esc(page["domain"])}</b> · '
            f'<span class="rep rep-{_esc(page["rating"])}">{_esc(page["label"])}</span> · {_esc(page["detail"])}</div>',
            unsafe_allow_html=True,
        )
    if ver.get("stale_warning"):
        st.warning(ver["stale_warning"])

    # ---------------------------------------------------------- claim check
    st.markdown("#### Claim check")
    st.caption("Each claim is compared with news headlines (Google News), fact-checkers and Wikipedia. "
               "A local NLI model (DeBERTa-v3) decides whether each source supports or contradicts it.")
    render_claims(ver)

    fact_checks = ver.get("fact_checks") or []
    if fact_checks:
        st.markdown("**Published fact-checks**")
        st.markdown("".join(
            f'<div class="ev"><span class="tag tag-{"fake" if fc.get("rating") else "none"}">'
            f'{_esc(fc.get("rating") or "Fact-check")}</span><div class="ev-body">'
            f'<a href="{_esc(fc.get("url"))}" target="_blank" rel="noopener">{_esc(fc.get("title") or fc.get("claim"))}</a>'
            f'<div class="ev-meta"><b>{_esc(fc.get("publisher"))}</b>{" · " + _esc(fc["date"]) if fc.get("date") else ""}</div>'
            f'</div></div>' for fc in fact_checks[:6]), unsafe_allow_html=True)

    extra = ver.get("sources") or []
    if extra:
        with st.expander(f"More coverage from web search ({len(extra)})", expanded=False):
            for s in extra:
                st.markdown(f"- [{s.get('title', 'Untitled')}]({s.get('url', '')}) · `{s.get('domain', '')}`")

    links = ver.get("manual_check_links") or []
    if links:
        st.caption("Check it yourself:")
        cols = st.columns(len(links))
        for col, link in zip(cols, links):
            with col:
                st.link_button(link["name"], link["url"], use_container_width=True)

    # --------------------------------------------------------- writing style
    if pred:
        st.markdown("#### Writing style")
        style_title, _ = VERDICT_DISPLAY.get(pred.get("prediction_label", "Uncertain"), ("Uncertain", "unsure"))
        conf = float(pred.get("confidence") or 0)
        st.progress(
            min(max(p_fake / 100, 0.0), 1.0),
            text=f"Style classifier: {style_title.lower()} ({conf:.0f}% confidence) · fake-looking {p_fake:.0f}%",
        )
        st.caption("This judges wording only. A false claim written like a news report can still look real, "
                   "and a true one written casually can look fake.")
        for n_ in pred.get("notes") or []:
            st.caption(n_)
        f1, f2 = st.columns(2)
        with f1:
            st.markdown("**Red flags**")
            flags = pred.get("red_flags") or []
            for f in flags:
                st.markdown(f"- {f}")
            if not flags:
                st.caption("None detected")
        with f2:
            st.markdown("**Credibility signals**")
            creds = pred.get("credibility_signals") or []
            for c in creds:
                st.markdown(f"- {c}")
            if not creds:
                st.caption("None detected")
        top = pred.get("top_features") or []
        if top:
            with st.expander("Words that drove the style score", expanded=False):
                for feat, impact in top:
                    st.markdown(f"- `{feat}` {'→ real' if impact > 0 else '→ fake'} (weight {abs(impact):.2f})")
                st.caption(pred.get("model_used", ""))

    render_feedback(result)

    # ------------------------------------------------------ community notes
    st.markdown("#### Community Notes")
    if notes.get("summary"):
        st.markdown(f'<div class="note">{_md_bold(notes["summary"])}</div>', unsafe_allow_html=True)
    refs = notes.get("references") or []
    if refs:
        with st.expander(f"References ({len(refs)})", expanded=False):
            for ref in refs:
                st.markdown(f"- [{ref.get('title') or ref.get('source', 'source')}]({ref.get('url', '')}) · {ref.get('source', '')}")

    with st.expander("Analyzed text (flagged phrases highlighted)", expanded=False):
        st.markdown(f'<div class="analyzed">{_highlight(result.get("text", ""), pred)}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
