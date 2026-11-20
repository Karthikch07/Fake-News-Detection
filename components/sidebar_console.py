from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import streamlit as st

from components.input_components import InputResult, InputType, render_input_panel
from components.ml_predictor import MLPredictor
from components.fact_verifier import FactVerifier


@dataclass
class ConsoleState:
    input_type: InputType
    model_choice: str
    input_result: InputResult


def init_console_state() -> None:
    if "queue" not in st.session_state:
        st.session_state.queue = []
    if "console_step" not in st.session_state:
        st.session_state.console_step = "1) Input"
    if "batch_stop_on_error" not in st.session_state:
        st.session_state.batch_stop_on_error = True
    if "batch_limit" not in st.session_state:
        st.session_state.batch_limit = 10
    # Widgets below own their state through their key alone. Passing index=/value=
    # computed from the previous run as well made every second click revert.
    if st.session_state.get("sb_workflow") not in STEPS:
        st.session_state.sb_workflow = st.session_state.console_step if st.session_state.console_step in STEPS else STEPS[0]
    st.session_state.setdefault("sb_batch_limit", int(st.session_state.batch_limit))
    st.session_state.setdefault("sb_stop_on_error", bool(st.session_state.batch_stop_on_error))


STEPS = ["1) Input", "2) Model", "3) Actions", "4) Queue/Export"]


def _queue_title(item: dict) -> str:
    itype = item.get("input_type", "Unknown")
    model = item.get("model_choice", "")
    snippet = item.get("snippet", "")
    return f"{itype} • {model} • {snippet}"


def _section(title: str) -> None:
    st.sidebar.markdown(
        f'<div class="sb-section">{title}<div class="sb-line"></div></div>',
        unsafe_allow_html=True,
    )


def _status_row(name: str, ok: bool, detail: str) -> str:
    return (
        f'<div class="sb-status"><span class="dot {"on" if ok else "off"}"></span>'
        f'<div><div class="n">{name}</div><div class="d">{detail}</div></div></div>'
    )


def render_console(
    *,
    extractor,
    predictor: MLPredictor,
    verifier: FactVerifier,
    history: list[dict],
) -> ConsoleState:
    """
    Sidebar console: wizard + model selection + input + queue + export + history loader.
    Returns the current InputResult and selections.
    """
    init_console_state()

    st.sidebar.markdown(
        '<div class="sb-brand"><div class="sb-title">Fake News Detector</div>'
        '<div class="sb-sub">Input, model, batch queue and export</div></div>',
        unsafe_allow_html=True,
    )
    _section("Workflow")

    # Step wizard
    st.session_state.setdefault("_retrain", False)
    st.session_state.console_step = st.sidebar.radio(
        "Workflow",
        STEPS,
        horizontal=False,
        format_func=lambda s: s.split(") ", 1)[-1].replace("/", " & "),
        label_visibility="collapsed",
        key="sb_workflow",
    )
    _section("Settings")

    # Always-available controls
    input_type: InputType = st.sidebar.selectbox("Input type", ["Text", "URL", "Image", "Video"], key="sb_input_type")

    models = predictor.available_models()
    if st.session_state.get("model_choice") not in models:
        st.session_state.model_choice = models[0]
    model_choice = st.session_state.model_choice
    if len(models) > 1:
        model_choice = st.sidebar.selectbox("Style classifier", models, index=models.index(model_choice))
        st.session_state.model_choice = model_choice

    from components.evidence import wikipedia_status

    google_ok = bool(verifier.google_api_key and verifier.google_cx)
    bing_ok = bool(verifier.bing_api_key)
    wiki = wikipedia_status()
    with st.sidebar.expander("System status", expanded=False):
        st.markdown(
            _status_row("Style classifier", True, model_choice)
            + _status_row("Evidence models", verifier.models_ready,
                          "MiniLM + DeBERTa NLI (ONNX)" if verifier.models_ready else "not loaded, needs internet on first run")
            + _status_row("News search", True, "Google News RSS, no key needed")
            + _status_row("Wikipedia", wiki == "ok", "no key needed" if wiki == "ok" else wiki)
            + _status_row("Image OCR", extractor.ocr_available, "Tesseract" if extractor.ocr_available else "install Tesseract")
            + _status_row("Video speech-to-text", extractor.video_available,
                          getattr(extractor, "speech_engine", "") or "install faster-whisper")
            + _status_row("Google search / Fact Check API", google_ok or bool(verifier.google_api_key),
                          "configured" if google_ok or verifier.google_api_key else "optional · add to .env")
            + _status_row("Bing search API", bing_ok, "configured" if bing_ok else "optional · add to .env"),
            unsafe_allow_html=True,
        )

    # Step panels
    input_result = InputResult(input_type=input_type, extracted_text=None)

    if st.session_state.console_step in ("1) Input", "3) Actions"):
        input_result = render_input_panel(extractor=extractor, input_type=input_type)
        meta = input_result.meta or {}
        _section("Input status")
        if input_result.errors:
            state, detail = "err", "Not ready yet. See the message under the input."
        elif input_result.extracted_text:
            state = "ok"
            detail = f"<b>{meta.get('words', 0)}</b> words · <b>{meta.get('chars', 0)}</b> chars ready"
        else:
            state, detail = "idle", "Nothing entered yet"
        st.sidebar.markdown(
            f'<div class="sb-card sb-{state}"><div class="sb-card-k">{input_type}</div>'
            f'<div class="sb-card-v">{detail}</div></div>',
            unsafe_allow_html=True,
        )

    if st.session_state.console_step == "2) Model":
        _section("Model card")
        m = getattr(predictor, "metrics", {}) or {}

        def _p(v):
            return f"{v:.1f}%" if isinstance(v, (int, float)) else "—"

        st.sidebar.markdown(
            '<div class="sb-card"><div class="sb-card-k">Style classifier</div>'
            f'<div class="sb-card-v">{getattr(predictor, "model_name", "TF-IDF classifier")}</div>'
            '<div class="sb-kpis">'
            f'<div><span>Test acc.</span><b>{_p(m.get("accuracy"))}</b></div>'
            f'<div><span>Real-world set</span><b>{_p(m.get("realworld_accuracy"))}</b></div>'
            f'<div><span>Fake F1</span><b>{_p(m.get("f1_fake"))}</b></div>'
            f'<div><span>Your graded items</span><b>{m.get("n_feedback", 0)}</b></div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        st.sidebar.caption(m.get("description", ""))
        st.sidebar.caption(
            "The classifier only judges writing style. Facts are checked separately against news, "
            "fact-checkers and Wikipedia using local MiniLM and DeBERTa NLI models."
        )

        _section("Retrain")
        from components.accuracy_tracker import feedback_rows

        rows = feedback_rows()
        new = len(rows) - int(m.get("n_feedback", 0) or 0)
        st.sidebar.caption(
            f"{len(rows)} item(s) graded with “Was this right?”"
            + (f", {new} not used for training yet." if new > 0 else ".")
            + " Retraining takes about 3 minutes."
        )
        st.session_state._retrain = st.sidebar.button(
            "Retrain with my feedback", use_container_width=True, disabled=not rows,
        )

    if st.session_state.console_step == "3) Actions":
        _section("Actions")
        can_use = bool(input_result.extracted_text and not input_result.errors)

        # "Analyze" lives on the input card in the main panel; the sidebar only queues.
        add_queue = st.sidebar.button("Add to queue", use_container_width=True, disabled=not can_use)
        st.sidebar.caption("Queue several inputs here, then run them together from Queue & Export.")

        if add_queue:
            text = input_result.extracted_text or ""
            snippet = (" ".join(text.split())[:72] + ("…" if len(text) > 72 else ""))
            st.session_state.queue = (
                [
                    {
                        "input_type": input_type,
                        "model_choice": model_choice,
                        "text": text,
                        "snippet": snippet,
                    }
                ]
                + st.session_state.queue
            )[: int(st.session_state.batch_limit)]
            st.sidebar.success("Added to queue.")


    if st.session_state.console_step == "4) Queue/Export":
        _section("Queue")
        st.session_state.batch_limit = st.sidebar.slider("Max queue size", min_value=1, max_value=25, key="sb_batch_limit")
        st.session_state.batch_stop_on_error = st.sidebar.toggle("Stop batch on error", key="sb_stop_on_error")

        q = st.session_state.queue
        st.sidebar.caption(f"Queued items: {len(q)}")
        if q:
            titles = [_queue_title(i) for i in q]
            sel = st.sidebar.selectbox("Select queued item", options=list(range(len(titles))), format_func=lambda i: titles[i])
            qc1, qc2 = st.sidebar.columns(2)
            with qc1:
                if st.button("Remove selected", use_container_width=True):
                    st.session_state.queue.pop(sel)
                    st.rerun()
            with qc2:
                if st.button("Clear queue", use_container_width=True):
                    st.session_state.queue = []
                    st.rerun()

            _section("Batch run")
            st.session_state._run_queue = st.sidebar.button("Run queue", type="primary", use_container_width=True)
        else:
            st.session_state._run_queue = False

        _section("Export")
        if history:
            st.sidebar.download_button(
                "Download results (JSON)",
                data=json.dumps(history, ensure_ascii=False, indent=2),
                file_name="fake_news_results.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.sidebar.caption("No results yet to export.")

        _section("History")
        if history:
            titles = [f"{h.get('timestamp','')} • {h.get('input_type','')} • {h.get('model_choice','')} • {h.get('snippet','')}" for h in history]
            idx = st.sidebar.selectbox("Previous runs", options=list(range(len(titles))), format_func=lambda i: titles[i])
            if st.sidebar.button("Load selected run", use_container_width=True):
                st.session_state.active_result = history[idx]
                st.rerun()
        else:
            st.sidebar.caption("No runs yet.")

    return ConsoleState(input_type=input_type, model_choice=model_choice, input_result=input_result)


def pop_actions() -> dict[str, Any]:
    """
    Helper to read and clear one-shot console actions set during rendering.
    """
    actions = {
        "analyze_now": bool(st.session_state.get("_analyze_now", False)),
        "run_queue": bool(st.session_state.get("_run_queue", False)),
        "retrain": bool(st.session_state.get("_retrain", False)),
    }
    st.session_state._analyze_now = False
    st.session_state._run_queue = False
    st.session_state._retrain = False
    return actions

