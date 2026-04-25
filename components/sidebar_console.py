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


def _queue_title(item: dict) -> str:
    itype = item.get("input_type", "Unknown")
    model = item.get("model_choice", "")
    snippet = item.get("snippet", "")
    return f"{itype} • {model} • {snippet}"


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

    st.sidebar.markdown("## Control Console")
    st.sidebar.markdown(
        '<div class="pill">wizard • queue • batch • export</div>',
        unsafe_allow_html=True,
    )

    # Step wizard
    steps = ["1) Input", "2) Model", "3) Actions", "4) Queue/Export"]
    st.session_state.console_step = st.sidebar.radio(
        "Workflow",
        steps,
        index=steps.index(st.session_state.console_step) if st.session_state.console_step in steps else 0,
        horizontal=False,
    )

    # Always-available controls
    input_type: InputType = st.sidebar.selectbox("Input type", ["Text", "URL", "Image", "Video"])

    models = predictor.available_models()
    if "model_choice" not in st.session_state or st.session_state.model_choice not in models:
        st.session_state.model_choice = models[0]
    model_choice = st.sidebar.selectbox(
        "Prediction model",
        models,
        index=models.index(st.session_state.model_choice),
    )
    st.session_state.model_choice = model_choice

    with st.sidebar.expander("Verification sources", expanded=False):
        st.caption("Optional APIs improve quality. If not set, the app runs in offline fallback mode.")
        google_ok = bool(verifier.google_api_key and verifier.google_cx)
        bing_ok = bool(verifier.bing_api_key)
        if not google_ok and not bing_ok:
            st.success("Offline verification mode is active.")
            st.write("Google Custom Search: optional")
            st.write("Bing Search: optional")
        else:
            st.write(f"Google Custom Search: {'configured' if google_ok else 'optional'}")
            st.write(f"Bing Search: {'configured' if bing_ok else 'optional'}")

    # Step panels
    input_result = InputResult(input_type=input_type, extracted_text=None)

    if st.session_state.console_step in ("1) Input", "3) Actions"):
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Input intake")
        input_result = render_input_panel(extractor=extractor, input_type=input_type)
        meta = input_result.meta or {}
        if input_result.extracted_text:
            st.sidebar.caption(f"Ready: {meta.get('words', 0)} words • {meta.get('chars', 0)} chars")
        if input_result.warnings:
            for w in input_result.warnings:
                st.sidebar.warning(w)
        if input_result.errors:
            for e in input_result.errors:
                st.sidebar.error(e)

    if st.session_state.console_step == "2) Model":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Model notes")
        st.sidebar.caption("All model labels are available. If advanced runtimes are missing, the app safely uses the baseline TF-IDF classifier.")
        st.sidebar.write(f"- Selected: **{model_choice}**")
        if model_choice.startswith("BERT"):
            st.sidebar.write("- Optional runtimes: `transformers` + `torch`")
        if model_choice.startswith("LSTM"):
            st.sidebar.write("- Optional runtime: `tensorflow`")
        if model_choice.startswith("GAN"):
            st.sidebar.write("- Optional runtime: `torch` (experimental score)")

    if st.session_state.console_step == "3) Actions":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Actions")
        can_use = bool(input_result.extracted_text and not input_result.errors)

        cols = st.sidebar.columns(2)
        with cols[0]:
            analyze_now = st.button("Analyze now", type="primary", use_container_width=True, disabled=not can_use)
        with cols[1]:
            add_queue = st.button("Add to queue", use_container_width=True, disabled=not can_use)

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

        # Returning the 'analyze_now' flag via session_state is simplest for the caller
        st.session_state._analyze_now = bool(analyze_now)

    if st.session_state.console_step == "4) Queue/Export":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Queue")
        st.session_state.batch_limit = st.sidebar.slider("Max queue size", min_value=1, max_value=25, value=int(st.session_state.batch_limit))
        st.session_state.batch_stop_on_error = st.sidebar.toggle("Stop batch on error", value=bool(st.session_state.batch_stop_on_error))

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

            st.sidebar.markdown("---")
            st.sidebar.markdown("### Batch run")
            st.session_state._run_queue = st.button("Run queue", type="primary", use_container_width=True)
        else:
            st.session_state._run_queue = False

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Export")
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

        st.sidebar.markdown("---")
        st.sidebar.markdown("### History")
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
    }
    st.session_state._analyze_now = False
    st.session_state._run_queue = False
    return actions

