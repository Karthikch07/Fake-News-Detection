from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Literal

import streamlit as st

from components.text_extractor import TextExtractor

InputType = Literal["Text", "URL", "Image", "Video"]

MIN_WORDS = 3


@dataclass
class InputResult:
    input_type: InputType
    extracted_text: str | None
    raw: Any = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return bool(self.extracted_text) and not self.errors


# Extraction is cached so clicking any other widget doesn't re-download the
# page or re-run OCR / transcription. Exceptions are never cached.
@st.cache_data(show_spinner=False, ttl=3600, max_entries=50)
def _cached_url_text(url: str, _extractor: TextExtractor) -> tuple[str, dict]:
    return _extractor.extract_from_url_with_meta(url)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=20)
def _cached_ocr_text(data: bytes, lang: str, preprocess: bool, _extractor: TextExtractor) -> str:
    return _extractor.extract_from_image(BytesIO(data), lang=lang, preprocess=preprocess)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=5)
def _cached_transcript(data: bytes, name: str, _extractor: TextExtractor) -> str:
    buf = BytesIO(data)
    buf.name = name
    return _extractor.extract_from_video(buf)


def _finish(input_type: InputType, text: str, raw: Any, *, preview_label: str | None = None, **meta) -> InputResult:
    text = text.strip()
    words = len(text.split())
    result = InputResult(input_type=input_type, extracted_text=text, raw=raw, meta={"chars": len(text), "words": words, **meta})
    if words < MIN_WORDS:
        result.errors.append(f"Only {words} word(s) found — at least {MIN_WORDS} words are needed to analyze.")
    elif words < 8:
        result.warnings.append("Short input: predictions on just a few words are less reliable.")
    if preview_label:
        with st.expander(f"{preview_label} ({words} words)", expanded=False):
            st.text_area(preview_label, text, height=160, disabled=True, label_visibility="collapsed")
    return result


def render_input_panel(*, extractor: TextExtractor, input_type: InputType) -> InputResult:
    if input_type == "Text":
        return _render_text()
    if input_type == "URL":
        return _render_url(extractor)
    if input_type == "Image":
        return _render_image(extractor)
    if input_type == "Video":
        return _render_video(extractor)
    return InputResult(input_type=input_type, extracted_text=None, errors=[f"Unsupported input type: {input_type}"])


def _render_text() -> InputResult:
    user_text = st.text_area(
        "Enter news text or headline",
        height=160,
        placeholder="Paste the article text or a headline + key claims…",
        key="intake_text",
    )
    text = (user_text or "").strip()
    if not text:
        return InputResult(input_type="Text", extracted_text=None)
    return _finish("Text", text, user_text)


def _render_url(extractor: TextExtractor) -> InputResult:
    url = st.text_input("News article URL", placeholder="https://example.com/news-article", key="intake_url")
    cleaned = (url or "").strip()
    if not cleaned:
        return InputResult(input_type="URL", extracted_text=None)
    if " " in cleaned or "." not in cleaned:
        return InputResult(input_type="URL", extracted_text=None, raw=cleaned, errors=["That doesn't look like a valid URL."])

    with st.spinner("Extracting article from URL…"):
        try:
            raw_text, page = _cached_url_text(cleaned, extractor)
            text = extractor.preprocess_text(raw_text)
        except Exception as e:
            return InputResult(input_type="URL", extracted_text=None, raw=cleaned, errors=[f"URL extraction failed: {e}"])
    _render_page_info(cleaned, page)
    return _finish("URL", text, cleaned, preview_label="Extracted article text", url=cleaned, page=page)


def _page_age_warning(date: str) -> str | None:
    import datetime as dt

    try:
        published = dt.date.fromisoformat(date[:10])
    except (ValueError, TypeError):
        return None
    days = (dt.date.today() - published).days
    if days > 365:
        years = days // 365
        return f"Published {years} year{'s' if years != 1 else ''} ago. Check it isn't an old story being shared as new."
    return None


def _render_page_info(url: str, page: dict) -> None:
    from html import escape

    from components import source_reputation as rep

    info = rep.describe(url)
    bits = [f'<b>{escape(page.get("sitename") or info["domain"])}</b>',
            f'<span class="rep rep-{info["rating"]}">{escape(info["label"])}</span>']
    if page.get("date"):
        bits.append(f'published {escape(page["date"])}')
    if page.get("author"):
        bits.append(f'by {escape(page["author"][:60])}')
    st.markdown(f'<div class="page-info">{" · ".join(bits)}</div>', unsafe_allow_html=True)
    if info["rating"] in ("satire", "unreliable"):
        st.warning(f'{info["label"]}: {info["detail"]}')
    warn = _page_age_warning(page.get("date") or "")
    if warn:
        st.warning(warn)


def _render_image(extractor: TextExtractor) -> InputResult:
    if not extractor.ocr_available:
        st.warning(
            "Image OCR is unavailable: the Tesseract program is not installed. Install it from "
            "https://github.com/UB-Mannheim/tesseract/wiki (Windows) or set `TESSERACT_CMD`, then restart. "
            "Meanwhile, paste the text using **Text** input."
        )
    uploaded_image = st.file_uploader(
        "Upload a screenshot or photo of the news",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        key="intake_image",
    )
    if uploaded_image is None:
        return InputResult(input_type="Image", extracted_text=None)

    c1, c2 = st.columns([2, 1])
    with c2:
        preprocess = st.toggle("Enhance before OCR", value=True, key="ocr_preprocess")
    with c1:
        st.image(uploaded_image, caption=uploaded_image.name, use_container_width=True)

    with st.spinner("Reading text from image (OCR)…"):
        try:
            text = _cached_ocr_text(uploaded_image.getvalue(), "eng", preprocess, extractor)
        except Exception as e:
            return InputResult(input_type="Image", extracted_text=None, raw=uploaded_image.name, errors=[str(e)])
    return _finish("Image", text, uploaded_image.name, preview_label="Text read from image", file=uploaded_image.name)


def _render_video(extractor: TextExtractor) -> InputResult:
    if not extractor.video_available:
        st.warning(
            f"Video transcription is unavailable: {extractor.video_unavailable_reason}. "
            "Install `faster-whisper` to enable it, or paste the transcript using **Text** input."
        )
    uploaded_video = st.file_uploader(
        "Upload a video clip",
        type=["mp4", "avi", "mov", "mkv", "wmv", "webm", "mp3", "wav", "m4a"],
        key="intake_video",
    )
    if uploaded_video is None:
        return InputResult(input_type="Video", extracted_text=None)

    st.video(uploaded_video)
    with st.spinner("Transcribing speech (first run downloads the speech model)…"):
        try:
            text = _cached_transcript(uploaded_video.getvalue(), uploaded_video.name, extractor)
        except Exception as e:
            return InputResult(input_type="Video", extracted_text=None, raw=uploaded_video.name, errors=[str(e)])
    return _finish("Video", text, uploaded_video.name, preview_label="Transcript", file=uploaded_video.name)
