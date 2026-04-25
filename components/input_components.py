from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import streamlit as st

from components.text_extractor import TextExtractor

InputType = Literal["Text", "URL", "Image", "Video"]


@dataclass
class InputResult:
    input_type: InputType
    extracted_text: str | None
    raw: Any = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


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
    )
    text = user_text.strip() if user_text else ""
    if not text:
        return InputResult(input_type="Text", extracted_text=None)
    return InputResult(
        input_type="Text",
        extracted_text=text,
        raw=user_text,
        meta={"chars": len(text), "words": len(text.split())},
    )


def _render_url(extractor: TextExtractor) -> InputResult:
    url = st.text_input("News article URL", placeholder="https://example.com/news-article")
    if not url or not url.strip():
        return InputResult(input_type="URL", extracted_text=None)

    cleaned = url.strip()
    with st.expander("URL extraction options", expanded=False):
        preprocess = st.toggle("Preprocess extracted text", value=True, help="Normalize whitespace and remove odd characters.")

    with st.spinner("Extracting content from URL…"):
        try:
            text = extractor.extract_from_url(cleaned)
            if preprocess:
                text = extractor.preprocess_text(text)
            if not text or not text.strip():
                return InputResult(input_type="URL", extracted_text=None, raw=cleaned, errors=["No extractable text found."])
            text = text.strip()
            with st.expander("Preview extracted content", expanded=False):
                st.text_area("Extracted text", text, height=140, disabled=True)
            return InputResult(
                input_type="URL",
                extracted_text=text,
                raw=cleaned,
                meta={"chars": len(text), "words": len(text.split())},
            )
        except Exception as e:
            return InputResult(input_type="URL", extracted_text=None, raw=cleaned, errors=[str(e)])


def _render_image(extractor: TextExtractor) -> InputResult:
    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Upload a screenshot/photo containing news text.",
    )
    if uploaded_image is None:
        return InputResult(input_type="Image", extracted_text=None)

    st.image(uploaded_image, caption="Uploaded image", use_container_width=True)
    with st.expander("OCR options", expanded=False):
        lang = st.selectbox("OCR language", options=["eng"], index=0)
        preprocess = st.toggle("Enhance contrast before OCR", value=True)

    with st.spinner("Extracting text from image (OCR)…"):
        try:
            text = extractor.extract_from_image(uploaded_image, lang=lang, preprocess=preprocess)
            if not text or not text.strip():
                return InputResult(input_type="Image", extracted_text=None, raw=uploaded_image, errors=["No text detected in image."])
            text = text.strip()
            with st.expander("Preview extracted text", expanded=False):
                st.text_area("OCR text", text, height=140, disabled=True)
            return InputResult(
                input_type="Image",
                extracted_text=text,
                raw=uploaded_image,
                meta={"chars": len(text), "words": len(text.split()), "ocr_lang": lang},
            )
        except Exception as e:
            return InputResult(input_type="Image", extracted_text=None, raw=uploaded_image, errors=[str(e)])


def _render_video(extractor: TextExtractor) -> InputResult:
    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv", "wmv"],
        help="Uploads can be large; transcription depends on Whisper availability.",
    )
    if uploaded_video is None:
        return InputResult(input_type="Video", extracted_text=None)

    st.video(uploaded_video)
    with st.expander("Transcription options", expanded=False):
        preprocess = st.toggle("Preprocess transcript", value=True)

    with st.spinner("Transcribing audio from video…"):
        try:
            text = extractor.extract_from_video(uploaded_video)
            if preprocess:
                text = extractor.preprocess_text(text)
            if not text or not text.strip():
                return InputResult(input_type="Video", extracted_text=None, raw=uploaded_video, errors=["No transcript available."])
            text = text.strip()
            with st.expander("Preview transcript", expanded=False):
                st.text_area("Transcript", text, height=140, disabled=True)
            warnings: list[str] = []
            if "not available" in text.lower():
                warnings.append("Video transcription is running in fallback mode (Whisper not installed/available).")
            return InputResult(
                input_type="Video",
                extracted_text=text,
                raw=uploaded_video,
                warnings=warnings,
                meta={"chars": len(text), "words": len(text.split())},
            )
        except Exception as e:
            return InputResult(input_type="Video", extracted_text=None, raw=uploaded_video, errors=[str(e)])

