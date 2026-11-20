import os
import re
import shutil
import tempfile

import cv2
import numpy as np
import pytesseract
import trafilatura
from PIL import Image, ImageOps

# Optional speech-to-text engines. faster-whisper is preferred: it needs no
# PyTorch and decodes audio with PyAV (bundled ffmpeg libraries).
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

def _decode_audio(path: str, sampling_rate: int = 16000) -> np.ndarray:
    """Decode the first audio track to 16 kHz mono float32 with PyAV.

    faster-whisper's own decoder passes an argument that PyAV 15+ removed, so
    decoding here keeps transcription working whichever PyAV version is installed.
    """
    import av

    chunks = []
    resampler = av.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
    with av.open(path, mode="r") as container:
        if not container.streams.audio:
            raise ValueError("the file has no audio track")
        for frame in container.decode(audio=0):
            try:
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray().reshape(-1))
            except av.error.InvalidDataError:
                continue
        for out in resampler.resample(None):  # flush
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError("no audio could be decoded")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


_TESSERACT_WINDOWS_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]


def _configure_tesseract() -> bool:
    """Point pytesseract at a Tesseract binary if one exists. Returns availability."""
    custom = os.getenv("TESSERACT_CMD", "")
    candidates = [custom] if custom else []
    candidates += [shutil.which("tesseract") or ""] + _TESSERACT_WINDOWS_PATHS
    for path in candidates:
        if path and os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return True
    return False


class TextExtractor:
    def __init__(self):
        self.ocr_available = _configure_tesseract()
        self.whisper_model = None
        self.speech_engine = ""
        self.video_unavailable_reason = ""
        model_name = os.getenv("WHISPER_MODEL", "base")
        if FASTER_WHISPER_AVAILABLE:
            # The model (~140 MB for "base") is downloaded on first use.
            self.speech_engine = f"faster-whisper ({model_name})"
            self._model_name = model_name
        elif WHISPER_AVAILABLE and shutil.which("ffmpeg"):
            try:
                self.whisper_model = whisper.load_model(model_name)
                self.speech_engine = f"openai-whisper ({model_name})"
            except Exception as e:
                self.video_unavailable_reason = f"the Whisper model could not be loaded ({e})"
        elif WHISPER_AVAILABLE:
            self.video_unavailable_reason = "`ffmpeg` is not installed or not on PATH"
        else:
            self.video_unavailable_reason = "no speech-to-text package is installed (`faster-whisper`)"

    @property
    def video_available(self) -> bool:
        return bool(self.speech_engine)

    def _transcribe(self, path: str) -> str:
        if FASTER_WHISPER_AVAILABLE:
            if self.whisper_model is None:  # lazy: download/load only when a video is used
                self.whisper_model = WhisperModel(self._model_name, device="cpu", compute_type="int8")
            segments, _info = self.whisper_model.transcribe(_decode_audio(path), vad_filter=True)
            return " ".join(seg.text.strip() for seg in segments)
        result = self.whisper_model.transcribe(path, fp16=False)
        return result.get("text") or ""

    def extract_from_url(self, url):
        """Extract the main article text (with its headline) from a URL."""
        return self.extract_from_url_with_meta(url)[0]

    def extract_from_url_with_meta(self, url):
        """(article text, {title, date, sitename, author, url}) for a URL."""
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            raise ValueError("Could not download the page (blocked, offline, or invalid URL).")
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if not text:
            raise ValueError("The page loaded but no article text could be found on it.")
        metadata = trafilatura.extract_metadata(downloaded)
        title = getattr(metadata, "title", None) if metadata else None
        if title and title not in text[:300]:
            text = f"{title}. {text}"
        meta = {
            "url": url,
            "title": title or "",
            "date": (getattr(metadata, "date", None) or "") if metadata else "",
            "sitename": (getattr(metadata, "sitename", None) or "") if metadata else "",
            "author": (getattr(metadata, "author", None) or "") if metadata else "",
        }
        return text.strip(), meta

    def extract_from_image(self, uploaded_file, lang="eng", preprocess=True):
        """Extract text from an image using Tesseract OCR."""
        if not self.ocr_available:
            raise RuntimeError(
                "Image OCR needs the Tesseract program, which is not installed. "
                "Install it (Windows: https://github.com/UB-Mannheim/tesseract/wiki) "
                "or set TESSERACT_CMD to tesseract.exe, then restart the app."
            )
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")  # RGBA / palette / grayscale / rotated photos

        if preprocess:
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            # Upscale small screenshots; Tesseract works best around 300 DPI.
            if gray.shape[1] < 1200:
                scale = 1200 / gray.shape[1]
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            # Dark-mode screenshots: invert so text is dark on light.
            if gray.mean() < 110:
                gray = cv2.bitwise_not(gray)
            _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image = Image.fromarray(threshold)

        text = pytesseract.image_to_string(image, lang=lang)
        text = " ".join(text.split())
        if not text:
            raise ValueError("No readable text was detected in the image.")
        return text

    def extract_from_video(self, uploaded_file):
        """Transcribe speech from a video using Whisper."""
        if not self.video_available:
            raise RuntimeError(
                f"Video transcription is unavailable because {self.video_unavailable_reason}. "
                "Install `faster-whisper` (pip) to enable it, or paste the transcript as Text instead."
            )
        suffix = os.path.splitext(getattr(uploaded_file, "name", "") or "")[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read())
            temp_path = temp_file.name
        try:
            text = self._transcribe(temp_path).strip()
        except Exception as e:
            raise ValueError(f"Could not transcribe the video's audio ({e}).") from e
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        if not text:
            raise ValueError("No speech was detected in the video.")
        return text

    def preprocess_text(self, text):
        """Normalise whitespace and strip control/odd symbols, keeping normal punctuation."""
        if not text:
            return ""
        text = " ".join(text.split())
        text = re.sub(r"[^\w\s.,;:!?'\"%$€£()\-–—/&@#]", "", text)
        return text.strip()
