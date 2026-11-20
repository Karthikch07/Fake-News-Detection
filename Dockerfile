# Fake News Detector: container for Hugging Face Spaces (also works on Render, Railway, Fly.io).
FROM python:3.12-slim

# Tesseract for image OCR; libgomp for onnxruntime / scikit-learn.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs the container as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    TESSERACT_CMD=/usr/bin/tesseract
# Create the app folder as "user" (WORKDIR alone would make it root-owned, and
# the app writes its model, cache and history inside it).
RUN mkdir -p /home/user/app
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Do the slow first-run work at build time so the app starts instantly:
# train the classifier (~45k articles) and download the evidence models.
RUN python train_model.py \
 && python -c "from components.nlp_models import get_embedder, get_nli; get_embedder(); get_nli()" \
 && rm -rf data/datasets

EXPOSE 7860
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableXsrfProtection=false", "--browser.gatherUsageStats=false"]
