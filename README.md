---
title: Fake News Detector
emoji: 📰
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Checks news claims against fact-checkers, news and Wikipedia
---

# Fake News Detector

Paste a headline, an article, a link, a screenshot or a video clip. The app pulls out the factual claims and checks each one against live sources, then explains its verdict with links.

## How it works

1. **Extract text** from what you paste, a URL (article extraction), an image (Tesseract OCR) or a video (faster-whisper speech-to-text).
2. **Find the claims.** The most check-worthy factual sentences are picked out.
3. **Check each claim against sources**, with no API key needed:
   - news headlines from Google News,
   - fact-checker coverage (AFP, Snopes, PolitiFact, Full Fact, …),
   - Wikipedia,
   - optionally, published ratings from the Google Fact Check Tools API.
4. **Judge the evidence.** Two small transformer models run locally with ONNX Runtime:
   - MiniLM sentence embeddings find the sources that are really about the claim,
   - a DeBERTa-v3 NLI model decides whether each source supports or contradicts it.

   Each claim is marked Supported, Contradicted, Disputed, Fact-checked or Not enough evidence.
5. **Verdict.** Evidence decides when there is any. Otherwise a writing-style classifier is the fallback, and the app says so.

Outlets are labelled (fact-checker, established outlet, social media, satire, known unreliable), and linked articles get a warning when they're old or from a satire site.

## Writing-style classifier

Logistic regression on TF-IDF features plus writing-style signals (all-caps words, sensational phrases, missing attribution), trained on [pushpdeep/fake_news_combined](https://huggingface.co/datasets/pushpdeep/fake_news_combined) (~45k labelled articles).

| Test | Result |
|---|---|
| Held-out test set (7,818 articles) | 98.7% |
| Hand-written real-world headlines | 52/52 confident calls correct, 8 of 60 marked uncertain |
| Style-swapped cases (true facts written casually, false claims written like wire news) | 0/13 |

The last row is the classifier's limit: it judges wording, not facts. That's why the source check, not the classifier, decides the verdict whenever sources cover the claim. You can grade results in the app ("Was this right?") and retrain the classifier with your grades.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env        # optional keys, see below
streamlit run app.py
```

The first launch downloads the training data and models (~180 MB) and trains the classifier, which takes about 3 minutes once. `python train_model.py` does the same ahead of time.

For image OCR, install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) (on Windows the default install path is detected automatically).

### Optional settings (`.env`)

| Variable | What it adds |
|---|---|
| `GOOGLE_API_KEY` | Published fact-check ratings (Google Fact Check Tools API, free) |
| `GOOGLE_CX` | Extra web results (Google Custom Search; needs `GOOGLE_API_KEY`) |
| `WIKIMEDIA_CONTACT` | A URL or email sent to Wikipedia to identify the app, which raises its rate limits |

## Deployment

Serverless platforms such as Vercel can't host this app: Streamlit needs a long-running server with websocket connections.

### Streamlit Community Cloud (free)

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub and click **Create app**.
2. Pick this repository, branch `main`, main file `app.py`. Under **Advanced settings**, choose Python 3.12.
3. In **Secrets**, add your optional keys in TOML format:
   ```toml
   GOOGLE_API_KEY = "your-key"
   WIKIMEDIA_CONTACT = "https://github.com/your-name"
   ```
4. Click **Deploy**.

The trained classifier (`models/`, ~6 MB) is committed, so the server never has to train it. The evidence models (~110 MB) download on first start. `packages.txt` installs Tesseract for image OCR. The app uses about 650 MB of memory.

### Docker (Render, Railway, Fly.io, Hugging Face Spaces)

The included `Dockerfile` trains the classifier and downloads the models at build time, then serves the app on port 7860. Hugging Face Docker Spaces now need a paid plan; `.github/workflows/sync-to-hf.yml` syncs `main` to a Space when the `HF_SPACE` variable and `HF_TOKEN` secret are set in the repository.
