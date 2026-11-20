"""Fake-news classifier trained on a real labelled dataset.

Training data
    pushpdeep/fake_news_combined (Hugging Face, Apache-2.0): ~45k news
    articles (ISOT corpus: Reuters wire copy vs. articles from sites flagged
    as unreliable), plus short hand-written headline examples
    (components/training_data.py) so headline-length inputs are covered.

Leakage handling
    In ISOT, real news is almost all Reuters copy and fake news often has
    "Featured image via Getty" credits, so a naive model learns publisher
    names instead of content. ``clean_text`` strips those artifacts both at
    training and prediction time.

Accuracy reporting
    * test-set metrics on a held-out 20% split of the dataset (in-distribution,
      optimistic), and
    * accuracy on a separate hand-written real-world set
      (components/realworld_eval.py) that was never used for training.
    Live accuracy from user feedback is tracked in components/accuracy_tracker.py.
"""

from __future__ import annotations

import os
import re
import time

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from components import realworld_eval
from components.training_data import FAKE_SAMPLES, REAL_SAMPLES

MODEL_VERSION = 4  # 4: calibrated confidence (temperature scaling) + feedback training
MODEL_PATH = os.path.join("models", "fake_news_model.joblib")
DATASET_NAME = "pushpdeep/fake_news_combined"
DATASET_URL = "https://huggingface.co/api/datasets/pushpdeep/fake_news_combined/parquet/default/train/0.parquet"
DATASET_PATH = os.path.join("data", "datasets", "fake_news_combined.parquet")

# Below this probability for the winning class, the verdict is "Uncertain".
UNCERTAIN_THRESHOLD = 0.65
SEED_WEIGHT = 25.0  # each hand-written headline counts like this many articles
FEEDBACK_WEIGHT = 15.0  # each item graded in the app counts like this many articles
MAX_CONFIDENCE = 0.99  # never claim certainty
MODEL_LABEL = "Logistic regression (TF-IDF)"
MIN_WORDS_FOR_VERDICT = 5  # fewer words than this → always "Uncertain"

# --------------------------------------------------------------------------
# Text cleaning (removes source/publisher artifacts the model could cheat on)
# --------------------------------------------------------------------------
_DATELINE_RE = re.compile(r"^[^()\n]{0,80}\((?:reuters|ap|afp)\)\s*[-–—]\s*", re.IGNORECASE)  # "WASHINGTON (Reuters) - "
# (substrings that must be present in the lowercased text, pattern). The cheap
# substring check skips the regex for most texts and keeps cleaning ~45k
# articles fast.
_ARTIFACTS = [
    (("(reuters)", "(ap)", "(afp)"), r"\((?:reuters|ap|afp)\)"),
    (("reuters",), r"\b(?:thomson )?reuters\b"),
    (("http", "www.", "pic.twitter"), r"https?://[^\s]+|www\.[^\s]+|pic\.twitter\.com/[^\s]+"),
    ((".com", ".org", ".net", ".co.uk"), r"\b[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|co\.uk)\b"),
    (("@",), r"@\w+"),
    (("featured image",), r"\bfeatured image\b[^.\n]*"),
    (("photo by", "photo via", "photo credit", "image by", "image via", "image credit"),
     r"\b(?:photo|image) (?:by|via|credit)\b[^.\n]*"),
    (("getty",), r"\bgetty(?: images?)?\b"),
    (("21st century wire", "21wire"), r"\b21st century wire\b|\b21wire\b"),
    (("video]", "video)", "videos]", "videos)", "tweet]", "tweet)", "tweets]", "tweets)", "image]", "image)",
      "images]", "images)", "watch]", "watch)", "details]", "details)"),
     r"[\[(](?:video|videos|tweets?|images?|watch|details)[\])]"),
]
_ARTIFACTS = [(gates, re.compile(p, re.IGNORECASE)) for gates, p in _ARTIFACTS]


def clean_text(text: str) -> str:
    text = str(text).replace("�", "'").replace("’", "'").replace("‘", "'")
    low = text.lower()
    if "(reuters)" in low or "(ap)" in low or "(afp)" in low:
        text = _DATELINE_RE.sub(" ", text, count=1)
    for gates, rx in _ARTIFACTS:
        if any(g in low for g in gates):
            text = rx.sub(" ", text)
    return " ".join(text.split())


# --------------------------------------------------------------------------
# Interpretable writing-style signals
# --------------------------------------------------------------------------
SENSATIONAL_TERMS = [
    "shocking", "you won't believe", "won't believe", "miracle", "secret", "exposed",
    "they don't want you", "doesn't want you to know", "don't want you to know",
    "what they don't tell you", "hidden", "hiding", "cover-up", "covering it up",
    "covering up", "wake up", "share before", "before they delete", "banned",
    "mainstream media", "big pharma", "hoax", "cures", "instantly", "overnight",
    "stunned", "baffled", "speechless", "furious", "doctors hate", "whistleblower",
    "leaked", "insider", "explosive", "truth about", "100% proof", "click here",
    "forward this", "share now", "share to", "unbelievable", "they lied",
    "new world order", "mind control", "microchip", "depopulate", "plandemic",
    "nobody is talking", "media won't", "act now", "urgent", "breaking!!!",
]

ATTRIBUTION_TERMS = [
    "according to", "said", "says", "told reporters", "in a statement", "spokesperson",
    "spokesman", "officials", "official", "reported", "announced", "confirmed",
    "published", "peer-reviewed", "study", "survey", "data", "researchers", "ministry",
    "agency", "department", "court", "percent", "%", "quarter", "statistics", "analysts",
]

STYLE_FEATURE_NAMES = [
    "ALL-CAPS words", "exclamation marks", "sensational phrases",
    "attribution / sourcing", "specific numbers", "clickbait hooks",
]

_CAPS_OK = {"USA", "NASA", "COVID", "WHO", "FBI", "CIA", "GDP", "BBC", "CNN", "ISRO", "RBI", "UNESCO", "MIT", "JAMA", "EU", "UK", "AI"}


def _terms_regex(terms: list[str]) -> re.Pattern:
    # One alternation, longest terms first, so "cures" wins over "cure" and
    # "you won't believe" over "won't believe" (matches don't overlap).
    alts = "|".join(re.escape(t) for t in sorted(set(terms), key=len, reverse=True))
    return re.compile(r"(?<![a-z])(?:" + alts + r")(?![a-z])")


_SENSATIONAL_RE = _terms_regex(SENSATIONAL_TERMS)
_ATTRIBUTION_RE = _terms_regex(ATTRIBUTION_TERMS)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_NUMBER_RE = re.compile(r"\b\d[\d,.]*\s*(?:%|percent|million|billion|crore|lakh|tonnes|votes)?")
_HOOK_RE = re.compile(r"\(watch|\(click|click to see|see the photo|what happened next|register on this link")


def _find_terms(text_lower: str, pattern: re.Pattern) -> list[str]:
    return list(dict.fromkeys(pattern.findall(text_lower)))  # unique, in order of appearance


def style_signals(text: str) -> dict:
    head = text[:1500]  # style cues live in the headline / lead
    words = _WORD_RE.findall(head)
    n_words = max(len(words), 1)
    caps = [w for w in words if len(w) >= 3 and w.isupper() and w not in _CAPS_OK]
    lower = head.lower()
    return {
        "caps_words": caps,
        "caps_ratio": len(caps) / n_words,
        "exclamations": head.count("!"),
        "sensational": _find_terms(lower, _SENSATIONAL_RE),
        "attribution": _find_terms(lower, _ATTRIBUTION_RE),
        "numbers": _NUMBER_RE.findall(head),
        "hooks": _HOOK_RE.findall(lower),
    }


class StyleFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            s = style_signals(text)
            rows.append([
                min(s["caps_ratio"] * 5, 1.0),
                min(s["exclamations"], 3) / 3,
                min(len(s["sensational"]), 3) / 3,
                min(len(s["attribution"]), 3) / 3,
                min(len(s["numbers"]), 3) / 3,
                min(len(s["hooks"]), 1),
            ])
        return csr_matrix(np.asarray(rows, dtype=float) * 2.0)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(STYLE_FEATURE_NAMES)


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("features", FeatureUnion([
            ("words", TfidfVectorizer(
                lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True,
                min_df=3, max_df=0.9, max_features=300_000, dtype=np.float32,
            )),
            ("style", StyleFeatures()),
        ])),
        # liblinear: ~4x faster than lbfgs here with the same accuracy.
        ("clf", LogisticRegression(C=4.0, solver="liblinear", class_weight="balanced")),
    ])


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def download_dataset(progress=print) -> str:
    if os.path.exists(DATASET_PATH) and os.path.getsize(DATASET_PATH) > 1_000_000:
        return DATASET_PATH
    import requests

    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    progress(f"Downloading training dataset {DATASET_NAME} (~70 MB)…")
    tmp = DATASET_PATH + ".part"
    with requests.get(DATASET_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(tmp, DATASET_PATH)
    return DATASET_PATH


SHORT_TEXT_WORDS = 40  # inputs shorter than this get proportionally softer confidence


def calibrated_proba(pipeline: Pipeline, texts: list[str], temperature: float = 1.0) -> np.ndarray:
    """Class probabilities (columns in clf.classes_ order) with temperature scaling.

    Temperature is never below 1 (the held-out split is in-distribution and
    would otherwise teach the model to be *more* confident), and short inputs
    are softened: a 10-word headline carries far less signal than an article.
    """
    z = np.asarray(pipeline.decision_function(texts), dtype=float) / max(temperature, 1.0)
    n_words = np.asarray([len(t.split()) for t in texts], dtype=float)
    z *= np.clip(0.4 + 0.6 * n_words / SHORT_TEXT_WORDS, 0.4, 1.0)
    p1 = 1.0 / (1.0 + np.exp(-z))
    p1 = np.clip(p1, 1 - MAX_CONFIDENCE, MAX_CONFIDENCE)
    return np.column_stack([1 - p1, p1])


def fit_temperature(pipeline: Pipeline, texts: list[str], labels: list[str]) -> float:
    """Temperature that minimises log-loss on held-out data (simple grid search)."""
    classes = list(pipeline.named_steps["clf"].classes_)
    y = np.asarray([1.0 if lab == classes[1] else 0.0 for lab in labels])
    z = np.asarray(pipeline.decision_function(texts), dtype=float)
    best_t, best_loss = 1.0, float("inf")
    for t in np.linspace(1.0, 8.0, 71):  # never sharpen
        p = np.clip(1.0 / (1.0 + np.exp(-z / t)), 1e-6, 1 - 1e-6)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if loss < best_loss:
            best_t, best_loss = float(t), loss
    return best_t


def _score_set(pipeline: Pipeline, real: list[str], fake: list[str], prefix: str, temperature: float = 1.0) -> dict:
    texts = real + fake
    truth = ["Real"] * len(real) + ["Fake"] * len(fake)
    proba = calibrated_proba(pipeline, [clean_text(t) for t in texts], temperature)
    classes = list(pipeline.named_steps["clf"].classes_)
    correct = wrong = uncertain = 0
    for p, t in zip(proba, truth):
        if p.max() < UNCERTAIN_THRESHOLD:
            uncertain += 1
        elif classes[int(p.argmax())] == t:
            correct += 1
        else:
            wrong += 1
    decided = correct + wrong
    return {
        f"{prefix}_accuracy": correct / decided * 100 if decided else None,
        f"{prefix}_correct": correct,
        f"{prefix}_wrong": wrong,
        f"{prefix}_uncertain": uncertain,
        f"{prefix}_n": len(texts),
    }


def evaluation_metrics(pipeline: Pipeline, temperature: float = 1.0) -> dict:
    """Accuracy on the hand-written sets that were never used for training."""
    m = _score_set(pipeline, realworld_eval.REAL, realworld_eval.FAKE, "realworld", temperature)
    m.update(_score_set(pipeline, realworld_eval.HARD_REAL, realworld_eval.HARD_FAKE, "hard", temperature))
    return m


def describe_metrics(m: dict) -> str:
    def frac(prefix):
        return f"{m[prefix + '_correct']}/{m[prefix + '_correct'] + m[prefix + '_wrong']}"

    return (
        f"Test split: {m['n_test']:,} held-out articles from {m['dataset']}. "
        f"Real-world headlines: {frac('realworld')} confident calls correct "
        f"({m['realworld_uncertain']} of {m['realworld_n']} marked uncertain). "
        f"Style-swapped hard cases (true facts written casually, false claims in news-wire style): "
        f"{frac('hard')} correct ({m['hard_uncertain']} of {m['hard_n']} uncertain)."
    )


def train_model(progress=print, feedback: list[dict] | None = None) -> dict:
    """Train on the real dataset (falls back to seed data if it can't be downloaded).

    feedback: rows graded in the app ({"text", "actual": "Real"|"Fake"}); they are
    added to the training data with extra weight.
    """
    t0 = time.time()
    seed_X = list(FAKE_SAMPLES) + list(REAL_SAMPLES)
    seed_y = ["Fake"] * len(FAKE_SAMPLES) + ["Real"] * len(REAL_SAMPLES)
    fb = [(clean_text(r.get("text") or r.get("snippet") or ""), r.get("actual")) for r in (feedback or [])]
    fb = [(t, y) for t, y in fb if len(t.split()) >= 3 and y in ("Real", "Fake")]

    try:
        import pandas as pd

        path = download_dataset(progress)
        progress("Cleaning dataset (removing duplicates and publisher artifacts)…")
        df = pd.read_parquet(path, columns=["text", "label"]).dropna()
        df["text"] = df["text"].astype(str).map(clean_text)
        df = df[df["text"].str.len() >= 40].drop_duplicates("text")
        df["y"] = np.where(df["label"].astype(int) == 0, "Fake", "Real")  # dataset: 0 = fake, 1 = real
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"].tolist(), df["y"].tolist(), test_size=0.2, stratify=df["y"], random_state=42
        )
        source = DATASET_NAME
    except Exception as e:  # offline / download blocked
        progress(f"Dataset unavailable ({e}); training on built-in examples only.")
        X_train, X_test, y_train, y_test = train_test_split(
            seed_X, seed_y, test_size=0.3, stratify=seed_y, random_state=42
        )
        seed_X, seed_y = [], []
        source = "built-in examples only (dataset download failed)"

    # Hold out a calibration split (articles + their headline-length leads) to
    # fit the confidence temperature, so short inputs don't come out at "100%".
    calib_X, calib_y = [], []
    if seed_X and len(X_train) > 2000:
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train, random_state=7
        )
        calib_X = list(X_cal) + [" ".join(t.split()[:22]) for t in X_cal]
        calib_y = list(y_cal) * 2

    # Headline-style augmentation: the lead of each article ≈ its headline.
    heads, head_y = [], []
    if seed_X:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), size=min(len(X_train) // 3, 12_000), replace=False)
        heads = [" ".join(X_train[i].split()[:22]) for i in idx]
        head_y = [y_train[i] for i in idx]

    fb_X = [t for t, _ in fb]
    fb_y = [y for _, y in fb]
    train_X = list(X_train) + heads + seed_X + fb_X
    train_y = list(y_train) + head_y + seed_y + fb_y
    weights = np.concatenate([
        np.ones(len(X_train) + len(heads)), np.full(len(seed_X), SEED_WEIGHT), np.full(len(fb_X), FEEDBACK_WEIGHT),
    ])
    if fb_X:
        progress(f"Including {len(fb_X)} item(s) you graded in the app.")

    progress(f"Training classifier on {len(train_X):,} examples…")
    pipeline = build_pipeline()
    pipeline.fit(train_X, train_y, clf__sample_weight=weights)

    temperature = fit_temperature(pipeline, calib_X, calib_y) if calib_X else 1.0
    progress(f"Calibrated confidence (temperature {temperature:.2f}).")

    progress("Measuring accuracy on held-out test data…")
    y_pred = pipeline.predict(X_test)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=["Fake"], zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=["Fake", "Real"])
    metrics = {
        "dataset": source,
        "accuracy": accuracy_score(y_test, y_pred) * 100,
        "precision_fake": float(prec[0]) * 100,
        "recall_fake": float(rec[0]) * 100,
        "f1_fake": float(f1[0]) * 100,
        "confusion": cm.tolist(),
        "n_train": len(train_X),
        "n_feedback": len(fb_X),
        "temperature": temperature,
        "n_test": len(X_test),
        "train_seconds": round(time.time() - t0, 1),
    }
    metrics.update(evaluation_metrics(pipeline, temperature))
    metrics["description"] = describe_metrics(metrics)
    bundle = {"version": MODEL_VERSION, "pipeline": pipeline, "metrics": metrics, "temperature": temperature}
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    tmp = MODEL_PATH + ".tmp"
    joblib.dump(bundle, tmp, compress=3)
    os.replace(tmp, MODEL_PATH)  # atomic: a crash mid-save never leaves a broken model
    with open(MODEL_PATH + ".version", "w", encoding="utf-8") as f:
        f.write(str(MODEL_VERSION))
    progress(f"Model trained in {metrics['train_seconds']}s. Test accuracy {metrics['accuracy']:.1f}%.")
    return bundle


# --------------------------------------------------------------------------
# Predictor used by the app
# --------------------------------------------------------------------------
def model_needs_training() -> bool:
    """True when there's no saved model, or it was saved by an older version of this code."""
    if not os.path.exists(MODEL_PATH):
        return True
    try:
        with open(MODEL_PATH + ".version", encoding="utf-8") as f:
            return f.read().strip() != str(MODEL_VERSION)
    except OSError:
        return True  # saved before version files existed


class MLPredictor:
    def __init__(self, progress=None):
        self.pipeline: Pipeline | None = None
        self.metrics: dict = {}
        self.temperature = 1.0
        self.model_loaded = False
        self.initialize_model(progress or (lambda msg: None))

    @property
    def model_accuracy(self):
        return self.metrics.get("accuracy")

    @property
    def model_source(self):
        return self.metrics.get("dataset", "")

    @property
    def model_name(self):
        return "TF-IDF + writing-style signals, logistic regression"

    def initialize_model(self, progress):
        bundle = None
        if os.path.exists(MODEL_PATH):
            try:
                bundle = joblib.load(MODEL_PATH)
                if bundle.get("version") != MODEL_VERSION:
                    bundle = None
            except Exception:
                bundle = None
        if bundle is None:
            bundle = train_model(progress)
        self.pipeline = bundle["pipeline"]
        self.metrics = dict(bundle["metrics"])
        self.temperature = float(bundle.get("temperature", 1.0))
        # Re-score the evaluation sets on every load (fast) so the reported
        # numbers always match the current sets and model.
        self.metrics.update(evaluation_metrics(self.pipeline, self.temperature))
        self.metrics["description"] = describe_metrics(self.metrics)
        self.model_loaded = True

    def available_models(self):
        # Only models that actually exist. (Evidence checking uses separate
        # transformer models; see components/nlp_models.py.)
        return [MODEL_LABEL]

    def _contributions(self, text: str, top_n: int = 6):
        """Signed per-feature contributions. Positive pushes towards 'Real'."""
        assert self.pipeline is not None
        features = self.pipeline.named_steps["features"]
        clf = self.pipeline.named_steps["clf"]
        vec = features.transform([text]).tocsr()
        names = features.get_feature_names_out()
        sign = 1.0 if clf.classes_[1] == "Real" else -1.0
        coefs = clf.coef_[0] * sign
        contribs = []
        for idx, val in zip(vec.indices, vec.data):
            c = float(coefs[idx] * val)
            if abs(c) > 1e-3:
                contribs.append((str(names[idx]).split("__", 1)[-1], c))
        contribs.sort(key=lambda x: abs(x[1]), reverse=True)
        return contribs[:top_n]

    def predict(self, text, model_choice=MODEL_LABEL):
        if not self.model_loaded or self.pipeline is None:
            raise RuntimeError("Model not loaded. Cannot make predictions.")

        cleaned = clean_text(text)
        proba = calibrated_proba(self.pipeline, [cleaned], self.temperature)[0]
        classes = list(self.pipeline.named_steps["clf"].classes_)
        probs = {c: float(p) for c, p in zip(classes, proba)}
        p_real, p_fake = probs.get("Real", 0.0), probs.get("Fake", 0.0)
        confidence = max(p_real, p_fake)

        n_words = len(cleaned.split())
        if confidence < UNCERTAIN_THRESHOLD or n_words < MIN_WORDS_FOR_VERDICT:
            label, as_bool, raw = "Uncertain", None, "Uncertain"
        elif p_real > p_fake:
            label, as_bool, raw = "True", True, "Real"
        else:
            label, as_bool, raw = "False", False, "Fake"

        s = style_signals(cleaned)
        red_flags, credibility = [], []
        if s["caps_words"]:
            red_flags.append(f"ALL-CAPS words: {', '.join(s['caps_words'][:5])}")
        if s["exclamations"] >= 2:
            red_flags.append(f"{s['exclamations']} exclamation marks")
        if s["sensational"]:
            red_flags.append("Sensational phrases: " + ", ".join(f"“{t}”" for t in s["sensational"][:6]))
        if s["hooks"]:
            red_flags.append("Clickbait hook (e.g. “watch”, “click to see”)")
        if s["attribution"]:
            credibility.append("Cites sources / attribution: " + ", ".join(f"“{t}”" for t in s["attribution"][:6]))
        else:
            red_flags.append("No attribution (no “according to”, “said”, official source…)")
        if s["numbers"]:
            credibility.append("Specific figures: " + ", ".join(n.strip() for n in s["numbers"][:5]))

        notes = []
        if n_words < MIN_WORDS_FOR_VERDICT:
            notes.append(f"Too short to judge ({n_words} word{'s' if n_words != 1 else ''}). Paste a full headline or a few sentences.")
        elif n_words < 8:
            notes.append("Very short input: headline-only predictions are less reliable. Paste more of the article if you can.")

        m = self.metrics
        return {
            "prediction": label,
            "prediction_label": label,
            "prediction_bool": as_bool,
            "raw_prediction": raw,
            "confidence": confidence * 100,
            "probabilities": {"False": p_fake * 100, "True": p_real * 100},
            "accuracy": m.get("accuracy"),
            "realworld_accuracy": m.get("realworld_accuracy"),
            "accuracy_note": m.get("description", ""),
            "top_features": self._contributions(cleaned),
            "red_flags": red_flags,
            "credibility_signals": credibility,
            "notes": notes,
            "model_used": f"{self.model_name}, trained on {self.model_source}",
            "model_choice": model_choice,
        }
