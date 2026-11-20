"""Small transformer models run locally with onnxruntime (no PyTorch needed).

* ``Embedder``: sentence-transformers/all-MiniLM-L6-v2 (ONNX export by Xenova),
  used to find the evidence passages most related to a claim.
* ``NLIModel``: cross-encoder/nli-deberta-v3-xsmall (ONNX export by Xenova),
  a natural-language-inference model that says whether a passage
  *entails*, *contradicts* or is *neutral* towards a claim.

Model files are downloaded from the Hugging Face Hub on first use
(~23 MB + ~87 MB) and cached in the usual Hugging Face cache folder.
"""

from __future__ import annotations

import json
import os
import threading

import numpy as np

EMBED_REPO = "Xenova/all-MiniLM-L6-v2"
NLI_REPO = "Xenova/nli-deberta-v3-xsmall"
ONNX_FILE = "onnx/model_quantized.onnx"


def _download(repo: str) -> tuple[str, str, str]:
    from huggingface_hub import hf_hub_download

    onnx_path = hf_hub_download(repo, ONNX_FILE)
    tok_path = hf_hub_download(repo, "tokenizer.json")
    cfg_path = hf_hub_download(repo, "config.json")
    return onnx_path, tok_path, cfg_path


def _session(path: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 2)))
    opts.log_severity_level = 3
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])


class _OnnxModel:
    max_length = 256

    def __init__(self, repo: str):
        from tokenizers import Tokenizer

        onnx_path, tok_path, cfg_path = _download(repo)
        self.session = _session(onnx_path)
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(tok_path)
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding()
        with open(cfg_path, encoding="utf-8") as f:
            self.config = json.load(f)
        self._lock = threading.Lock()  # tokenizer padding state is shared

    def _feeds(self, encodings) -> dict:
        feeds = {
            "input_ids": np.asarray([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.asarray([e.attention_mask for e in encodings], dtype=np.int64),
        }
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.asarray([e.type_ids for e in encodings], dtype=np.int64)
        return feeds


class Embedder(_OnnxModel):
    max_length = 128

    def __init__(self):
        super().__init__(EMBED_REPO)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            with self._lock:
                enc = self.tokenizer.encode_batch(texts[i:i + batch_size])
            feeds = self._feeds(enc)
            hidden = self.session.run(None, feeds)[0]  # (batch, seq, dim)
            mask = feeds["attention_mask"][..., None].astype(np.float32)
            pooled = (hidden * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
            out.append(pooled)
        return np.vstack(out) if out else np.zeros((0, 384), dtype=np.float32)


class NLIModel(_OnnxModel):
    def __init__(self):
        super().__init__(NLI_REPO)
        id2label = {int(k): v.lower() for k, v in self.config["id2label"].items()}
        self.labels = [id2label[i] for i in range(len(id2label))]

    def predict(self, premises: list[str], hypothesis: str, batch_size: int = 16) -> list[dict]:
        """Probabilities that each premise entails / contradicts / is neutral to the hypothesis."""
        return self._pairs([(p, hypothesis) for p in premises], batch_size)

    def predict_reverse(self, premise: str, hypotheses: list[str], batch_size: int = 16) -> list[dict]:
        """Same, with one premise and several hypotheses."""
        return self._pairs([(premise, h) for h in hypotheses], batch_size)

    def _pairs(self, all_pairs: list[tuple[str, str]], batch_size: int) -> list[dict]:
        results = []
        for i in range(0, len(all_pairs), batch_size):
            pairs = all_pairs[i:i + batch_size]
            with self._lock:
                enc = self.tokenizer.encode_batch(pairs)
            logits = self.session.run(None, self._feeds(enc))[0]
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
            results.extend({lab: float(p[j]) for j, lab in enumerate(self.labels)} for p in probs)
        return results


_models: dict[str, object] = {}
_models_lock = threading.Lock()


def get_embedder() -> Embedder:
    with _models_lock:
        if "embed" not in _models:
            _models["embed"] = Embedder()
        return _models["embed"]  # type: ignore[return-value]


def get_nli() -> NLIModel:
    with _models_lock:
        if "nli" not in _models:
            _models["nli"] = NLIModel()
        return _models["nli"]  # type: ignore[return-value]
