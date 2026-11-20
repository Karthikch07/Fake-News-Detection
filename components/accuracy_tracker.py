"""Live ("real-time") accuracy tracking from user feedback.

Every analyzed item can be graded by the user ("the model was right / wrong").
Grades are appended to a local JSONL file so the live accuracy survives
restarts. This complements the offline test-set accuracy measured at
training time: the test-set number says how the model does on the dataset,
the live number says how it does on what people actually paste in.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass

FEEDBACK_PATH = os.path.join("data", "feedback.jsonl")
_LOCK = threading.Lock()


@dataclass
class LiveStats:
    graded: int
    correct: int
    recent_n: int
    recent_correct: int
    uncertain_graded: int
    tp: int  # predicted fake, actually fake
    fp: int  # predicted fake, actually real
    tn: int  # predicted real, actually real
    fn: int  # predicted real, actually fake

    @property
    def accuracy(self) -> float | None:
        decided = self.graded - self.uncertain_graded
        return self.correct / decided * 100 if decided else None

    @property
    def recent_accuracy(self) -> float | None:
        return self.recent_correct / self.recent_n * 100 if self.recent_n else None

    @property
    def fake_precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) * 100 if (self.tp + self.fp) else None

    @property
    def fake_recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) * 100 if (self.tp + self.fn) else None


def record_feedback(result_id: str, predicted: str, actual: str, text: str, confidence: float | None) -> None:
    """predicted: 'True' | 'False' | 'Uncertain'; actual: 'Real' | 'Fake'.

    The full text is kept (up to 20k characters) so graded items can be used
    to retrain the classifier.
    """
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    row = {
        "id": result_id,
        "predicted": predicted,
        "actual": actual,
        "confidence": confidence,
        "snippet": text[:200],
        "text": text[:20000],
    }
    with _LOCK, open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load() -> list[dict]:
    if not os.path.exists(FEEDBACK_PATH):
        return []
    rows: dict[str, dict] = {}
    with open(FEEDBACK_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[row.get("id", str(len(rows)))] = row  # last grade per item wins
    return list(rows.values())


def feedback_rows() -> list[dict]:
    """All graded items (last grade per item), for retraining."""
    return _load()


def graded_ids() -> dict[str, str]:
    return {r["id"]: r["actual"] for r in _load() if "id" in r}


def live_stats(recent_window: int = 20) -> LiveStats:
    rows = _load()
    correct = tp = fp = tn = fn = unc = 0
    outcomes: list[bool] = []
    for r in rows:
        pred, actual = r.get("predicted"), r.get("actual")
        if pred == "Uncertain":
            unc += 1
            continue
        pred_fake = pred == "False"
        is_fake = actual == "Fake"
        ok = pred_fake == is_fake
        outcomes.append(ok)
        correct += ok
        tp += pred_fake and is_fake
        fp += pred_fake and not is_fake
        tn += (not pred_fake) and (not is_fake)
        fn += (not pred_fake) and is_fake
    recent = outcomes[-recent_window:]
    return LiveStats(
        graded=len(rows), correct=correct, recent_n=len(recent), recent_correct=sum(recent),
        uncertain_graded=unc, tp=tp, fp=fp, tn=tn, fn=fn,
    )


def reset_feedback() -> None:
    with _LOCK:
        if os.path.exists(FEEDBACK_PATH):
            os.remove(FEEDBACK_PATH)
