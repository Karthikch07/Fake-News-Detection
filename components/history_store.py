"""Analysis history saved to disk so it survives page reloads and restarts."""

from __future__ import annotations

import json
import os
import threading

HISTORY_PATH = os.path.join("data", "history.json")
MAX_ITEMS = 50
_LOCK = threading.Lock()


def load_history() -> list[dict]:
    try:
        with open(HISTORY_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(history: list[dict]) -> None:
    with _LOCK:
        try:
            os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
            tmp = HISTORY_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(history[:MAX_ITEMS], f, ensure_ascii=False, default=str)
            os.replace(tmp, HISTORY_PATH)
        except Exception:
            pass  # history is a convenience; never break an analysis over it


def clear_history() -> None:
    with _LOCK:
        if os.path.exists(HISTORY_PATH):
            os.remove(HISTORY_PATH)
