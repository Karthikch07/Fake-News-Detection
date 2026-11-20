"""Fact verification: claim-by-claim evidence check plus optional search APIs.

The claim check (components/evidence.py) needs no API keys: it searches
Google News RSS and Wikipedia and judges each passage with local
transformer models. Google Custom Search, Bing and the Google Fact Check
Tools API are used in addition when their keys are set in .env.
"""

from __future__ import annotations

import os
import re
from urllib.parse import quote

import requests

from components import evidence
from components import source_reputation as rep


_TRUE_RATINGS = re.compile(r"^\s*(true|correct|accurate|mostly true|mostly correct|verdadero|vrai)\b", re.I)
_FALSE_RATINGS = re.compile(
    r"\b(false|fake|pants on fire|misleading|incorrect|inaccurate|no evidence|not true|untrue|baseless|"
    r"fabricated|hoax|scam|altered|doctored|manipulated|satire|unproven|unsupported|debunked|wrong|"
    r"exaggerat\w*|distort\w*|missing context|out of context|lacks context|partly false|mostly false|"
    r"no scientific|not supported|there is no|there's no|we've seen no|faux|falso)\b",
    re.I,
)


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()[:80]


def rating_stance(rating: str) -> str:
    """Map a fact-checker's textual rating to a stance on the claim it reviewed."""
    if _FALSE_RATINGS.search(rating or ""):
        return "contradicts"
    if _TRUE_RATINGS.search(rating or ""):
        return "supports"
    return "neutral"


class FactVerifier:
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "")
        self.google_cx = os.getenv("GOOGLE_CX", "")
        self.bing_api_key = os.getenv("BING_API_KEY", "")
        self.models_error = ""
        self.embedder = self.nli = None
        try:
            from components.nlp_models import get_embedder, get_nli

            self.embedder = get_embedder()
            self.nli = get_nli()
        except Exception as e:  # offline on first run, or onnxruntime missing
            self.models_error = f"{type(e).__name__}: {e}"

    @property
    def models_ready(self) -> bool:
        return self.embedder is not None and self.nli is not None

    # ------------------------------------------------------------------ main
    def verify_claims(self, text: str, page_url: str | None = None) -> dict:
        result: dict = {
            "mode": "live",
            "claims": [],
            "overall": evidence.overall_verdict([]),
            "stale_warning": None,
            "fact_checks": [],
            "sources": [],
            "manual_check_links": self.manual_check_links(text),
            "search_queries": [],
            "errors": [],
        }
        if not self.models_ready:
            result["mode"] = "unavailable"
            result["errors"].append(
                "The evidence models couldn't be loaded, so claims weren't checked automatically "
                f"({self.models_error}). Check your internet connection and restart the app."
            )
            return result

        checked = evidence.check_text(text, self.embedder, self.nli)
        if self.google_api_key:
            self._add_rated_fact_checks(checked)
        result.update(claims=checked["claims"], overall=checked["overall"], stale_warning=checked["stale_warning"])
        result["search_queries"] = [q for c in checked["claims"] for q in c.get("queries", [])[:1]]
        errs = {e for c in checked["claims"] for e in c.get("errors", [])}
        result["errors"].extend(sorted(errs))

        # Published fact-checks found among the evidence.
        seen = set()
        for c in checked["claims"]:
            for e in c["evidence"]:
                key = _norm_title(e["title"]) or e["url"]
                if e["kind"] == "fact_check" and key not in seen and e["similarity"] >= evidence.MIN_SUPPORT_SIMILARITY:
                    seen.add(key)
                    rating = e.get("published_rating") or ("Rated false" if e["stance"] == "contradicts" else "")
                    result["fact_checks"].append({
                        "claim": c["claim"], "title": e["title"], "url": e["url"], "publisher": e["outlet"],
                        "date": e.get("date", ""), "rating": rating[:60],
                    })

        # Optional keyed web search, listed as extra coverage.
        if (self.google_api_key and self.google_cx) or self.bing_api_key:
            q = self.key_query(text)
            found = self.search_google(q) if self.google_api_key and self.google_cx else self.search_bing(q)
            result["sources"] = found[:5]
        if page_url:
            result["page_source"] = rep.describe(page_url)
        return result

    # ------------------------------------------- rated fact-checks (Google API)
    def _add_rated_fact_checks(self, checked: dict) -> None:
        """Add published fact-check ratings as evidence and re-score each claim.

        A rating counts only when the fact-checked claim closely matches ours
        (same embedding model as the rest of the evidence).
        """
        for c in checked["claims"]:
            reviews = self.search_fact_checks(evidence.search_query(c["claim"], 8) or c["claim"])
            if not reviews:
                continue
            vecs = self.embedder.encode([c["claim"]] + [r["claim"] or r["title"] for r in reviews])
            sims = vecs[1:] @ vecs[0]
            known = {e["url"] for e in c["evidence"]}
            by_title = {_norm_title(e["title"]): e for e in c["evidence"]}
            for r, sim in zip(reviews, sims):
                if sim < evidence.MIN_CONTRA_SIMILARITY or not r["url"] or r["url"] in known:
                    continue
                verdict = rating_stance(r["rating"])
                same = by_title.get(_norm_title(r["title"]))
                if same is not None:
                    # Already found via news search: attach the published rating to it.
                    same.update(kind="fact_check", rating="fact_checker", published_rating=r["rating"])
                    if verdict != "neutral":
                        same.update(stance=verdict, p_support=0.95 if verdict == "supports" else 0.0,
                                    p_contradict=0.95 if verdict == "contradicts" else 0.0)
                    continue
                c["evidence"].insert(0, {
                    "text": r["claim"] or r["title"], "title": r["title"] or r["claim"], "url": r["url"],
                    "outlet": r["publisher"], "domain": rep.domain_of(r["url"]), "date": r["date"],
                    "kind": "fact_check", "rating": "fact_checker", "similarity": float(sim), "shared_terms": 2,
                    "stance": verdict, "p_support": 0.95 if verdict == "supports" else 0.0,
                    "p_contradict": 0.95 if verdict == "contradicts" else 0.0, "published_rating": r["rating"],
                })
            S = sum(e["similarity"] * rep.EVIDENCE_WEIGHT[e["rating"]] * e["p_support"]
                    for e in c["evidence"] if e["stance"] == "supports")
            C = sum(e["similarity"] * rep.EVIDENCE_WEIGHT[e["rating"]] * e["p_contradict"]
                    for e in c["evidence"] if e["stance"] == "contradicts")
            c["support_score"], c["contradict_score"] = round(S, 3), round(C, 3)
            c["verdict"] = evidence._claim_verdict(S, C, c["evidence"])
        checked["overall"] = evidence.overall_verdict(checked["claims"])

    # --------------------------------------------------------- keyed search
    def search_google(self, query: str) -> list[dict]:
        try:
            r = requests.get("https://www.googleapis.com/customsearch/v1", params={
                "key": self.google_api_key, "cx": self.google_cx, "q": query, "num": 8, "safe": "active",
            }, timeout=10)
            r.raise_for_status()
            return self._web_items((i.get("title", ""), i.get("link", ""), i.get("snippet", ""))
                                   for i in r.json().get("items", []))
        except Exception:
            return []  # never surface the exception: its message contains the key

    def search_bing(self, query: str) -> list[dict]:
        try:
            r = requests.get("https://api.bing.microsoft.com/v7.0/search",
                             headers={"Ocp-Apim-Subscription-Key": self.bing_api_key},
                             params={"q": query, "count": 8, "safeSearch": "Moderate"}, timeout=10)
            r.raise_for_status()
            return self._web_items((i.get("name", ""), i.get("url", ""), i.get("snippet", ""))
                                   for i in r.json().get("webPages", {}).get("value", []))
        except Exception:
            return []

    @staticmethod
    def _web_items(rows) -> list[dict]:
        out = []
        for title, url, snippet in rows:
            rating = rep.rate(url)
            if rating in ("reputable", "fact_checker"):
                out.append({"title": title, "url": url, "summary": snippet, "domain": rep.domain_of(url), "rating": rating})
        return out

    def search_fact_checks(self, text: str, max_results: int = 5) -> list[dict]:
        """Google Fact Check Tools API (ClaimReview data from Snopes, PolitiFact, AFP, ...)."""
        try:
            r = requests.get("https://factchecktools.googleapis.com/v1alpha1/claims:search", params={
                "query": self.key_query(text, 10), "key": self.google_api_key,
                "pageSize": max_results, "languageCode": "en",
            }, timeout=10)
            r.raise_for_status()
        except Exception:
            return []
        out = []
        for claim in r.json().get("claims", []):
            for review in claim.get("claimReview", [])[:1]:
                out.append({
                    "claim": claim.get("text", ""), "title": review.get("title", ""), "url": review.get("url", ""),
                    "publisher": (review.get("publisher") or {}).get("name", ""),
                    "rating": review.get("textualRating", ""), "date": review.get("reviewDate", "")[:10],
                })
        return out[:max_results]

    # ------------------------------------------------------------- helpers
    @staticmethod
    def key_query(text: str, max_words: int = 10) -> str:
        claims = evidence.extract_claims(text, max_claims=1)
        return evidence.search_query(claims[0] if claims else text[:300], max_words) or text[:100]

    def manual_check_links(self, text: str) -> list[dict]:
        q = self.key_query(text)
        enc = quote(q)
        return [
            {"name": "Google Fact Check", "url": f"https://toolbox.google.com/factcheck/explorer/search/{enc}"},
            {"name": "Google News", "url": f"https://news.google.com/search?q={enc}"},
            {"name": "Snopes", "url": f"https://www.snopes.com/search/?q={enc}"},
            {"name": "Reuters Fact Check", "url": f"https://duckduckgo.com/?q={quote('site:reuters.com/fact-check ' + q)}"},
            {"name": "AP News", "url": f"https://duckduckgo.com/?q={quote('site:apnews.com ' + q)}"},
        ]
