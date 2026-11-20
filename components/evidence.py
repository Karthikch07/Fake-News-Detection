"""Claim-by-claim fact checking against live sources, without API keys.

Pipeline for each check-worthy claim in the text:

1. Retrieve candidate evidence
   * Google News RSS search (news headlines with outlet and date),
   * the same search restricted to fact-check coverage,
   * Wikipedia (search + the intro of the best matching articles).
2. Keep the passages most related to the claim (MiniLM sentence embeddings
   plus a shared-key-term check, so unrelated stories are never counted).
3. Ask an NLI model (DeBERTa-v3) whether each passage supports or
   contradicts the claim. Fact-check articles are also read for their
   rating words ("false", "misleading", "no evidence", ...), because their
   titles often repeat the false claim word for word.
4. Weigh the stances by relevance and outlet rating to get a verdict per
   claim: supported / contradicted / disputed / not enough evidence.
"""

from __future__ import annotations

import datetime as dt
import email.utils
import json
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import requests

from components import source_reputation as rep

HEADERS = {"User-Agent": "FakeNewsDetector/1.0 (local research app; python-requests)"}
TIMEOUT = 8

MIN_SIMILARITY = 0.45        # embedding similarity for a passage to count as "about" the claim
MIN_SUPPORT_SIMILARITY = 0.55   # before a passage may support the claim
MIN_REFERENCE_SUPPORT_SIMILARITY = 0.66  # Wikipedia sentences need to be closer: they're often background
MIN_CONTRA_SIMILARITY = 0.62    # before it may contradict (NLI over-calls contradiction on related-but-different stories)
FACT_CHECK_SIMILARITY = 0.72    # a fact-check this close to the claim is about the same claim
STANCE_PROB = 0.70           # NLI probability needed to call a stance
DECISIVE_SCORE = 0.60        # weighted evidence needed for a claim verdict

STOPWORDS = set("""
a an the and or but if then than so of in on at to for from by with about into over after before under
is are was were be been being am do does did done has have had having will would shall should can could may might must
this that these those it its it's they them their there here he she his her him we our you your i me my
not no nor only just also very too more most less least much many some any all each every both either neither
what which who whom whose when where why how as up down out off again further once
said says say new news report reports reported according claims claim claimed
""".split())

NOISE_WORDS = {
    "shocking", "breaking", "urgent", "share", "viral", "must", "watch", "read", "click", "wow", "omg",
    "truth", "exposed", "secret", "miracle", "unbelievable", "stunned", "insane", "amazing", "finally",
    "doctors", "hate", "don't", "doesn't", "want", "know", "people", "everyone", "nobody", "really",
    "instantly", "overnight", "big", "they", "hiding", "hidden",
}

FALSE_RATING_RE = re.compile(
    r"\b(false|fake|hoax|misleading|mislead|no evidence|not true|untrue|baseless|debunk\w*|fabricated|"
    r"doctored|altered|manipulated|satire|scam|myth|incorrect|inaccurate|unfounded|unproven|"
    r"does not|doesn't|did not|didn't|is not|isn't|was not|wasn't|cannot|can't|won't|no,|nope)\b",
    re.IGNORECASE,
)
TRUE_RATING_RE = re.compile(r"\b(rating:\s*true|is true|correct|accurate|confirmed|yes,)\b", re.IGNORECASE)

PREFIX_RE = re.compile(
    r"^\s*(?:(?:breaking|shocking|urgent|just in|alert|update|exclusive|must read|viral|watch|"
    r"fact|fake|true|read this)\s*(?:news)?\s*[:!\-–—|]+\s*)+",
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://\S+|www\.\S+")
CTA_RE = re.compile(r"\b(share (this|before|now|with)|forward this|click (here|the link)|subscribe|follow us|like and share)\b", re.I)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'“A-Z0-9])|\n+")
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


# --------------------------------------------------------------------------
# Claim extraction
# --------------------------------------------------------------------------
def _clean_sentence(s: str) -> str:
    s = URL_RE.sub(" ", s)
    s = re.sub(r"[#@]\w+", " ", s)
    s = re.sub(r"[^\w\s.,;:%$€£₹'\"“”‘’()\-–—/!?&]", " ", s)  # emoji and symbols
    s = PREFIX_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip(" -–—:|")
    return s


def _checkworthiness(s: str, first: bool) -> float:
    words = s.split()
    score = 0.0
    score += 1.5 * bool(re.search(r"\d", s))
    score += 0.4 * min(sum(1 for w in words[1:] if w[:1].isupper()), 4)
    score += 1.0 * bool(re.search(
        r"\b(confirm\w*|announc\w*|cause\w*|cure\w*|kill\w*|ban\w*|approv\w*|arrest\w*|die[sd]?|dead|"
        r"found|discover\w*|prove\w*|report\w*|launch\w*|declar\w*|raise\w*|cut\w*|kept|won|lost|"
        r"increase\w*|decrease\w*|will|has|have|is|are|was|were)\b", s, re.I))
    score += 1.0 if first else 0.0
    score -= 2.0 * bool(CTA_RE.search(s))
    score -= 1.0 * s.rstrip().endswith("?")
    return score


def extract_claims(text: str, max_claims: int = 3) -> list[str]:
    """The most check-worthy factual sentences, in their original order."""
    text = (text or "").strip()
    raw = [p for p in SENT_SPLIT_RE.split(text) if p and p.strip()]
    cands: list[tuple[int, str, float]] = []
    seen: set[str] = set()
    for i, sent in enumerate(raw):
        s = _clean_sentence(sent)
        words = s.split()
        if len(words) < 5:
            continue
        if len(words) > 45:
            s = " ".join(words[:45])
        key = s.lower()
        if key in seen or CTA_RE.search(s) and len(words) < 12:
            continue
        seen.add(key)
        cands.append((i, s, _checkworthiness(s, i == 0)))
    if not cands:
        s = _clean_sentence(text)
        return [" ".join(s.split()[:45])] if len(s.split()) >= 3 else []
    best = sorted(cands, key=lambda c: c[2], reverse=True)[:max_claims]
    return [s for _, s, _ in sorted(best)]


def content_terms(text: str) -> list[str]:
    out = []
    for w in WORD_RE.findall(text):
        lw = w.lower().strip("'-")
        if len(lw) < 3 and not lw.isdigit():
            continue
        if lw in STOPWORDS or lw in NOISE_WORDS:
            continue
        out.append(lw)
    return list(dict.fromkeys(out))


def _stem(w: str) -> str:
    for suf in ("ing", "ies", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def shared_terms(a: str, b: str) -> int:
    sa = {_stem(w) for w in content_terms(a)}
    sb = {_stem(w) for w in content_terms(b)}
    return len(sa & sb)


def entity_phrases(claim: str, limit: int = 2) -> list[str]:
    """Runs of capitalised words ("Eiffel Tower", "Neil Armstrong", "NASA")."""
    words = claim.split()
    phrases, cur = [], []
    for i, raw in enumerate(words + ["."]):
        w = raw.strip(".,;:!?\"'“”‘’()")
        cap = bool(w) and w[:1].isupper() and w.lower() not in STOPWORDS | NOISE_WORDS
        if i == 0 and cap and not (len(words) > 1 and words[1][:1].isupper()) and not w.isupper():
            cap = False  # ordinary sentence-initial word
        if cap:
            cur.append(w)
        if not cap or raw[-1:] in ".,;:!?":
            if cur:
                phrases.append(" ".join(cur))
            cur = []
    return list(dict.fromkeys(p for p in phrases if len(p) > 2))[:limit]


_ATTRIB_LEAD_RE = re.compile(
    r"^\s*(?:(?:scientists|experts|researchers|doctors|officials|reports|studies|a (?:new )?study|sources|police|authorities)"
    r"\s+(?:have\s+)?(?:confirm(?:s|ed)?|say|says|said|found|find|finds|show|shows|showed|reveal(?:s|ed)?|claim(?:s|ed)?)\s+(?:that\s+)?"
    r"|according to [^,]+,\s*)",
    re.I,
)
_ATTRIB_TAIL_RE = re.compile(
    r",?\s+(?:(?:officials|experts|analysts|sources|police|authorities|researchers|scientists|the \w+|a spokes\w+|he|she|they)\s+)?"
    r"(?:said|says|say|reported|announced|confirmed|told \w+)(?:\s+(?:in a statement|on \w+day|in a report|on \w+))?"
    r"(?:,?\s+citing [^.]*)?\.?\s*$|,?\s+according to [^.]*\.?\s*$",
    re.I,
)
_TIME_RE = re.compile(
    r"\s+(?:at (?:its|the|their|his|her) (?:latest|last|recent|next|first|annual) [a-z]+(?: meeting)?"
    r"|on (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|in (?:january|february|march|april|may|june|july|august|september|october|november|december)(?: \d{4})?"
    r"|last (?:week|month|year|night|weekend)|this (?:week|month|year|morning)|earlier this \w+"
    r"|yesterday|today|on \d{1,2} \w+|on \w+ \d{1,2}(?:, \d{4})?)\b",
    re.I,
)


def claim_core(claim: str) -> str:
    """The claim without attribution and time phrases, used only when looking for support:
    a headline can't confirm "officials said" or "on Thursday", but it can confirm the event."""
    s = _ATTRIB_LEAD_RE.sub("", claim.strip())
    s = _ATTRIB_TAIL_RE.sub("", s)
    s = _TIME_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;:-")
    if len(s.split()) < 4:
        return claim
    return s[0].upper() + s[1:] + ("" if s.endswith((".", "!", "?")) else ".")


def search_query(claim: str, max_terms: int = 7) -> str:
    """Short keyword query: named entities and numbers first, then other content words."""
    words = WORD_RE.findall(claim)
    entities = [w for i, w in enumerate(words) if (w[:1].isupper() and i > 0) or any(c.isdigit() for c in w)]
    terms = content_terms(" ".join(entities)) + content_terms(claim)
    return " ".join(list(dict.fromkeys(terms))[:max_terms])


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
def _parse_date(value: str) -> str:
    try:
        return email.utils.parsedate_to_datetime(value).date().isoformat()
    except Exception:
        return ""


def _get(url: str, params: dict) -> requests.Response:
    """GET with one retry on rate limiting / server errors."""
    for attempt in range(2):
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code in (429, 500, 502, 503, 504) and attempt == 0:
            time.sleep(1.0)
            continue
        r.raise_for_status()
        return r
    return r


@lru_cache(maxsize=256)
def google_news(query: str, limit: int = 12) -> tuple[dict, ...]:
    """Headlines from Google News RSS (free, no key)."""
    if not query.strip():
        return ()
    r = _get("https://news.google.com/rss/search", {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    root = ET.fromstring(r.content)
    items = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        src = it.find("source")
        outlet = (src.text or "").strip() if src is not None else ""
        outlet_url = src.get("url", "") if src is not None else ""
        if outlet and title.endswith(" - " + outlet):
            title = title[: -len(outlet) - 3].strip()
        title = re.sub(r"\s*\((?:Published|Updated) [^)]*\)\s*$", "", title)  # NYT-style suffix
        items.append({
            "text": title,
            "title": title,
            "url": (it.findtext("link") or "").strip(),
            "outlet": outlet,
            "domain": rep.domain_of(outlet_url),
            "date": _parse_date(it.findtext("pubDate") or ""),
            "kind": "news",
        })
        if len(items) >= limit:
            break
    return tuple(items)


WIKI_API = "https://en.wikipedia.org/w/api.php"
_WIKI_LOCK = threading.Lock()  # Wikipedia rate-limits bursts of anonymous requests
WIKI_CACHE_PATH = os.path.join("data", "cache", "wikipedia.json")
WIKI_COOLDOWN_S = 300
_wiki_blocked_until = 0.0
_wiki_cache: dict[str, list] | None = None


class WikipediaRateLimited(RuntimeError):
    pass


def _wiki_headers() -> dict:
    # Wikimedia asks API clients to identify themselves; a contact in the
    # User-Agent gets more generous rate limits. Set WIKIMEDIA_CONTACT in .env.
    contact = os.getenv("WIKIMEDIA_CONTACT", "").strip()
    ua = HEADERS["User-Agent"] if not contact else f"FakeNewsDetector/1.0 ({contact}) python-requests"
    return {"User-Agent": ua}


def _wiki_cache_load() -> dict:
    global _wiki_cache
    if _wiki_cache is None:
        try:
            with open(WIKI_CACHE_PATH, encoding="utf-8") as f:
                _wiki_cache = json.load(f)
        except Exception:
            _wiki_cache = {}
    return _wiki_cache


def _wiki_cache_save() -> None:
    try:
        os.makedirs(os.path.dirname(WIKI_CACHE_PATH), exist_ok=True)
        cache = _wiki_cache or {}
        if len(cache) > 2000:  # keep the most recent entries
            for k in list(cache)[: len(cache) - 2000]:
                cache.pop(k, None)
        tmp = WIKI_CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        os.replace(tmp, WIKI_CACHE_PATH)
    except Exception:
        pass


def wikipedia_status() -> str:
    """'ok' or a short reason Wikipedia is being skipped."""
    left = _wiki_blocked_until - time.time()
    return "ok" if left <= 0 else f"rate limited, retrying in {int(left // 60) + 1} min"


def _wiki_pages(query: str, n: int) -> tuple[tuple[str, str], ...]:
    """(title, intro text) of the top search hits, in one request (cached on disk)."""
    global _wiki_blocked_until
    key = f"{n}|{query.lower()}"
    with _WIKI_LOCK:
        cache = _wiki_cache_load()
        if key in cache:
            return tuple(tuple(x) for x in cache[key])
        if time.time() < _wiki_blocked_until:
            raise WikipediaRateLimited(wikipedia_status())
        params = {"action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": n,
                  "prop": "extracts", "exintro": 1, "explaintext": 1, "redirects": 1,
                  "format": "json", "formatversion": 2}
        r = requests.get(WIKI_API, params=params, headers=_wiki_headers(), timeout=TIMEOUT)
        if r.status_code == 429:
            _wiki_blocked_until = time.time() + WIKI_COOLDOWN_S
            raise WikipediaRateLimited(wikipedia_status())
        r.raise_for_status()
        pages = sorted(r.json().get("query", {}).get("pages", []), key=lambda p: p.get("index", 99))
        result = [[p.get("title", ""), p.get("extract") or ""] for p in pages]
        cache[key] = result
        _wiki_cache_save()
        return tuple(tuple(x) for x in result)


@lru_cache(maxsize=256)
def wikipedia(query: str, entities: tuple[str, ...] = (), pages: int = 3) -> tuple[dict, ...]:
    """Sentences from the intros of the best-matching English Wikipedia articles.

    Each named entity in the claim is looked up on its own first ("Eiffel Tower"),
    then the keyword query fills the remaining slots.
    """
    found: list[tuple[str, str]] = []
    for ent in entities:
        found += _wiki_pages(ent, 1)
    if query.strip():
        found += _wiki_pages(query, 2)
    found = list(dict(found).items())[:pages]  # unique titles, first occurrence order
    out = []
    for title, extract in found:
        extract = re.sub(r"\s+", " ", extract)
        url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
        title_words = {w.lower() for w in WORD_RE.findall(re.sub(r"\(.*?\)", "", title)) if len(w) > 2}
        for sent in re.split(r"(?<=[.!?])\s+", extract)[:12]:
            if len(sent.split()) < 6:
                continue
            # Give pronoun sentences ("It was designed by...") their subject back.
            sent_words = {w.lower() for w in WORD_RE.findall(sent)}
            text = sent if title_words & sent_words else f"{title}: {sent}"
            out.append({
                "text": text, "title": title, "url": url, "outlet": "Wikipedia",
                "domain": "en.wikipedia.org", "date": "", "kind": "reference",
            })
    return tuple(out)


def gather_evidence(claim: str) -> tuple[list[dict], list[str], list[str]]:
    """All candidate passages for a claim, the queries used, and any retrieval errors."""
    q = search_query(claim)
    q_short = search_query(claim, 5)
    jobs = {
        "news": (google_news, (q,)),
        "fact-checks": (google_news, (f"{q_short} fact check",)),
        "Wikipedia": (wikipedia, (q_short, tuple(entity_phrases(claim)))),
    }
    results: list[dict] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {name: pool.submit(fn, *args) for name, (fn, args) in jobs.items()}
        for name, fut in futures.items():
            try:
                results.extend(dict(x) for x in fut.result())
            except WikipediaRateLimited as e:
                errors.append(f"Wikipedia skipped ({e})")
            except Exception as e:
                code = getattr(getattr(e, "response", None), "status_code", None)
                errors.append(f"{name} search failed ({type(e).__name__}{f' {code}' if code else ''})")
    # Broaden once if the precise query found no news at all.
    if not any(r["kind"] == "news" for r in results) and q_short != q:
        try:
            results.extend(dict(x) for x in google_news(q_short))
        except Exception:
            pass
    seen, unique = set(), []
    for r in results:
        key = r["text"].lower()[:160]
        if key not in seen:
            seen.add(key)
            r["rating"] = rep.rate(r["url"] if r["kind"] == "reference" else (r["domain"] or r["url"]))
            if r["kind"] == "news" and r["rating"] == "fact_checker":
                r["kind"] = "fact_check"
            elif r["kind"] == "news" and re.match(r"\s*fact[- ]?check", r["text"], re.I):
                r["kind"] = "fact_check"
            unique.append(r)
    return unique, [q, f"{q_short} fact check", q_short], errors


# --------------------------------------------------------------------------
# Stance and verdicts
# --------------------------------------------------------------------------
def _stance(item: dict, fwd: dict, rev: dict, core: dict | None = None) -> tuple[str, float, float]:
    """(stance, support probability, contradiction probability) for one passage.

    fwd: passage -> claim; rev: claim -> passage; core: passage -> claim without
    attribution / time phrases (support only, slightly discounted).
    """
    support = fwd["entailment"]
    if core is not None:
        support = max(support, 0.9 * core["entailment"])
    if item["kind"] in ("news", "fact_check") and len(item["text"].split()) <= 25:
        # A headline the claim implies is reporting the same thing. (Not used for
        # reference sentences: a claim trivially "implies" generic background facts.)
        support = max(support, rev["entailment"])
    # Real contradictions hold in both directions; one-way "contradictions" are
    # usually the NLI model reacting to a related but different statement.
    contra = min(fwd["contradiction"], rev["contradiction"])
    if item["kind"] == "fact_check":
        text = item["text"]
        if FALSE_RATING_RE.search(text) and not TRUE_RATING_RE.search(text):
            contra = max(contra, 0.9)
            support = 0.0
        elif not TRUE_RATING_RE.search(text):
            support = min(support, 0.3)  # a fact-check title restating a claim is not support
    if contra >= STANCE_PROB and contra > support:
        return "contradicts", support, contra
    if support >= STANCE_PROB:
        return "supports", support, contra
    return "neutral", support, contra


def check_claim(claim: str, embedder, nli, max_evidence: int = 8) -> dict:
    evidence, queries, errors = gather_evidence(claim)
    result = {
        "claim": claim, "queries": queries, "errors": errors, "evidence": [],
        "verdict": "Not enough evidence", "support_score": 0.0, "contradict_score": 0.0,
    }
    if not evidence:
        return result

    vecs = embedder.encode([claim] + [e["text"] for e in evidence])
    sims = vecs[1:] @ vecs[0]
    for e, s in zip(evidence, sims):
        e["similarity"] = float(s)
        e["shared_terms"] = shared_terms(claim, e["text"])
    related = [e for e in evidence if e["similarity"] >= MIN_SIMILARITY and e["shared_terms"] >= 1]
    related.sort(key=lambda e: e["similarity"] * rep.EVIDENCE_WEIGHT[e["rating"]] ** 0.5, reverse=True)
    related = related[:max_evidence]
    if not related:
        return result

    texts = [e["text"] for e in related]
    fwd = nli.predict(texts, claim)
    rev = nli.predict_reverse(claim, texts)
    core = claim_core(claim)
    fwd_core = nli.predict(texts, core) if core != claim else [None] * len(texts)
    result["core"] = core

    S = C = 0.0
    for i, e in enumerate(related):
        stance, p_sup, p_con = _stance(e, fwd[i], rev[i], fwd_core[i])
        sim, terms = e["similarity"], e["shared_terms"]
        if "?" in e["text"] and not (e["kind"] == "fact_check" and FALSE_RATING_RE.search(e["text"])):
            stance = "neutral"  # a question headline asserts nothing (unless it carries a rating)
        if stance == "contradicts" and e["kind"] == "fact_check":
            if sim < MIN_CONTRA_SIMILARITY or terms < 2:
                stance = "neutral"  # a fact-check about some other claim
        elif stance == "contradicts" and (sim < MIN_CONTRA_SIMILARITY or terms < 2):
            stance = "neutral"
        elif stance == "supports" and (sim < MIN_SUPPORT_SIMILARITY or terms < 2):
            stance = "neutral"
        elif stance == "supports" and e["kind"] == "reference" and sim < MIN_REFERENCE_SUPPORT_SIMILARITY:
            stance = "neutral"  # background sentence, not a statement of the claim
        e.update(stance=stance, p_support=round(p_sup, 3), p_contradict=round(p_con, 3))
        w = e["similarity"] * rep.EVIDENCE_WEIGHT[e["rating"]]
        if stance == "supports":
            S += w * p_sup
        elif stance == "contradicts":
            C += w * p_con
    result["evidence"] = related
    result["support_score"], result["contradict_score"] = round(S, 3), round(C, 3)
    result["verdict"] = _claim_verdict(S, C, related)
    supporting = [e for e in related if e["stance"] == "supports"]
    # Only news-dated support can be "old news"; a reference source backing it means it's a standing fact.
    if result["verdict"] == "Supported" and not any(e["kind"] == "reference" for e in supporting):
        result["dates"] = sorted({e["date"] for e in supporting if e.get("date")})
    return result


def _claim_verdict(S: float, C: float, ev: list[dict]) -> str:
    fact_check_contra = any(e["kind"] == "fact_check" and e["stance"] == "contradicts" for e in ev)
    if (C >= DECISIVE_SCORE or fact_check_contra) and C > 1.5 * S:
        return "Contradicted"
    if S >= DECISIVE_SCORE and S > 1.5 * C:
        return "Supported"
    if S >= DECISIVE_SCORE and C >= DECISIVE_SCORE:
        return "Disputed"
    if any(e["kind"] == "fact_check" and e["similarity"] >= FACT_CHECK_SIMILARITY for e in ev):
        return "Fact-checked"  # a fact-checker covered this claim; its rating isn't in the headline
    return "Not enough evidence"


def overall_verdict(claims: list[dict]) -> dict:
    verdicts = [c["verdict"] for c in claims]
    n_con = verdicts.count("Contradicted")
    n_sup = verdicts.count("Supported")
    n_dis = verdicts.count("Disputed")
    n_fc = verdicts.count("Fact-checked")
    if n_con and not n_sup:
        label, text = "Contradicted", "Sources contradict at least one claim and support none."
    elif n_con and n_sup:
        label, text = "Mixed", "Some claims are supported by sources, others are contradicted."
    elif n_dis:
        label, text = "Disputed", "Sources disagree about this."
    elif n_sup and n_sup == len(claims):
        label, text = "Supported", "Every checked claim is supported by sources."
    elif n_fc:
        label, text = "Fact-checked", "Fact-checkers have covered this claim. Read their rating below; claims they cover are often false."
    elif n_sup:
        label, text = "Partly supported", "Some claims are supported; others couldn't be checked."
    else:
        label, text = "Unverified", "No source clearly confirms or contradicts these claims."
    return {"label": label, "text": text, "supported": n_sup, "contradicted": n_con,
            "disputed": n_dis, "fact_checked": n_fc, "checked": len(claims)}


def stale_coverage(claims: list[dict], years: float = 1.0) -> str | None:
    """Warn when the only supporting coverage is old (news may be recirculating)."""
    dates = [d for c in claims for d in c.get("dates", [])]
    if not dates:
        return None
    newest = max(dates)
    try:
        age = (dt.date.today() - dt.date.fromisoformat(newest)).days / 365.25
    except ValueError:
        return None
    if age >= years:
        return f"The newest supporting coverage is from {newest} ({age:.0f}+ year{'s' if age >= 2 else ''} ago). This may be an old story being shared as new."
    return None


def check_text(text: str, embedder, nli, max_claims: int = 3) -> dict:
    claims = extract_claims(text, max_claims=max_claims)
    if not claims:
        return {"claims": [], "overall": overall_verdict([]), "stale_warning": None}
    with ThreadPoolExecutor(max_workers=min(3, len(claims))) as pool:
        checked = list(pool.map(lambda c: check_claim(c, embedder, nli), claims))
    return {"claims": checked, "overall": overall_verdict(checked), "stale_warning": stale_coverage(checked)}
