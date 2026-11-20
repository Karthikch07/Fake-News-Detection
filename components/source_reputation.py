"""Coarse outlet ratings used to weight evidence and to rate a pasted link.

The lists are deliberately short and conservative: well-known fact-checkers,
established news organisations and reference sites, plus satire sites and
sites with a documented record of publishing fabricated stories. Anything
not listed is "unrated", which is not the same as unreliable.
"""

from __future__ import annotations

from urllib.parse import urlparse

FACT_CHECKERS = {
    "snopes.com", "politifact.com", "factcheck.org", "fullfact.org", "leadstories.com",
    "checkyourfact.com", "truthorfiction.com", "healthfeedback.org", "climatefeedback.org",
    "scicheck.org", "boomlive.in", "altnews.in", "factly.in", "newschecker.in", "vishvasnews.com",
    "factcrescendo.com", "thequint.com/news/webqoof", "africacheck.org", "chequeado.com",
    "maldita.es", "correctiv.org", "factcheck.afp.com", "misbar.com", "verafiles.org",
    "factcheckni.org", "theferret.scot", "teyit.org", "newtral.es", "pagellapolitica.it",
    "demagog.org.pl", "stopfake.org", "mythdetector.com", "aosfatos.org", "lupa.uol.com.br",
    "rappler.com/newsbreak/fact-check", "aap.com.au/factcheck", "logically.ai",
}
# Fact-check sections of larger outlets (matched against domain + path).
FACT_CHECK_PATHS = (
    "reuters.com/fact-check", "apnews.com/hub/ap-fact-check", "usatoday.com/story/news/factcheck",
    "bbc.com/news/reality_check", "bbc.co.uk/news/reality_check", "indiatoday.in/fact-check",
    "washingtonpost.com/politics/fact-checker", "factcheck.afp.com", "thequint.com/news/webqoof",
)

REPUTABLE = {
    # wire services and public broadcasters
    "reuters.com", "apnews.com", "afp.com", "bbc.com", "bbc.co.uk", "npr.org", "pbs.org",
    "cbc.ca", "abc.net.au", "dw.com", "france24.com", "aljazeera.com", "euronews.com",
    # newspapers and broadcasters
    "nytimes.com", "washingtonpost.com", "wsj.com", "theguardian.com", "ft.com", "economist.com",
    "bloomberg.com", "latimes.com", "usatoday.com", "cnn.com", "nbcnews.com", "cbsnews.com",
    "abcnews.go.com", "politico.com", "axios.com", "time.com", "theatlantic.com", "independent.co.uk",
    "telegraph.co.uk", "thetimes.co.uk", "sky.com", "news.sky.com", "cnbc.com", "forbes.com",
    "thehindu.com", "indianexpress.com", "hindustantimes.com", "timesofindia.indiatimes.com",
    "ndtv.com", "indiatoday.in", "livemint.com", "business-standard.com", "scroll.in", "theprint.in",
    "straitstimes.com", "scmp.com", "japantimes.co.jp", "smh.com.au", "theglobeandmail.com",
    # science / health / reference
    "nature.com", "science.org", "sciencedaily.com", "scientificamerican.com", "newscientist.com",
    "who.int", "cdc.gov", "nih.gov", "nhs.uk", "mayoclinic.org", "cancer.org", "cancer.gov",
    "nasa.gov", "noaa.gov", "un.org", "worldbank.org", "imf.org", "europa.eu", "gov.uk",
    "wikipedia.org", "britannica.com",
}

SATIRE = {
    "theonion.com", "babylonbee.com", "clickhole.com", "thebeaverton.com", "waterfordwhispersnews.com",
    "newsthump.com", "thedailymash.co.uk", "fauxy.com", "fakingnews.com", "thespoof.com",
    "duffelblog.com", "reductress.com", "newsbiscuit.com",
}

UNRELIABLE = {
    "infowars.com", "naturalnews.com", "beforeitsnews.com", "newspunch.com", "yournewswire.com",
    "worldnewsdailyreport.com", "nationalreport.net", "empirenews.net", "abcnews.com.co",
    "theonion.news", "realrawnews.com", "thegatewaypundit.com", "zerohedge.com", "globalresearch.ca",
    "naturalnews.tv", "dailybuzzlive.com", "react365.com", "newsexaminer.net", "huzlers.com",
    "now8news.com", "thelastlineofdefense.org", "100percentfedup.com", "christianpost.news",
    "mercola.com", "childrenshealthdefense.org", "thetruthaboutcancer.com", "greenmedinfo.com",
}

SOCIAL = {
    "facebook.com", "x.com", "twitter.com", "instagram.com", "tiktok.com", "youtube.com", "reddit.com",
    "threads.net", "linkedin.com", "pinterest.com", "quora.com", "medium.com", "substack.com",
    "t.me", "telegram.me", "whatsapp.com", "rumble.com", "bitchute.com", "truthsocial.com",
}

RATING_INFO = {
    "social": ("Social media post", "User-posted content. Treat it as a claim, not as evidence."),
    "fact_checker": ("Fact-checker", "Independent fact-checking organisation."),
    "reputable": ("Established outlet", "Established news organisation or reference source."),
    "satire": ("Satire site", "Publishes satire. Its stories are not meant to be true."),
    "unreliable": ("Known unreliable", "Has a documented record of publishing false or fabricated stories."),
    "unrated": ("Unrated", "Not in this app's lists. That doesn't make it unreliable; check who runs it."),
}

# How much a passage from each kind of outlet counts as evidence.
EVIDENCE_WEIGHT = {"fact_checker": 1.5, "reputable": 1.2, "unrated": 0.7, "social": 0.2, "unreliable": 0.15, "satire": 0.0}


def domain_of(url: str) -> str:
    try:
        netloc = urlparse(url if "://" in url else "https://" + url).netloc.lower()
    except Exception:
        return ""
    netloc = netloc.split("@")[-1].split(":")[0]
    return netloc[4:] if netloc.startswith("www.") else netloc


def _matches(domain: str, names: set[str]) -> bool:
    return any(domain == n or domain.endswith("." + n) for n in names if "/" not in n)


def rate(url_or_domain: str) -> str:
    """'fact_checker' | 'reputable' | 'satire' | 'unreliable' | 'unrated'."""
    raw = (url_or_domain or "").lower()
    domain = domain_of(raw)
    if not domain:
        return "unrated"
    path_key = domain + (urlparse(raw if "://" in raw else "https://" + raw).path or "")
    if _matches(domain, FACT_CHECKERS) or any(path_key.startswith(p) or ("." + p) in path_key for p in FACT_CHECK_PATHS):
        return "fact_checker"
    if _matches(domain, SATIRE):
        return "satire"
    if _matches(domain, UNRELIABLE):
        return "unreliable"
    if _matches(domain, SOCIAL):
        return "social"
    if _matches(domain, REPUTABLE) or domain.endswith((".gov", ".edu", ".gov.in", ".nic.in", ".ac.uk", ".gov.uk")):
        return "reputable"
    return "unrated"


def describe(url_or_domain: str) -> dict:
    key = rate(url_or_domain)
    label, detail = RATING_INFO[key]
    return {"domain": domain_of(url_or_domain), "rating": key, "label": label, "detail": detail}
