"""Community Notes and the final verdict, built from the claim-by-claim evidence."""

from __future__ import annotations

VERDICT_WORDS = {"True": "likely real", "False": "likely fake", "Uncertain": "uncertain"}


def verdict_sentence(prediction_result: dict) -> str:
    label = prediction_result.get("prediction_label") or prediction_result.get("prediction")
    confidence = prediction_result.get("confidence")
    words = VERDICT_WORDS.get(label, str(label))
    if isinstance(confidence, (int, float)):
        return f"The writing-style classifier rates this as **{words}** ({confidence:.0f}% confidence)."
    return f"The writing-style classifier rates this as **{words}**."


def final_assessment(prediction: dict | None, verification: dict | None) -> dict:
    """One verdict for the result: evidence decides when there is any, style otherwise.

    Returns {label, kind ('fake'|'real'|'unsure'), basis, reason}.
    """
    prediction = prediction or {}
    overall = (verification or {}).get("overall") or {}
    ev = overall.get("label", "Unverified")
    style = prediction.get("prediction_label", "Uncertain")
    style_fake = style == "False"

    if ev == "Contradicted":
        return {"label": "Likely false", "kind": "fake", "basis": "evidence",
                "reason": "Sources contradict the claim."}
    if ev in ("Mixed", "Disputed"):
        return {"label": "Disputed", "kind": "unsure", "basis": "evidence",
                "reason": overall.get("text", "Sources disagree.")}
    if ev == "Fact-checked":
        return {"label": "Fact-checked: read the rating", "kind": "unsure", "basis": "evidence",
                "reason": "A fact-checker has covered this claim. Their rating is linked below."}
    if ev == "Supported":
        reason = "Sources support the claims."
        if style_fake:
            reason += " The wording is sensational, but the facts check out."
        return {"label": "Likely true", "kind": "real", "basis": "evidence", "reason": reason}
    if ev == "Partly supported":
        if style_fake:
            return {"label": "Uncertain", "kind": "unsure", "basis": "evidence + style",
                    "reason": "Some claims are supported, but the wording looks like misinformation and other claims couldn't be checked."}
        return {"label": "Likely true", "kind": "real", "basis": "evidence",
                "reason": "The checkable claims are supported by sources."}
    # No usable evidence: only the writing style is left to go on.
    if style == "False":
        return {"label": "Likely fake", "kind": "fake", "basis": "writing style only",
                "reason": "No source confirms it and the wording matches misinformation. Not fact-checked."}
    if style == "True":
        return {"label": "Unverified", "kind": "unsure", "basis": "writing style only",
                "reason": "Reads like normal news, but no source could confirm it."}
    return {"label": "Uncertain", "kind": "unsure", "basis": "writing style only",
            "reason": "No source confirms or contradicts it, and the writing style is inconclusive."}


class CommunityNotesGenerator:
    def generate_notes(self, original_text: str, prediction_result: dict, verification_result: dict) -> dict:
        claims = verification_result.get("claims") or []
        contradictions, confirmations, references = [], [], []
        for c in claims:
            short = c["claim"] if len(c["claim"]) <= 140 else c["claim"][:137] + "…"
            for e in c.get("evidence", []):
                ref = {"title": e["title"] if e["kind"] != "reference" else f"Wikipedia: {e['title']}",
                       "url": e["url"], "source": e.get("outlet") or e.get("domain")}
                if e["stance"] == "contradicts":
                    contradictions.append({"claim": short, **ref, "text": e["text"]})
                elif e["stance"] == "supports":
                    confirmations.append({"claim": short, **ref, "text": e["text"]})
                else:
                    continue
                if ref["url"] not in {r["url"] for r in references}:
                    references.append(ref)

        final = final_assessment(prediction_result, verification_result)
        overall = verification_result.get("overall") or {}
        parts = []
        if claims:
            n = len(claims)
            parts.append(f"{n} claim{'s' if n != 1 else ''} checked against news coverage, fact-checkers and Wikipedia. "
                         f"{overall.get('text', '')}")
        else:
            parts.append("No checkable factual claim was found in this text.")
        if final["basis"] == "writing style only":
            parts.append(verdict_sentence(prediction_result) + " That judges wording only, not facts.")
        if verification_result.get("stale_warning"):
            parts.append(verification_result["stale_warning"])
        return {
            "summary": " ".join(p.strip() for p in parts if p.strip()),
            "contradictions": contradictions[:8],
            "confirmations": confirmations[:8],
            "references": references[:12],
            "original_claims": [c["claim"] for c in claims],
        }
