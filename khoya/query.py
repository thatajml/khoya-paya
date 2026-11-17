"""Turns a passenger's free-text complaint into (a) structured filters and
(b) a clean visual description for CLIP.

"I lost my black Wildcraft backpack on 12951 near Borivali"
  -> train_no="12951", station="Borivali", clip_text="a photo of a black wildcraft backpack"
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TRAIN_RE = re.compile(r"\b(?:train\s*(?:no\.?|number)?\s*)?(\d{5})\b", re.I)
LOST_PHRASE_RE = re.compile(
    r"\b(?:i\s+)?(?:have\s+|had\s+)?(?:lost|left|forgot|forgotten|misplaced|dropped)\s*(?:my|a|an|the|our)?\b",
    re.I,
)
DROP_WORDS = {
    "train", "station", "platform", "coach", "bogie", "seat", "berth", "no", "number",
    "yesterday", "today", "morning", "evening", "night", "please", "help", "kindly", "sir",
}
TRAILING = {"on", "at", "near", "in", "from", "to", "while", "during", "of", "the", "a", "an", "and", "my"}

# Common ways passengers name stations -> canonical names used in the data.
STATION_ALIASES = {
    "bombay central": "Mumbai Central",
    "mumbai central": "Mumbai Central",
    "bct": "Mumbai Central",
    "cst": "Mumbai CSMT",
    "csmt": "Mumbai CSMT",
    "vt": "Mumbai CSMT",
    "pune": "Pune Junction",
    "baroda": "Vadodara",
    "amdavad": "Ahmedabad",
}


@dataclass
class ParsedQuery:
    clip_text: str | None
    train_no: str | None
    station: str | None


def _find_station(text: str, stations: list[str]) -> tuple[str | None, str]:
    lowered = text.lower()
    candidates = {s.lower(): s for s in stations}
    for alias, canonical in STATION_ALIASES.items():
        if canonical in stations:
            candidates.setdefault(alias, canonical)
    for name in sorted(candidates, key=len, reverse=True):  # longest match first
        pattern = re.compile(rf"\b{re.escape(name)}\b", re.I)
        if pattern.search(lowered):
            return candidates[name], pattern.sub(" ", text)
    return None, text


def parse_query(text: str | None, stations: list[str]) -> ParsedQuery:
    if not text or not text.strip():
        return ParsedQuery(None, None, None)

    train_no = None
    m = TRAIN_RE.search(text)
    if m:
        train_no = m.group(1)
        text = TRAIN_RE.sub(" ", text)

    station, text = _find_station(text, stations)
    text = LOST_PHRASE_RE.sub(" ", text)
    text = re.sub(r"[^\w\s'-]", " ", text)

    words = [w for w in text.split() if w.lower() not in DROP_WORDS]
    while words and words[-1].lower() in TRAILING:
        words.pop()
    while words and words[0].lower() in TRAILING - {"a", "an"}:
        words.pop(0)
    cleaned = " ".join(words).strip().lower()

    clip_text = None
    if cleaned:
        article = "" if cleaned.split()[0] in {"a", "an", "the"} else ("an " if cleaned[0] in "aeiou" else "a ")
        clip_text = f"a photo of {article}{cleaned}"
    return ParsedQuery(clip_text=clip_text, train_no=train_no, station=station)
