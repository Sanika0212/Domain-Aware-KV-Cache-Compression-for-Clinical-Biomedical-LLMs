"""Section parsing for clinical notes, with real tokenizer-level section tagging.

Two layers:
1. `extract_sections` — regex/header based segmentation of raw note text into
   named sections with character spans (chief complaint, HPI, meds, labs, ...).
2. `tag_token_sections` — maps every *token* produced by a real HuggingFace
   tokenizer to the section it falls in, via `return_offsets_mapping`. This is
   what lets the budget allocator and `DomainAwarePress` operate on the actual
   KV-cache sequence dimension instead of on strings.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import re

import numpy as np

DEFAULT_HEADERS = [
    r"(?i)^chief complaint", r"(?i)^history of present illness", r"(?i)^past medical history",
    r"(?i)^medications?\b", r"(?i)^allergies", r"(?i)^vital", r"(?i)^labs?\b",
    r"(?i)^assessment and plan", r"(?i)^plan\b", r"(?i)^physical exam", r"(?i)^impression",
    r"(?i)^social history", r"(?i)^family history", r"(?i)^review of systems",
]

# Header pattern -> canonical section name, used by allocator's importance profile.
HEADER_CANONICAL = {
    r"(?i)^chief complaint": "chief_complaint",
    r"(?i)^history of present illness": "hpi",
    r"(?i)^past medical history": "pmh",
    r"(?i)^medications?\b": "medications",
    r"(?i)^allergies": "allergies",
    r"(?i)^vital": "vitals",
    r"(?i)^labs?\b": "labs",
    r"(?i)^assessment and plan": "assessment_and_plan",
    r"(?i)^plan\b": "assessment_and_plan",
    r"(?i)^physical exam": "physical_exam",
    r"(?i)^impression": "assessment_and_plan",
    r"(?i)^social history": "social_history",
    r"(?i)^family history": "family_history",
    r"(?i)^review of systems": "review_of_systems",
}


@dataclass
class Section:
    name: str
    canonical: str
    start: int  # char offset, inclusive
    end: int    # char offset, exclusive


@dataclass
class SectionedNote:
    text: str
    sections: List[Section]

    def section_texts(self) -> Dict[str, str]:
        return {s.name: self.text[s.start:s.end].strip() for s in self.sections}


def extract_sections(text: str, headers: List[str] = None) -> SectionedNote:
    """Segment `text` into named sections with character spans.

    A line is treated as a header if it matches one of `headers` (case-insensitive
    regexes anchored at line start) and is short (<=60 chars, the rest of the line
    is the section body start). Falls back to a single "full_note" section if no
    headers are found.
    """
    headers = headers or DEFAULT_HEADERS
    header_regex = re.compile(r"^\s*(?P<header>[^:\n]{1,60}):?\s*$")

    lines = text.splitlines(keepends=True)
    offsets = []
    pos = 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln)

    sections: List[Section] = []
    current_name = "preamble"
    current_canonical = "preamble"
    current_start = 0

    def matched_canonical(header_text: str):
        for pattern in headers:
            if re.search(pattern, header_text):
                return HEADER_CANONICAL.get(pattern, header_text.lower().strip())
        return None

    for i, ln in enumerate(lines):
        stripped = ln.rstrip("\n")
        m = header_regex.match(stripped)
        if m:
            canon = matched_canonical(m.group("header").strip())
            if canon is not None:
                end = offsets[i]
                if end > current_start:
                    sections.append(Section(current_name, current_canonical, current_start, end))
                current_name = m.group("header").strip().lower()
                current_canonical = canon
                current_start = offsets[i] + len(ln)
    sections.append(Section(current_name, current_canonical, current_start, len(text)))

    # Drop empty/whitespace-only sections.
    sections = [s for s in sections if text[s.start:s.end].strip()]
    if not sections:
        sections = [Section("full_note", "full_note", 0, len(text))]

    return SectionedNote(text=text, sections=sections)


def tag_token_sections(note: SectionedNote, tokenizer) -> Tuple[np.ndarray, List[str]]:
    """Map each token produced by `tokenizer(note.text)` to a section index.

    Returns
    -------
    section_ids : np.ndarray of shape (seq_len,), dtype int64
        section_ids[i] is the index into `canonical_names` for token i.
    canonical_names : list[str]
        canonical_names[section_ids[i]] is the canonical section name of token i.
    """
    encoded = tokenizer(note.text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = encoded["offset_mapping"]

    canonical_names = sorted({s.canonical for s in note.sections})
    name_to_idx = {n: i for i, n in enumerate(canonical_names)}

    # Build a char->section lookup via sorted spans for fast bisection.
    spans = sorted(((s.start, s.end, s.canonical) for s in note.sections), key=lambda x: x[0])

    def section_for_char(c: int) -> str:
        # Header text preceding a section's content span is assigned to the
        # section it introduces (the first span with start > c), not to
        # whichever span happens to sort last.
        for start, end, canon in spans:
            if c < end:
                return canon
        return spans[-1][2] if spans else "full_note"

    section_ids = np.zeros(len(offsets), dtype=np.int64)
    for i, (start, end) in enumerate(offsets):
        if start == end:
            # Special token (e.g. BOS/EOS) with empty span; assign to the section
            # of the nearest real token, defaulting to the first section.
            section_ids[i] = section_ids[i - 1] if i > 0 else name_to_idx[canonical_names[0]]
            continue
        canon = section_for_char(start)
        section_ids[i] = name_to_idx.get(canon, 0)

    return section_ids, canonical_names
