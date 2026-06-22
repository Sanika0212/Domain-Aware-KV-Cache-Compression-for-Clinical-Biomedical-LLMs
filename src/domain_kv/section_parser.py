"""Simple section parser for clinical notes.

Parses a clinical text into sections using common header keywords and newline structure.
This is intentionally lightweight — for production use replace with a clinical NLP parser.
"""
from typing import List, Tuple, Dict
import re

DEFAULT_HEADERS = [
    r"(?i)^chief complaint", r"(?i)^history of present illness", r"(?i)^past medical history",
    r"(?i)^medication", r"(?i)^medications", r"(?i)^allergies", r"(?i)^vital", r"(?i)^lab",
    r"(?i)^assessment and plan", r"(?i)^plan", r"(?i)^physical exam", r"(?i)^impression",
]

def extract_sections(text: str, headers: List[str] = DEFAULT_HEADERS) -> Dict[str, str]:
    """Extract sections from clinical text.

    Returns a dict mapping section name -> text. If no headers found, returns {'full_note': text}.
    """
    lines = text.splitlines()
    sections: Dict[str, List[str]] = {}
    current = "preamble"
    sections[current] = []
    header_regex = re.compile(r"^\s*(?P<header>[^:]{1,60}):?\s*$")

    for ln in lines:
        m = header_regex.match(ln)
        if m:
            h = m.group("header").strip()
            # check header against provided headers list
            for pattern in headers:
                if re.search(pattern, h):
                    current = h.lower()
                    sections[current] = []
                    break
            else:
                sections[current].append(ln)
        else:
            sections[current].append(ln)

    # join
    joined = {k: "\n".join(v).strip() for k, v in sections.items() if v and "".join(v).strip()}
    if not joined:
        return {"full_note": text}
    return joined
