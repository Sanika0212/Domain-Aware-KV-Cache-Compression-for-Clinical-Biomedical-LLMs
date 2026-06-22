from transformers import AutoTokenizer

from domain_kv.section_parser import extract_sections, tag_token_sections, from_labeled_paragraphs

NOTE = """Chief complaint:
Chest pain.

History of present illness:
Patient has chest pain for 2 days.

Medications:
Aspirin 81mg daily.
"""


def test_extract_sections_basic():
    note = extract_sections(NOTE)
    texts = note.section_texts()
    assert texts["chief complaint"] == "Chest pain."
    assert "chest pain for 2 days" in texts["history of present illness"]
    assert "Aspirin" in texts["medications"]


def test_extract_sections_canonical_names():
    note = extract_sections(NOTE)
    canon = {s.name: s.canonical for s in note.sections}
    assert canon["history of present illness"] == "hpi"
    assert canon["medications"] == "medications"


def test_extract_sections_no_headers_falls_back_to_full_note():
    note = extract_sections("just some plain text with no headers at all")
    assert [s.canonical for s in note.sections] == ["full_note"]


def test_tag_token_sections_aligns_with_tokenizer_length():
    tok = AutoTokenizer.from_pretrained("gpt2")
    note = extract_sections(NOTE)
    section_ids, names = tag_token_sections(note, tok)
    encoded = tok(note.text, add_special_tokens=True)
    assert len(section_ids) == len(encoded["input_ids"])
    assert set(section_ids.tolist()) <= set(range(len(names)))


def test_tag_token_sections_header_text_belongs_to_its_own_section():
    # Regression test: header text (e.g. "Chief complaint:") must be assigned
    # to the section it introduces, not to whichever section sorts last.
    tok = AutoTokenizer.from_pretrained("gpt2")
    note = extract_sections(NOTE)
    section_ids, names = tag_token_sections(note, tok)
    assert names[section_ids[0]] == "chief_complaint"


def test_from_labeled_paragraphs():
    note = from_labeled_paragraphs(
        ["BACKGROUND", "RESULTS"],
        ["Some background text.", "Some results text."],
    )
    texts = note.section_texts()
    assert texts["background"] == "Some background text."
    assert texts["results"] == "Some results text."
