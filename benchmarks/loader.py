"""Dataset loaders for the domain-aware KV-compression benchmark.

Two real sources, both public and requiring no credentialing:

1. `load_synthetic_notes` — a seeded generator of EHR-style clinical notes
   (chief complaint, HPI, PMH, meds, allergies, vitals, labs, physical exam,
   assessment & plan) with a deterministic QA pair per note and *injected
   copy-forwarded redundancy* (the same finding/sentence repeated verbatim
   across sections), mirroring the note-bloat pattern documented in Wornow et
   al. (arXiv:2412.16178) and the "Addressing Note Bloat" EHR study cited in
   the project brief.

2. `load_pubmedqa` — real biomedical long-context QA from the public
   `qiaojin/PubMedQA` dataset on the Hugging Face Hub (no gating, no license
   agreement required). Each example's structured-abstract paragraphs
   (BACKGROUND/OBJECTIVE/METHODS/RESULTS/CONCLUSIONS) become sections via
   `domain_kv.section_parser.from_labeled_paragraphs`, and the yes/no/maybe
   `final_decision` is the scoring target.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import random

from domain_kv.section_parser import SectionedNote, extract_sections, from_labeled_paragraphs


@dataclass
class BenchmarkExample:
    example_id: str
    note: SectionedNote
    question: str
    expected_answer: str
    answer_type: str  # "f1" (open generation, scored by token-F1) or "choice" (yes/no/maybe)
    choices: Optional[List[str]] = field(default=None)


_DIAGNOSES = ["acute coronary syndrome", "community-acquired pneumonia", "diabetic ketoacidosis", "pulmonary embolism", "acute appendicitis"]
_DRUGS = [("aspirin", "81 mg daily"), ("metformin", "500 mg twice daily"), ("lisinopril", "10 mg daily"), ("atorvastatin", "40 mg nightly"), ("heparin", "5000 units subcutaneously")]
_LABS = [("troponin", "elevated at 0.8 ng/mL"), ("white blood cell count", "elevated at 14.2 K/uL"), ("creatinine", "elevated at 1.9 mg/dL"), ("lactate", "elevated at 3.1 mmol/L")]
_COMPLAINTS = ["chest pain", "shortness of breath", "abdominal pain", "fever and cough", "severe headache"]


def _make_note_text(rng: random.Random) -> tuple:
    age = rng.randint(35, 85)
    sex = rng.choice(["male", "female"])
    complaint = rng.choice(_COMPLAINTS)
    diagnosis = rng.choice(_DIAGNOSES)
    drug, dose = rng.choice(_DRUGS)
    lab_name, lab_val = rng.choice(_LABS)

    hpi_sentence = (
        f"Patient is a {age}-year-old {sex} presenting with {complaint} for the past 2 days, "
        f"with associated symptoms concerning for {diagnosis}."
    )
    plan_sentence = f"Working diagnosis is {diagnosis}; will start {drug} {dose} and monitor closely."

    sections = {
        "chief complaint": complaint.capitalize() + ".",
        "history of present illness": (
            hpi_sentence + " No prior similar episodes. " + hpi_sentence  # copy-forwarded duplicate
        ),
        "past medical history": "Hypertension. Hyperlipidemia. No prior surgeries.",
        "medications": f"{drug.capitalize()} {dose}. Multivitamin daily.",
        "allergies": "No known drug allergies.",
        "vital signs": "BP 138/86, HR 92, RR 18, Temp 98.9 F, SpO2 97% on room air.",
        "labs": f"{lab_name.capitalize()} {lab_val}. Basic metabolic panel otherwise unremarkable.",
        "physical exam": "Alert and oriented. No acute distress. Exam notable for findings consistent with chief complaint.",
        "assessment and plan": plan_sentence + " " + hpi_sentence,  # copy-forwarded into plan too
    }
    text = "\n\n".join(f"{name.title()}:\n{body}" for name, body in sections.items())
    return text, drug, dose


def load_synthetic_notes(n: int = 20, seed: int = 0) -> List[BenchmarkExample]:
    """Generate `n` synthetic clinical notes with a deterministic medication
    QA target (open-generation, scored via token-F1 against "<drug> <dose>").
    """
    rng = random.Random(seed)
    examples = []
    for i in range(n):
        text, drug, dose = _make_note_text(rng)
        note = extract_sections(text)
        examples.append(
            BenchmarkExample(
                example_id=f"synthetic_{i}",
                note=note,
                question="What medication was started and at what dose?",
                expected_answer=f"{drug} {dose}",
                answer_type="f1",
            )
        )
    return examples


def load_pubmedqa(n: int = 20, split: str = "train") -> List[BenchmarkExample]:
    """Load `n` examples from the public `qiaojin/PubMedQA` (pqa_labeled)
    dataset. Requires network access on first call (cached by `datasets`
    afterwards). Each example asks the dataset's natural yes/no/maybe question
    against the structured abstract.
    """
    from datasets import load_dataset

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split=split)
    ds = ds.select(range(min(n, len(ds))))

    examples = []
    for row in ds:
        ctx = row["context"]
        note = from_labeled_paragraphs(ctx["labels"], ctx["contexts"])
        examples.append(
            BenchmarkExample(
                example_id=f"pubmedqa_{row['pubid']}",
                note=note,
                question=row["question"],
                expected_answer=row["final_decision"],
                answer_type="choice",
                choices=["yes", "no", "maybe"],
            )
        )
    return examples
