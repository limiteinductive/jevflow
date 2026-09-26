"""Scan prose with the mirror: every sentence gets a calibrated P(reads as AI) and its reasons."""

from dataclasses import dataclass

from mirror import jev
from mirror.model import Reason, Stacker, sentence_features
from text_processing import split_sentences

FLAG_THRESHOLD = 0.7
CONTEXT_SENTENCES = 2


@dataclass(frozen=True)
class Finding:
    start: int
    """Character offset of the sentence in the scanned text."""
    end: int
    sentence: str
    probability: float
    """Calibrated P(the sentence reads as AI-generated)."""
    reasons: list[Reason]
    """Writer-facing reasons, strongest first; empty when nothing pushes the sentence toward AI."""


@dataclass(frozen=True)
class Scan:
    findings: list[Finding]
    probability: float
    """Calibrated P(the whole text reads as AI-generated)."""
    input_tokens: int


async def scan(text: str, stacker: Stacker) -> Scan:
    """Ask the question bank about each sentence with its neighbors, one Jev request per sentence, and score it with the stacker."""
    sentences = split_sentences(text)
    texts = [sentence.text for sentence in sentences]
    states = [
        jev.render(sentence, " ".join(texts[max(0, index - CONTEXT_SENTENCES) : index + CONTEXT_SENTENCES + 1]))
        for index, sentence in enumerate(texts)
    ]
    records = await jev.ask_many(states)
    findings, logits, input_tokens = [], [], 0
    for sentence, record in zip(sentences, records):
        if "error" in record:
            raise RuntimeError(record["error"])
        features = sentence_features(record["probabilities"], sentence.text)
        logit = stacker.logit(features)
        logits.append(logit)
        input_tokens += record["input_tokens"]
        findings.append(Finding(sentence.start, sentence.end, sentence.text, stacker.sentence_probability(logit), stacker.reasons(features)))
    return Scan(findings, stacker.document_probability(logits) if logits else 0.0, input_tokens)
