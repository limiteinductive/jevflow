"""Scan prose with the mirror: every sentence gets a calibrated P(reads as AI) and its reasons."""

from dataclasses import dataclass

from mirror import jev
from mirror.model import ASK_BY_NAME, NUM_REASONS, REASON_BY_NAME, Reason, Stacker, sentence_features
from text_processing import split_sentences

FLAG_THRESHOLD = 0.7
CONTEXT_SENTENCES = 2
RHYTHM_THRESHOLD = 0.7
"""Jev's P(flat rhythm around the sentence) that adds the "uniform rhythm" reason; a run of same-length sentences scores 0.83 to 0.96 and a varied paragraph at most 0.32."""
RHYTHM_REASON = Reason(REASON_BY_NAME["same_shape"][0], ASK_BY_NAME["same_shape"], 0.0)
"""Decided by Jev alone: the stacker gives `same_shape` no weight, so it never ranks among the stacker's reasons."""


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
        reasons = stacker.reasons(features)
        if features["same_shape"] >= RHYTHM_THRESHOLD:
            reasons = [RHYTHM_REASON, *reasons][:NUM_REASONS]
        findings.append(Finding(sentence.start, sentence.end, sentence.text, stacker.sentence_probability(logit), reasons))
    return Scan(findings, stacker.document_probability(logits) if logits else 0.0, input_tokens)
