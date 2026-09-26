"""The mirror's stacker: a logistic model over the question bank's P(yes) answers.

Trained by `validation/train.py` on provenance labels and stored in `mirror/model.json`.
Inference needs no scikit-learn.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from mirror.questions import QUESTIONS

MODEL_PATH = Path(__file__).with_name("model.json")
NUM_REASONS = 3
NON_REASONS = {"reads_ai", "reads_human", "log_words", "polished", "typo", "fragment", "quote"}
"""Features the stacker uses but never shows: they name no property, their reason would tell the writer to add roughness, or (quote) almost all prose lacks it."""
REASON_BY_NAME = {question.name: (question.yes_reason, question.no_reason) for question in QUESTIONS}
ASK_BY_NAME = {question.name: question.ask for question in QUESTIONS}


def sentence_features(probabilities: dict[str, float], sentence: str) -> dict[str, float]:
    return {**probabilities, "log_words": math.log1p(len(sentence.split()))}


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, value))))


@dataclass(frozen=True)
class Reason:
    label: str
    ask: str
    """The question to the writer for this reason; empty when there is none."""
    weight: float
    """This feature's contribution to the sentence's logit toward AI."""


@dataclass(frozen=True)
class Stacker:
    features: list[str]
    means: list[float]
    scales: list[float]
    weights: list[float]
    intercept: float
    sentence_calibration: tuple[float, float]
    """Platt slope and offset mapping the raw logit to a calibrated sentence probability."""
    document_calibration: tuple[float, float]
    """Platt slope and offset mapping a document's mean sentence logit to a calibrated document probability."""

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "Stacker":
        return cls(**json.loads(path.read_text()))

    def save(self, path: Path = MODEL_PATH) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2) + "\n")

    def contributions(self, features: dict[str, float]) -> list[float]:
        return [
            weight * (features[name] - mean) / scale
            for name, mean, scale, weight in zip(self.features, self.means, self.scales, self.weights)
        ]

    def logit(self, features: dict[str, float]) -> float:
        return self.intercept + sum(self.contributions(features))

    def sentence_probability(self, logit: float) -> float:
        slope, offset = self.sentence_calibration
        return sigmoid(slope * logit + offset)

    def document_probability(self, logits: list[float]) -> float:
        slope, offset = self.document_calibration
        return sigmoid(slope * sum(logits) / len(logits) + offset)

    def reasons(self, features: dict[str, float]) -> list[Reason]:
        """Writer-facing reasons for the features that push this sentence furthest toward AI, strongest first."""
        ranked = sorted(zip(self.contributions(features), self.features, self.means), reverse=True)
        reasons = []
        for contribution, name, mean in ranked:
            if contribution <= 0 or len(reasons) == NUM_REASONS:
                break
            if name in NON_REASONS:
                continue
            yes_reason, no_reason = REASON_BY_NAME[name]
            reasons.append(Reason(yes_reason if features[name] > mean else no_reason, ASK_BY_NAME[name], round(contribution, 3)))
        return reasons
