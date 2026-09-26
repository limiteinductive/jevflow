"""Coworker reactions: an emoji on a finished sentence that lands, like 😂 on the joke.

The reaction questions ride in the note gate's Jev payload, so a reaction costs no extra request.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from text_processing import Sentence

if TYPE_CHECKING:
    from flow.decide import Note

REACTION_THRESHOLD = 0.8
"""A sentence gets an emoji only when its reaction question reaches this, so reactions stay rare."""
QUESTIONS = {"funny": "a joke", "sharp": "a sharp point"}
"""One yes/no question per reaction, "Is this line <property>?". Jev calls most plain lines clear, and a line judged alone reads as confusing when it calls back to the line before, so neither is a reaction."""
EMOJIS = {"funny": "😂", "sharp": "🔥"}


@dataclass(frozen=True)
class Reaction:
    start: int
    end: int
    text: str
    emoji: str
    probability: float
    """Jev's P(yes) on the reaction's question."""


sentence_reactions: dict[tuple[str, str], tuple[str, float] | None] = {}
"""The emoji and its probability for each (sentence, goal), or None; filled by the note gate, which decides each sentence once."""


def fields(goal: str) -> dict:
    """The reaction questions, as pydantic fields to add to the note gate."""
    setting = f"In {goal}, is" if goal else "Is"
    return {name: (Literal["yes", "no"], Field(description=f"{setting} this line {property}?")) for name, property in QUESTIONS.items()}


def record(sentence: str, goal: str, gate: dict[str, float | str]) -> None:
    """Store the emoji whose question scores highest for `sentence` with its probability, or None when none reaches `REACTION_THRESHOLD`."""
    name = max(QUESTIONS, key=lambda name: gate[name])
    sentence_reactions[sentence, goal] = (EMOJIS[name], gate[name]) if gate[name] >= REACTION_THRESHOLD else None


def unnoted(sentences: list[Sentence], notes: list["Note"]) -> list[Sentence]:
    """The sentences that hold no note."""
    return [sentence for sentence in sentences if not any(note.start < sentence.start + len(sentence.text) and sentence.start < note.end for note in notes)]


def find(sentences: list[Sentence], goal: str, notes: list["Note"]) -> list[Reaction]:
    """The reactions in the draft, skipping every sentence that holds a note."""
    reactions = []
    for sentence in unnoted(sentences, notes):
        reaction = sentence_reactions.get((sentence.text.rstrip("."), goal))
        if reaction:
            reactions.append(Reaction(sentence.start, sentence.start + len(sentence.text), sentence.text, *reaction))
    return reactions
