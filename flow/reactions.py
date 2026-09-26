"""Coworker reactions: an emoji on a finished sentence that lands, like 😂 on the joke, and a check-in when the writer types keyboard mash.

The reaction questions ride in the note gate's Jev payload, so a reaction costs no extra request.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from flow import generator
from text_processing import Sentence

if TYPE_CHECKING:
    from flow.decide import Note

REACTION_THRESHOLD = 0.8
"""A sentence gets an emoji only when its reaction question reaches this, so reactions stay rare."""
QUESTIONS = {"funny": "a joke", "sharp": "a sharp point"}
"""One yes/no question per reaction, "Is this line <property>?". Jev calls most plain lines clear, and a line judged alone reads as confusing when it calls back to the line before, so neither is a reaction."""
EMOJIS = {"funny": "😂", "sharp": "🔥"}
MASH_QUESTION = "Is this keyboard mashing or gibberish rather than words in any language?"
MASH_THRESHOLD = 0.8
"""A sentence is keyboard mash when `MASH_QUESTION` reaches this; it then gets a check-in instead of an emoji. Mash scores 0.83 to 1.00; real, non-English and typo-heavy sentences at most 0.01."""


@dataclass(frozen=True)
class Reaction:
    start: int
    end: int
    text: str
    emoji: str
    probability: float
    """Jev's P(yes) on the reaction's question."""


@dataclass(frozen=True)
class Mash:
    start: int
    end: int
    text: str
    comment: str
    """The coworker's check-in, written once per stretch of mash."""
    probability: float
    """The highest P(yes) on `MASH_QUESTION` in the stretch."""
    question: str


sentence_mash: dict[str, float] = {}
"""P(yes) on `MASH_QUESTION` for each sentence, filled by the note gate."""
mash_comments: dict[str, str] = {}
"""The check-in for each stretch of mash, keyed by the stretch's first sentence so a growing stretch keeps its comment."""

sentence_reactions: dict[tuple[str, str], tuple[str, float] | None] = {}
"""The emoji and its probability for each (sentence, goal), or None; filled by the note gate, which decides each sentence once."""


def fields(goal: str) -> dict:
    """The reaction questions, as pydantic fields to add to the note gate."""
    setting = f"In {goal}, is" if goal else "Is"
    return {"mash": (Literal["yes", "no"], Field(description=MASH_QUESTION))} | {name: (Literal["yes", "no"], Field(description=f"{setting} this line {property}?")) for name, property in QUESTIONS.items()}


def record(sentence: str, goal: str, gate: dict[str, float | str]) -> None:
    """Store P(mash) for `sentence`, and the emoji whose question scores highest with its probability, or None when none reaches `REACTION_THRESHOLD` or the sentence is mash."""
    sentence_mash[sentence] = gate["mash"]
    name = max(QUESTIONS, key=lambda name: gate[name])
    sentence_reactions[sentence, goal] = (EMOJIS[name], gate[name]) if gate[name] >= REACTION_THRESHOLD and gate["mash"] < MASH_THRESHOLD else None


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


async def find_mash(sentences: list[Sentence], goal: str) -> list[Mash]:
    """Each stretch of consecutive mash sentences as one span with the coworker's check-in, which the local model writes once per stretch."""
    stretches: list[list[Sentence]] = []
    previous = -2
    for position, sentence in enumerate(sentences):
        if sentence_mash.get(sentence.text.rstrip("."), 0) < MASH_THRESHOLD:
            continue
        if position == previous + 1:
            stretches[-1].append(sentence)
        else:
            stretches.append([sentence])
        previous = position
    mash = []
    for stretch in stretches:
        first, last = stretch[0], stretch[-1]
        end = last.start + len(last.text)
        if first.text not in mash_comments:
            mash_comments[first.text] = await generator.mash_comment(first.text, goal)
        probability = max(sentence_mash[sentence.text.rstrip(".")] for sentence in stretch)
        mash.append(Mash(first.start, end, " ".join(sentence.text for sentence in stretch), mash_comments[first.text], probability, MASH_QUESTION))
    return mash
