"""Coworker reactions: an emoji on a finished sentence that lands, and a check-in when the writer types keyboard mash.

The gate questions ride in the note gate's Jev payload. A line past the gate costs one more round: the local model proposes emoji and Jev picks one.
"""

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import Field, create_model

from flow import feed, generator, jev
from text_processing import Sentence

if TYPE_CHECKING:
    from flow.decide import Note

REACT_QUESTION = "Would a friend texting you react to this line with an emoji?"
REACTION_THRESHOLD = 0.86
"""A line gets emoji candidates only when `REACT_QUESTION` reaches this: a joke, a sweet line and a bold claim score 0.88 to 0.94, plain lines at most 0.85."""
PICK_FLOOR = 0.5
"""The emoji Jev picks is shown only when its P reaches this."""
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
    """Jev's P on the picked emoji among the local model's candidates."""


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

sentence_react: dict[str, float] = {}
"""P(yes) on `REACT_QUESTION` for each sentence, filled by the note gate."""
sentence_reactions: dict[tuple[str, str], tuple[str, float] | None] = {}
"""The picked emoji and its P for each (sentence, goal) past the gate, or None when no candidate reaches `PICK_FLOOR`."""


def fields() -> dict:
    """The gate questions, as pydantic fields to add to the note gate."""
    return {"mash": (Literal["yes", "no"], Field(description=MASH_QUESTION)), "react": (Literal["yes", "no"], Field(description=REACT_QUESTION))}


def record(sentence: str, gate: dict[str, float | str]) -> None:
    """Store P(mash) and P(react) for `sentence` from the note gate's answers."""
    sentence_mash[sentence] = gate["mash"]
    sentence_react[sentence] = gate["react"]


async def pick(sentence: str, goal: str) -> tuple[str, float] | None:
    """The emoji Jev picks for `sentence` from the local model's candidates, with its P, or None when no candidate reaches `PICK_FLOOR`."""
    candidates = await generator.emoji_candidates(sentence, goal)
    if not candidates:
        return None
    Pick = create_model(
        "ReactionPick",
        __doc__=f"A writer drafting {goal or 'a text'} wrote a line.",
        emoji=(Literal[tuple(candidates)], Field(description="Which emoji would a friend texting you react to this line with?")),
    )
    result, answers = await feed.ask(jev.agent, f"Line: {sentence}", Pick)
    option, probability = max(result.response.provider_details["probabilities"]["emoji"].items(), key=lambda item: item[1])
    feed.act(answers, "shown" if probability >= PICK_FLOOR else "dropped", "emoji")
    return (option.split()[0], probability) if probability >= PICK_FLOOR else None


def unnoted(sentences: list[Sentence], notes: list["Note"]) -> list[Sentence]:
    """The sentences that hold no note."""
    return [sentence for sentence in sentences if not any(note.start < sentence.start + len(sentence.text) and sentence.start < note.end for note in notes)]


async def find(sentences: list[Sentence], goal: str, notes: list["Note"]) -> list[Reaction]:
    """The reactions in the draft, skipping every sentence that holds a note or is mash; lines past the gate get their pick round once."""
    gated = [
        sentence for sentence in unnoted(sentences, notes)
        if sentence_react.get(sentence.text.rstrip("."), 0) >= REACTION_THRESHOLD and sentence_mash.get(sentence.text.rstrip("."), 0) < MASH_THRESHOLD
    ]
    fresh = list(dict.fromkeys(sentence.text.rstrip(".") for sentence in gated if (sentence.text.rstrip("."), goal) not in sentence_reactions))
    for text, reaction in zip(fresh, await asyncio.gather(*(pick(text, goal) for text in fresh))):
        sentence_reactions[text, goal] = reaction
    reactions = []
    for sentence in gated:
        reaction = sentence_reactions[sentence.text.rstrip("."), goal]
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
