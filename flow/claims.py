"""Claim checks: a finished sentence that states a checkable fact Jev reads as false gets a margin ring with P(true), and a one-click corrected version when Jev reads one as true.

The claim questions ride in the note gate's Jev payload; only a flagged claim costs a second round, run as a task so /notes never waits on it.
"""

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import Field, create_model

from flow import feed, generator, jev, reactions
from text_processing import Sentence

if TYPE_CHECKING:
    from flow.decide import Note

CLAIM_GATE = 0.5
FALSE_THRESHOLD = 0.3
"""A claim shows only when P(true) is below this: the demo's wrong claim scores 0.00, a true one 0.95 to 1.00, and an unverifiable one ('we shipped on friday') 0.59."""
SWAP_THRESHOLD = 0.8
"""The best correction is offered only when Jev's P(true) on it and P(it changes only wrong facts) both reach this."""
NUM_CORRECTIONS = 3


@dataclass(frozen=True)
class Correction:
    text: str
    probability: float
    """Jev's P(true) on the corrected sentence."""


@dataclass(frozen=True)
class Claim:
    start: int
    end: int
    text: str
    probability: float
    """Jev's P(true) on the sentence."""
    correction: Correction | None
    """The corrected sentence, once its round has finished and Jev accepted it."""


sentence_claims: dict[tuple[str, str], tuple[float, asyncio.Task[Correction | None]] | None] = {}
"""P(true) and the correction round for each (sentence, goal) Jev reads as a false claim, or None; filled by the note gate, which decides each sentence once."""


def fields(goal: str) -> dict:
    """The claim questions, as pydantic fields to add to the note gate."""
    return {
        "claim": (Literal["yes", "no"], Field(description="Does the sentence state a fact about the world that a knowledgeable stranger could check without knowing the writer?")),
        "true": (Literal["yes", "no"], Field(description="Is what the sentence states about the world true?")),
    }


def record(sentence: str, goal: str, gate: dict[str, float | str]) -> None:
    """Store P(true) and start the correction round when `sentence` is a checkable claim below `FALSE_THRESHOLD`, else None."""
    flagged = gate["claim"] >= CLAIM_GATE and gate["true"] < FALSE_THRESHOLD
    sentence_claims[sentence, goal] = (gate["true"], asyncio.create_task(correct(sentence))) if flagged else None


async def correct(sentence: str) -> Correction | None:
    """The local model's corrections of `sentence`, ranked by Jev's P(true) times P(it changes only wrong facts); the best one when both reach `SWAP_THRESHOLD`."""
    drafts = [draft for draft in dict.fromkeys(await generator.correct_claims(sentence, NUM_CORRECTIONS)) if draft and draft != sentence]
    if not drafts:
        return None
    CorrectionCheck = create_model(
        "CorrectionCheck",
        __doc__=f"A writer's sentence states a fact that may be wrong: '{sentence}'. A writing assistant proposes corrected versions.",
        **{
            name: (Literal["yes", "no"], Field(description=template.format(draft=draft)))
            for index, draft in enumerate(drafts)
            for name, template in ((f"true_{index}", "Is what '{draft}' states about the world true?"), (f"minimal_{index}", "Does '{draft}' keep the writer's sentence, changing only facts that were wrong?"))
        },
    )
    check = await jev.run(CorrectionCheck, f"Sentence: {sentence}")
    _, index = max((check[f"true_{index}"] * check[f"minimal_{index}"], index) for index in range(len(drafts)))
    accepted = min(check[f"true_{index}"], check[f"minimal_{index}"]) >= SWAP_THRESHOLD
    feed.act(check, "shown" if accepted else "dropped", f"true_{index}", f"minimal_{index}", component="claims")
    return Correction(drafts[index], check[f"true_{index}"]) if accepted else None


def find(sentences: list[Sentence], goal: str, notes: list["Note"]) -> list[Claim]:
    """The flagged claims in the draft, skipping every sentence that holds a note; a correction appears once its round has finished."""
    claims = []
    for sentence in reactions.unnoted(sentences, notes):
        claim = sentence.text.rstrip(".")
        flagged = sentence_claims.get((claim, goal))
        if not flagged:
            continue
        probability, round_task = flagged
        finished = round_task.done() and not round_task.cancelled() and round_task.exception() is None
        claims.append(Claim(sentence.start, sentence.start + len(claim), claim, probability, round_task.result() if finished else None))
    return claims
