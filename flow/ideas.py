"""Ideas: when the writer asks for ideas or inspiration, the local model proposes angles to write about and Jev keeps the ones that fit the Goal, are specific to this writer and are worth posting.

The ask rides in the note gate's Jev payload; the angles cost one generator call and one Jev round.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import Field, create_model

from flow import decide, feed, generator

if TYPE_CHECKING:
    from flow.decide import Measure

NUM_SAMPLES = 3
NUM_SHOWN = 3
IDEA_FLOOR = 0.5
"""An angle is shown only when Jev's answers on `FLOORED` reach this; on a one-line draft every angle scores at most 0.42 on "specific", so it only ranks."""
FLOORED = ("fits", "would_post")
CHECKS = {
    "fits": "Does the angle '{angle}' fit {goal}?",
    "specific": "Is the angle '{angle}' specific to this writer, not generic?",
    "would_post": "Would the writer post about '{angle}'?",
}


@dataclass(frozen=True)
class Idea:
    text: str
    probability: float
    """The product of Jev's three answers on the angle."""
    measures: list["Measure"]


def fields() -> dict:
    """The ideas question, as a pydantic field to add to the note gate."""
    return {"wants_ideas": (Literal["yes", "no"], Field(description="Is the writer asking the assistant for ideas or inspiration for what to write?"))}


async def find(text: str, goal: str, context: str, shown: list[str]) -> list[Idea]:
    """The best `NUM_SHOWN` angles for the draft `text`, leaving out `shown`; empty when none passes `IDEA_FLOOR`."""
    angles = [angle for angle in dict.fromkeys(await generator.ideas(text, goal, context, NUM_SAMPLES)) if angle not in shown]
    if not angles:
        return []
    questions = {f"{name}_{index}": template.format(angle=angle, goal=f"'{goal}'" if goal else "the draft") for index, angle in enumerate(angles) for name, template in CHECKS.items()}
    IdeaCheck = create_model(
        "IdeaCheck",
        __doc__=f"{context}A writer is stuck on their draft and asks for ideas. Goal: {goal or 'not set'}. Draft: '{text}'. A writing assistant proposes angles to write about.",
        **{field: (Literal["yes", "no"], Field(description=question)) for field, question in questions.items()},
    )
    check = await decide.run(IdeaCheck, f"Draft: {text}")
    scores = {index: math.prod(check[f"{name}_{index}"] for name in CHECKS) for index in range(len(angles)) if min(check[f"{name}_{index}"] for name in FLOORED) >= IDEA_FLOOR}
    kept = sorted(scores, key=scores.get, reverse=True)[:NUM_SHOWN]
    feed.act(check, "shown" if kept else "dropped", *(f"{name}_{index}" for index in kept or range(len(angles)) for name in CHECKS))
    return [Idea(angles[index], scores[index], [decide.Measure(name, questions[f"{name}_{index}"], check[f"{name}_{index}"]) for name in CHECKS]) for index in kept]
