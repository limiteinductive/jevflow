"""Diagrams: a small flowchart beside a finished paragraph that explains how something works, is organized or changed.

Jev gates each paragraph once. A paragraph past the gate costs one generator call and one Jev round: the local model proposes Mermaid flowcharts and Jev picks one or none.
"""

import asyncio
import re
from dataclasses import asdict, dataclass
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field, create_model

from flow import feed, generator, jev

DIAGRAM_QUESTION = "Does the passage explain how a system, process or organization works or how it changed, through concrete steps, parts or stages and how they connect, that a small diagram could draw?"
DIAGRAM_GATE = 0.85
"""A paragraph gets diagram candidates only when `DIAGRAM_QUESTION` reaches this: pipelines, hierarchies and before-and-afters score 0.90 to 1.00, stories, arguments and casual lines at most 0.21."""
FIT_FLOOR = 0.5
"""The likeliest flowchart is shown only when 1 minus Jev's P(`NONE`) reaches this: near-equal candidates split the rest (0.39, 0.38, 0.21 with `NONE` at 0.02), and wrong ones leave `NONE` at 0.96 to 0.99."""
NUM_CANDIDATES = 3
NONE = "none of these"
INSTRUCTIONS = (
    "The user gives a passage a writer wrote. Draw it as a small Mermaid flowchart of its steps, parts or before and after: 3 to 6 nodes, each labeled with 1 to 4 plain words from the passage. "
    "Write each flowchart on one line: 'flowchart TD;' then its statements separated by semicolons, such as A[label] --> B[label]."
)
EXAMPLES = [
    (
        "Passage: Every order takes the same path. The web app writes it to a queue, a worker picks it up and charges the card, and a second worker emails the receipt once the charge clears.",
        "1. flowchart TD; A[web app] --> B[queue]; B --> C[charge worker]; C --> D[receipt worker]\n"
        "2. flowchart TD; A[order] -->|written to| B[queue]; B -->|picked up| C[card charged]; C -->|charge clears| D[receipt emailed]\n"
        "3. flowchart TD; A[web app] --> B[queue]; B --> C{charge clears}; C --> D[email receipt]",
    ),
    (
        "Passage: The studio has three teams. Sound records the voices, animation builds the scenes on top of them, and render turns the scenes into frames overnight.",
        "1. flowchart TD; A[studio] --> B[sound]; A --> C[animation]; A --> D[render]\n"
        "2. flowchart TD; A[sound records voices] --> B[animation builds scenes]; B --> C[render makes frames]\n"
        "3. flowchart TD; A[voices] -->|sound| B[scenes]; B -->|animation| C[frames]; C -->|render overnight| D[film]",
    ),
]
LABEL = r"[^\[\](){}|\";#&<>]+"
NODE = rf"(?!end\b)[A-Za-z]\w*(?:\[{LABEL}\]|\({LABEL}\)|\{{{LABEL}\}})?"
STATEMENT = re.compile(rf"{NODE}(?: (?:-->|---|-\.->|==>)(?:\|{LABEL}\|)? {NODE})*")
"""One flowchart statement in the subset Mermaid parses: nodes with plain labels, joined by arrows with an optional edge label."""
PARAGRAPH = re.compile(r"\S(?:[^\n]*\S)?")
"""One line of the draft without its surrounding spaces; the page ends a paragraph at a newline."""


@dataclass(frozen=True)
class Diagram:
    start: int
    end: int
    text: str
    mermaid: str
    """The picked flowchart as one line of Mermaid."""
    probability: float
    """Jev's P that one of the local model's flowcharts fits the passage: 1 minus P(`NONE`)."""


paragraph_diagrams: dict[tuple[str, str], asyncio.Task[tuple[str, float] | None]] = {}
"""The round for each (paragraph, goal): the picked flowchart and its fit, or None; kept as a task so a pause that repeats the paragraph waits on the same round."""

router = APIRouter()


def parses(candidate: str) -> bool:
    """Whether `candidate` is a one-line `flowchart TD` whose every statement is in `STATEMENT`'s subset."""
    header, *statements = [statement.strip() for statement in candidate.rstrip(";").split(";")]
    return header == "flowchart TD" and bool(statements) and all(STATEMENT.fullmatch(statement) for statement in statements)


async def pick(paragraph: str, goal: str) -> tuple[str, float] | None:
    """Jev's gate on `paragraph`, then Jev's likeliest flowchart among the local model's candidates that parse, with its fit; None when the gate or `FIT_FLOOR` stops it."""
    context = f"A writer drafting {goal or 'a text'} wrote a passage."
    Gate = create_model("DiagramGate", __doc__=context, helps=(jev.YesNo, Field(description=DIAGRAM_QUESTION)))
    gate = await jev.run(Gate, f"Passage: {paragraph}")
    if gate["helps"] < DIAGRAM_GATE:
        feed.act(gate, "silent", "helps", component="diagrams")
        return None
    candidates = [candidate for candidate in await generator.candidates(INSTRUCTIONS, EXAMPLES, f"Passage: {paragraph}", NUM_CANDIDATES, 80) if parses(candidate)]
    if not candidates:
        feed.act(gate, "dropped", "helps", component="diagrams")
        return None
    Pick = create_model(
        "DiagramPick",
        __doc__=f"{context} A writing assistant proposes small Mermaid flowcharts to show beside it.",
        diagram=(Literal[(*candidates, NONE)], Field(description="Which flowchart shows what the passage says most faithfully and clearly?")),
    )
    result, answers = await feed.ask(jev.agent, f"Passage: {paragraph}", Pick)
    distribution = result.response.provider_details["probabilities"]["diagram"]
    fits = 1 - distribution[NONE]
    shown = fits >= FIT_FLOOR
    feed.act(gate, "shown" if shown else "dropped", "helps", component="diagrams")
    feed.act(answers, "shown" if shown else "dropped", "diagram", component="diagrams")
    return (max(candidates, key=distribution.get), fits) if shown else None


class Draft(BaseModel):
    text: str
    """The draft's content text, without notes, meta sentences or references."""
    limit: int
    """Offset where the caret's line starts; only the paragraphs before it are finished."""
    goal: str


@router.post("/diagrams")
async def diagrams(draft: Draft) -> dict:
    """A flowchart for each finished paragraph that has one; each paragraph gets its round once per goal, and a failed round is forgotten so the next pause retries it."""
    paragraphs = [(match.start(), match.group()) for match in PARAGRAPH.finditer(draft.text, 0, draft.limit)]
    for _, paragraph in paragraphs:
        if (paragraph, draft.goal) not in paragraph_diagrams:
            paragraph_diagrams[paragraph, draft.goal] = asyncio.create_task(pick(paragraph, draft.goal))
    rounds = [paragraph_diagrams[paragraph, draft.goal] for _, paragraph in paragraphs]
    if rounds:
        await asyncio.wait(rounds)
    shown = []
    for (start, paragraph), round_task in zip(paragraphs, rounds):
        if round_task.exception():
            print(f"/diagrams: no round for {paragraph[:40]!r} ({round_task.exception()!r}); retried on the next pause", flush=True)
            paragraph_diagrams.pop((paragraph, draft.goal), None)
        elif round_task.result():
            shown.append(asdict(Diagram(start, start + len(paragraph), paragraph, *round_task.result())))
    return {"diagrams": shown}
