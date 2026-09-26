"""The jevflow components a writer can turn on or off, each with one line on what it does; the page keeps the on/off flags.

A disabled component asks Jev nothing.
"""

from dataclasses import dataclass
from typing import Literal

from fastapi import APIRouter
from pydantic import Field

from flow import feed
from text_processing import Sentence

COMPONENTS = {
    "notes": "files what you say about the draft (goal, audience, tone, to do) as headers at the top",
    "replies": "answers your questions to me in the margin",
    "pages": "keeps your pages and opens a new one when you start writing something else",
    "corrections": "fixes small mistakes in finished sentences, keeping your voice",
    "gauges": "scores the draft on what matters for your Goal, as rings",
    "reactions": "drops an emoji on a line that lands, like a joke",
    "flags": "highlights sentences that read as AI-written",
    "memory": "remembers facts about you and recalls the ones that matter",
    "probes": "answers your own yes/no question about a copied selection, as a ring",
    "feed": "shows every decision live in a column on the right",
    "claims": "rings a claim that reads as false and offers the fix",
    "titles": "names each page after what it says, so no two pages look alike in the sidebar",
    "ideas": "gives you angles to write about when you ask for inspiration",
    "diagrams": "draws a small flowchart beside a paragraph that explains how something works",
}
"""Name to one line on what the component does, as the coworker would say it. A new component adds its line here."""
ComponentName = Literal[tuple(COMPONENTS)]

router = APIRouter()


def context(enabled: frozenset[str]) -> str:
    """The components and their on/off state, as lines that prefix the coworker's reply prompt."""
    on = "; ".join(f"{name}: {does}" for name, does in COMPONENTS.items() if name in enabled)
    off = ", ".join(name for name in COMPONENTS if name not in enabled) or "none"
    return f"Your features that are on: {on}\nYour features that are off: {off}\n"


@router.get("/components")
async def components() -> dict[str, str]:
    return COMPONENTS


TOGGLE_GATE = 0.55
"""A sentence toggles a component when Jev's P(it asks to turn one on or off) reaches this."""


@dataclass(frozen=True)
class Toggle:
    start: int
    end: int
    text: str
    component: ComponentName
    on: bool
    probability: float


sentence_toggles: dict[str, tuple[ComponentName, bool, float] | None] = {}
"""The toggle each sentence asks for, or None; filled by the note gate, which decides each sentence once."""


def fields() -> dict:
    """The toggle questions, as pydantic fields to add to the note gate."""
    listing = "; ".join(f"{name}: {does}" for name, does in COMPONENTS.items())
    return {
        "toggles": (Literal["yes", "no"], Field(description=f"Does the sentence ask to turn one of the writing assistant's features on or off? The features: {listing}.")),
        "component": (ComponentName, Field(description="Which feature does the sentence name?")),
        "turn": (Literal["on", "off"], Field(description="Does the sentence ask to turn it on or off?")),
    }


def record(sentence: str, gate: dict[str, float | str]) -> bool:
    """Store the toggle `sentence` asks for and mark it in the feed; True when it is one, so the sentence is not also a note."""
    toggles = gate["toggles"] >= TOGGLE_GATE
    sentence_toggles[sentence] = (gate["component"], gate["turn"] == "on", gate["toggles"]) if toggles else None
    if toggles:
        feed.act(gate, "applied", "toggles", "component", "turn", component=None)
    return toggles


def find(sentences: list[Sentence]) -> list[Toggle]:
    """The toggles the draft's sentences ask for."""
    toggles = []
    for sentence in sentences:
        toggle = sentence_toggles.get(sentence.text.rstrip("."))
        if toggle:
            toggles.append(Toggle(sentence.start, sentence.start + len(sentence.text), sentence.text, *toggle))
    return toggles
