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
    "corrections": "fixes small mistakes in finished sentences, keeping your voice",
    "gauges": "scores the draft on what matters for your Goal, as rings",
    "reactions": "drops an emoji on a line that lands, like a joke",
    "flags": "highlights sentences that read as AI-written",
    "memory": "remembers facts about you and recalls the ones that matter",
}
"""Name to one line on what the component does, as the coworker would say it."""
ComponentName = Literal["corrections", "gauges", "reactions", "flags", "memory"]

router = APIRouter()


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
        feed.act(gate, "applied", "toggles", "component", "turn")
    return toggles


def find(sentences: list[Sentence]) -> list[Toggle]:
    """The toggles the draft's sentences ask for."""
    toggles = []
    for sentence in sentences:
        toggle = sentence_toggles.get(sentence.text.rstrip("."))
        if toggle:
            toggles.append(Toggle(sentence.start, sentence.start + len(sentence.text), sentence.text, *toggle))
    return toggles
