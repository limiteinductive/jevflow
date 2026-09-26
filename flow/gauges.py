"""Live impact gauges for the Goal: the local LLM proposes categories for that kind of text, Jev picks the ones it succeeds or fails on and scores the draft on each.

The LLM only names categories. Every Jev question comes from one fixed template, so no LLM wording reaches Jev.
"""

import asyncio
import re
from dataclasses import dataclass

from fastapi import APIRouter
from pydantic import BaseModel, Field, create_model

from flow import decide, generator

NUM_CANDIDATES = 8
NUM_GAUGES = 5
X_POST_LIMIT = 280
X_POST = re.compile(r"\b(tweet|x post|x thread|twitter)\b", re.IGNORECASE)
LIST_MARKER = re.compile(r"^[\s*•\d.)-]+")
CATEGORY_INSTRUCTIONS = (
    f"The user names a kind of text a writer is drafting, and maybe its audience and tone. List {NUM_CANDIDATES} things that could decide how readers of that kind of text react to it, "
    "specific to that kind of text rather than generic virtues like clear or concise. One per line, each 1 to 4 plain words that finish the sentence 'The text is ...'. Reply with the list only."
)
CATEGORY_EXAMPLES = [
    ("Goal: a cover letter", "specific to the job\nconfident\nshort\nfree of cliches\nclear about the ask\nwarm\nproof of results\nfree of typos"),
    ("Goal: a meme caption", "funny\ngot in one second\nrelatable\nworth sharing\nsurprising\nshort\nin on the joke\ntimely"),
    ("Goal: a LinkedIn post\nAudience: recruiters", "strong first line\nconcrete\nhumble\nworth commenting on\neasy to skim\nshows results\nfree of buzzwords\npersonal"),
]


@dataclass(frozen=True)
class Brief:
    goal: str
    audience: str
    tone: str

    def prompt(self) -> str:
        return "\n".join(f"{name}: {value}" for name, value in (("Goal", self.goal), ("Audience", self.audience), ("Tone", self.tone)) if value)

    def kind(self) -> str:
        """The goal as a noun phrase with an article, and the audience, such as "an X post for founders"."""
        noun = self.goal if re.match(r"(a|an|the|my|our)\b", self.goal, re.IGNORECASE) else f"a {self.goal}"
        return f"{noun} for {self.audience}" if self.audience else noun


@dataclass(frozen=True)
class Category:
    name: str
    question: str
    """The Jev question that scores the draft on this category."""


router = APIRouter()

brief_categories: dict[Brief, asyncio.Task[list[Category]]] = {}
"""Gauge categories by brief, picked on the first pause with a draft, as tasks so concurrent pauses share one LLM and Jev round."""

draft_scores: dict[tuple[Brief, str], list[float]] = {}
"""P(yes) for each category's question by (brief, draft text)."""


class GaugeDraft(BaseModel):
    text: str
    goal: str
    audience: str = ""
    tone: str = ""


async def choose_categories(brief: Brief, opening: str) -> list[Category]:
    """The `NUM_GAUGES` LLM candidates that Jev says readers of this kind of text care about most, strongest first.

    Jev sees the draft's `opening` as well as the brief: a bare "X post" does not need to be funny, a joke X post does.
    """
    reply = await generator.chat(CATEGORY_INSTRUCTIONS, CATEGORY_EXAMPLES, brief.prompt(), 100)
    candidates = list(dict.fromkeys(filter(None, (LIST_MARKER.sub("", line).strip().lower() for line in reply.splitlines()))))[:NUM_CANDIDATES]
    Matters = create_model(
        "Matters",
        __doc__=f"A writer is drafting {brief.kind()}.",
        **{
            f"category_{index}": (decide.YesNo, Field(description=f"Would readers of {brief.kind()} like the one started here care whether it is {name}?"))
            for index, name in enumerate(candidates)
        },
    )
    check = await decide.run(Matters, f"{brief.prompt()}\n\nStart of the draft: {opening}")
    ranked = sorted(((check[f"category_{index}"], name) for index, name in enumerate(candidates)), reverse=True)[:NUM_GAUGES]
    return [Category(name, f"Is this {brief.goal} {name}?") for probability, name in ranked]


async def score(text: str, brief: Brief, categories: list[Category]) -> list[float]:
    """P(yes) for every category's question about the draft, all in one Jev payload."""
    Impact = create_model(
        "Impact",
        __doc__=f"A writer is drafting {brief.kind()}.",
        **{f"category_{index}": (decide.YesNo, Field(description=category.question)) for index, category in enumerate(categories)},
    )
    check = await decide.run(Impact, f"Draft: {text}")
    return [check[f"category_{index}"] for index in range(len(categories))]


@router.post("/gauges")
async def gauges(draft: GaugeDraft) -> dict:
    """The gauges for the draft under its brief, and the character limit when the goal is an X post; empty when there is no goal or no draft."""
    brief, text = Brief(draft.goal.strip(), draft.audience.strip(), draft.tone.strip()), draft.text.strip()
    if not brief.goal or not text:
        return {"gauges": [], "limit": None}
    if brief not in brief_categories:
        brief_categories[brief] = asyncio.create_task(choose_categories(brief, text))
    try:
        categories = await brief_categories[brief]
    except Exception:
        brief_categories.pop(brief, None)
        raise
    if (brief, text) not in draft_scores:
        draft_scores[brief, text] = await score(text, brief, categories)
    return {
        "gauges": [
            {"name": category.name, "question": category.question, "probability": probability} for category, probability in zip(categories, draft_scores[brief, text])
        ],
        "limit": X_POST_LIMIT if X_POST.search(brief.goal) else None,
    }
