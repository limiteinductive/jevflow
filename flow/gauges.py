"""Live impact gauges for the Goal: the local LLM names what decides how that kind of text lands, Jev keeps the ones that matter and scores the draft on each.

The LLM only names categories. Every Jev question comes from one fixed template, so no LLM wording reaches Jev.
"""

import asyncio
import re

from fastapi import APIRouter
from pydantic import BaseModel, Field, create_model

from flow import decide, feed, generator

NUM_CATEGORIES = 5
MIN_GAUGES = 4
KEEP_THRESHOLD = 0.5
"""A category stays when Jev's P(it matters for this kind of text) reaches this; the top `MIN_GAUGES` stay regardless."""
X_POST_LIMIT = 280
X_POST = re.compile(r"\b(tweet|x post|x thread|twitter)\b", re.IGNORECASE)
LIST_MARKER = re.compile(r"^[\s*\u2022\d.)-]+")
CATEGORY_INSTRUCTIONS = (
    f"The user names a kind of text a writer is drafting. List the {NUM_CATEGORIES} things that most decide how readers of that kind of text react to it, "
    "specific to that kind of text rather than generic virtues like clear or concise. One per line, each 1 to 4 plain words that finish the sentence 'The text is ...'. Reply with the list only."
)
CATEGORY_EXAMPLES = [
    ("a cover letter", "specific to the job\nconfident\nshort\nfree of cliches\nclear about the ask"),
    ("a meme caption", "funny\ngot in one second\nrelatable\nworth sharing\nsurprising"),
    ("a LinkedIn post", "strong first line\nconcrete\nhumble\nworth commenting on\neasy to skim"),
]

router = APIRouter()

goal_categories: dict[str, asyncio.Task[list[str]]] = {}
"""Kept categories by goal, as tasks so concurrent pauses share one LLM and Jev round."""

draft_scores: dict[tuple[str, str], list[float]] = {}
"""P(the draft has each category) by (goal, draft text)."""


class GaugeDraft(BaseModel):
    text: str
    goal: str


async def choose_categories(goal: str) -> list[str]:
    """The LLM's categories for `goal` that Jev says matter, strongest first."""
    reply = await generator.chat(CATEGORY_INSTRUCTIONS, CATEGORY_EXAMPLES, goal, 60)
    proposed = list(dict.fromkeys(filter(None, (LIST_MARKER.sub("", line).strip().lower() for line in reply.splitlines()))))[:NUM_CATEGORIES]
    Matters = create_model(
        "Matters",
        __doc__=f"A writer is drafting a text. Their goal: {goal}.",
        **{f"category_{index}": (decide.YesNo, Field(description=f"Does it matter for this kind of text that it is {category}?")) for index, category in enumerate(proposed)},
    )
    check = await decide.run(Matters, f"Goal: {goal}")
    ranked = sorted(((check[f"category_{index}"], category) for index, category in enumerate(proposed)), reverse=True)
    feed.act(check, "applied", *(name for name in check if check[name] >= KEEP_THRESHOLD))
    return [category for rank, (probability, category) in enumerate(ranked) if probability >= KEEP_THRESHOLD or rank < MIN_GAUGES]


async def score(text: str, goal: str, categories: list[str]) -> list[float]:
    """P(yes) that the draft is each category, all in one Jev payload."""
    Impact = create_model(
        "Impact",
        __doc__=f"A writer is drafting a text. Their goal: {goal}.",
        **{f"category_{index}": (decide.YesNo, Field(description=f"Is the draft {category}?")) for index, category in enumerate(categories)},
    )
    check = await decide.run(Impact, f"Draft: {text}")
    feed.act(check, "shown", *check)
    return [check[f"category_{index}"] for index in range(len(categories))]


@router.post("/gauges")
async def gauges(draft: GaugeDraft) -> dict:
    """The gauges for the draft under its goal, and the character limit when the goal is an X post; empty when there is no goal or no draft."""
    goal, text = draft.goal.strip(), draft.text.strip()
    if not goal or not text:
        return {"gauges": [], "limit": None}
    if goal not in goal_categories:
        goal_categories[goal] = asyncio.create_task(choose_categories(goal))
    try:
        categories = await goal_categories[goal]
    except Exception:
        goal_categories.pop(goal, None)
        raise
    if (goal, text) not in draft_scores:
        draft_scores[goal, text] = await score(text, goal, categories)
    return {
        "gauges": [{"name": category, "probability": probability} for category, probability in zip(categories, draft_scores[goal, text])],
        "limit": X_POST_LIMIT if X_POST.search(goal) else None,
    }
