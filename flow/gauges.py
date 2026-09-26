"""Live impact gauges for the Goal: the local LLM proposes categories for that kind of text, Jev picks the ones it succeeds or fails on and scores the draft on each.

The LLM only names categories. Every Jev question comes from one fixed template, so no LLM wording reaches Jev.
"""

import asyncio
import re
import time
from dataclasses import dataclass

from fastapi import APIRouter
from pydantic import BaseModel, Field, create_model

from flow import decide, generator

NUM_CANDIDATES = 8
NUM_GAUGES = 5
X_POST_LIMIT = 280
X_POST = re.compile(r"\b(tweet|x post|twitter post)\b", re.IGNORECASE)
"""A goal for one post on X; a thread matches none of these words and gets no limit."""
THREAD = re.compile(r"\bthread\b", re.IGNORECASE)
LIST_MARKER = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")
FIRST_SENTENCE = re.compile(r"\S.*?[.!?\n]", re.DOTALL)
MAX_BRIEFS = 64
MAX_DRAFTS = 512
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
"""Gauge categories by brief, picked from the draft's first finished sentence, as tasks so concurrent pauses share one LLM and Jev round."""


@dataclass(frozen=True)
class Scores:
    probabilities: list[float]
    """P(yes) for each category's question, in category order."""
    seconds: float
    """How long the Jev round took."""


draft_scores: dict[tuple[Brief, str], Scores] = {}
"""Scores by (brief, draft text), oldest first so the cache drops the oldest draft."""


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
    return [Category(name, f"Is this {brief.goal} {name}?") for _, name in ranked]


async def score(text: str, brief: Brief, categories: list[Category]) -> Scores:
    """Every category's question about the draft, all in one Jev payload."""
    Impact = create_model(
        "Impact",
        __doc__=f"A writer is drafting {brief.kind()}.",
        **{f"category_{index}": (decide.YesNo, Field(description=category.question)) for index, category in enumerate(categories)},
    )
    began = time.perf_counter()
    check = await decide.run(Impact, f"Draft: {text}")
    return Scores([check[f"category_{index}"] for index in range(len(categories))], time.perf_counter() - began)


@router.post("/gauges")
async def gauges(draft: GaugeDraft) -> dict:
    """The gauges for the draft under its brief, and the character limit when the goal is one X post.

    Empty until the draft has a finished first sentence or line, which picks the categories.
    """
    brief, text, opening = Brief(draft.goal.strip(), draft.audience.strip(), draft.tone.strip()), draft.text.strip(), FIRST_SENTENCE.search(draft.text)
    if not brief.goal or not text or (brief not in brief_categories and not opening):
        return {"gauges": [], "limit": None, "seconds": None}
    if brief not in brief_categories:
        brief_categories[brief] = asyncio.create_task(choose_categories(brief, opening.group().strip()))
        if len(brief_categories) > MAX_BRIEFS:
            brief_categories.pop(next(iter(brief_categories)))
    try:
        categories = await brief_categories[brief]
    except Exception:
        brief_categories.pop(brief, None)
        raise
    if (brief, text) not in draft_scores:
        draft_scores[brief, text] = await score(text, brief, categories)
        if len(draft_scores) > MAX_DRAFTS:
            draft_scores.pop(next(iter(draft_scores)))
    scores = draft_scores[brief, text]
    return {
        "gauges": [{"name": category.name, "question": category.question, "probability": probability} for category, probability in zip(categories, scores.probabilities)],
        "limit": X_POST_LIMIT if X_POST.search(brief.goal) and not THREAD.search(brief.goal) else None,
        "seconds": scores.seconds,
    }
