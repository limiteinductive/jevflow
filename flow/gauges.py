"""Live impact gauges for the Goal: the local LLM proposes categories for that kind of text, Jev picks the ones it succeeds or fails on and scores the draft on each.

The LLM only names categories. Every Jev question comes from one fixed template, so no LLM wording reaches Jev.
"""

import asyncio
import re
import time
from dataclasses import dataclass

from fastapi import APIRouter
from pydantic import BaseModel, Field, create_model

from flow import decide, feed, generator

NUM_CANDIDATES = 8
NUM_GAUGES = 5
LIMIT_THRESHOLD = 0.5
"""The LLM's proposed character limit is shown when Jev's P(the text should stay under it) reaches this."""
NUMBER = re.compile(r"\d+")
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

CHANGE_INSTRUCTIONS = (
    "A writer's draft is scored on the listed things, one per line, and the writer comments on the list. Rewrite the list as the comment asks, changing only what it asks for. "
    "Same format: one per line, each 1 to 4 plain words that finish the sentence 'The text is ...'. Reply with the list only."
)
CHANGE_EXAMPLES = [("List:\nfunny\nshort\nrelatable\n\nComment: swap short for surprising", "funny\nsurprising\nrelatable")]
CHANGE_THRESHOLD = 0.6
"""A changed list replaces the categories only when both of Jev's checks reach this."""

LIMIT_INSTRUCTIONS = (
    "The user names a kind of text a writer is drafting. If that kind of text has a hard or customary maximum length, "
    "reply with that maximum in characters, as a number only. If it has none, reply with none."
)
LIMIT_EXAMPLES = [("Goal: a text message", "160"), ("Goal: a cover letter", "none"), ("Goal: a LinkedIn headline", "220")]


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


@dataclass(frozen=True)
class Pick:
    categories: list[Category]
    limit: int | None
    """Character limit for this kind of text, or None when Jev does not confirm one."""


router = APIRouter()

brief_categories: dict[Brief, asyncio.Task[Pick]] = {}
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


async def choose_categories(brief: Brief, opening: str) -> Pick:
    """The `NUM_GAUGES` LLM candidates that Jev says readers of this kind of text care about most, strongest first, and the LLM's character limit if Jev confirms it.

    Jev sees the draft's `opening` as well as the brief: a bare "X post" does not need to be funny, a joke X post does.
    """
    reply, limit_reply = await asyncio.gather(
        generator.chat(CATEGORY_INSTRUCTIONS, CATEGORY_EXAMPLES, brief.prompt(), 100), generator.chat(LIMIT_INSTRUCTIONS, LIMIT_EXAMPLES, brief.prompt(), 8)
    )
    number = NUMBER.search(limit_reply)
    limit = int(number.group()) if number else None
    limit_field = {"limit": (decide.YesNo, Field(description=f"Should {brief.kind()} stay under {limit} characters?"))} if limit else {}
    candidates = list(dict.fromkeys(filter(None, (LIST_MARKER.sub("", line).strip().lower() for line in reply.splitlines()))))[:NUM_CANDIDATES]
    Matters = create_model(
        "Matters",
        __doc__=f"A writer is drafting {brief.kind()}.",
        **{
            f"category_{index}": (decide.YesNo, Field(description=f"Would readers of {brief.kind()} like the one started here care whether it is {name}?"))
            for index, name in enumerate(candidates)
        },
        **limit_field,
    )
    check = await decide.run(Matters, f"{brief.prompt()}\n\nStart of the draft: {opening}")
    ranked = sorted(((check[f"category_{index}"], name) for index, name in enumerate(candidates)), reverse=True)[:NUM_GAUGES]
    confirmed = limit if limit and check["limit"] >= LIMIT_THRESHOLD else None
    feed.act(check, "applied", *(f"category_{candidates.index(name)}" for _, name in ranked), *(["limit"] if confirmed else []))
    return Pick([Category(name, f"Is this {brief.goal} {name}?") for _, name in ranked], confirmed)


async def score(text: str, brief: Brief, categories: list[Category]) -> Scores:
    """Every category's question about the draft, all in one Jev payload."""
    Impact = create_model(
        "Impact",
        __doc__=f"A writer is drafting {brief.kind()}.",
        **{f"category_{index}": (decide.YesNo, Field(description=category.question)) for index, category in enumerate(categories)},
    )
    began = time.perf_counter()
    check = await decide.run(Impact, f"Draft: {text}")
    feed.act(check, "shown", *check)
    return Scores([check[f"category_{index}"] for index in range(len(categories))], time.perf_counter() - began)


async def pick(brief: Brief, text: str) -> Pick | None:
    """The categories and limit for `brief`, chosen once from the draft's first finished sentence; None until `text` has one.

    A failed round is forgotten so the next call retries it.
    """
    if brief not in brief_categories:
        opening = FIRST_SENTENCE.search(text)
        if not opening:
            return None
        brief_categories[brief] = asyncio.create_task(choose_categories(brief, opening.group().strip()))
        if len(brief_categories) > MAX_BRIEFS:
            brief_categories.pop(next(iter(brief_categories)))
    try:
        return await brief_categories[brief]
    except Exception:
        brief_categories.pop(brief, None)
        raise


class CategoryChange(BaseModel):
    goal: str
    audience: str
    tone: str
    comment: str


@router.post("/gauges/change")
async def change(request: CategoryChange) -> dict:
    """The categories rewritten as the writer's `comment` asks, applied when Jev says the rewrite does what it asks and keeps the rest; the next /gauges call scores the new ones.

    Also applies to the goal-only brief that corrections read, and drops the brief's cached scores.
    """
    brief = Brief(request.goal.strip(), request.audience.strip(), request.tone.strip())
    if brief not in brief_categories:
        return {"categories": None}
    old = await brief_categories[brief]
    names = [category.name for category in old.categories]
    reply = await generator.chat(CHANGE_INSTRUCTIONS, CHANGE_EXAMPLES, "List:\n" + "\n".join(names) + f"\n\nComment: {request.comment}", 60)
    proposed = list(dict.fromkeys(filter(None, (LIST_MARKER.sub("", line).strip().lower() for line in reply.splitlines()))))[:NUM_GAUGES]
    if not proposed or proposed == names:
        return {"categories": None}
    ChangeCheck = create_model(
        "ChangeCheck",
        __doc__=f"A writer's draft of {brief.kind()} is scored on: {'; '.join(names)}. The writer commented: '{request.comment}'. The new list: {'; '.join(proposed)}.",
        follows=(decide.YesNo, Field(description="Does the new list do what the writer's comment asks?")),
        keeps=(decide.YesNo, Field(description="Does the new list keep every item the comment does not ask to change?")),
    )
    check = await decide.run(ChangeCheck, f"Comment: {request.comment}\n\nOld: {'; '.join(names)}\n\nNew: {'; '.join(proposed)}")
    applied = min(check["follows"], check["keeps"]) >= CHANGE_THRESHOLD
    feed.act(check, "applied" if applied else "dropped", "follows", "keeps")
    if not applied:
        return {"categories": None}
    changed = Pick([Category(name, f"Is this {brief.goal} {name}?") for name in proposed], old.limit)
    for key in {brief, Brief(brief.goal, "", "")}:
        brief_categories[key] = asyncio.create_task(asyncio.sleep(0, changed))
    for key in [key for key in draft_scores if key[0].goal == brief.goal]:
        draft_scores.pop(key)
    return {"categories": proposed}


async def categories(goal: str, text: str) -> list[str]:
    """The category names for `goal` alone, empty until `text` has a finished first sentence."""
    chosen = await pick(Brief(goal, "", ""), text)
    return [category.name for category in chosen.categories] if chosen else []


@router.post("/gauges")
async def gauges(draft: GaugeDraft) -> dict:
    """The gauges for the draft under its brief, and the character limit for this kind of text, if any.

    Empty until the draft has a finished first sentence or line, which picks the categories.
    """
    brief, text = Brief(draft.goal.strip(), draft.audience.strip(), draft.tone.strip()), draft.text.strip()
    chosen = await pick(brief, draft.text) if brief.goal and text else None
    if not chosen:
        return {"gauges": [], "limit": None, "seconds": None}
    if (brief, text) not in draft_scores:
        draft_scores[brief, text] = await score(text, brief, chosen.categories)
        if len(draft_scores) > MAX_DRAFTS:
            draft_scores.pop(next(iter(draft_scores)))
    scores = draft_scores[brief, text]
    return {
        "gauges": [{"name": category.name, "question": category.question, "probability": probability} for category, probability in zip(chosen.categories, scores.probabilities)],
        "limit": chosen.limit,
        "seconds": scores.seconds,
    }
