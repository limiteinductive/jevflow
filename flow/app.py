"""The writing flow: one page with one text box. Jev reads and decides; the LLM only edits.

Run: `uv run python -m flow.app`, then open http://127.0.0.1:8000
"""

import asyncio
import time
from dataclasses import asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from flow import decide, feed, gauges, memory, reactions
from mirror.model import Stacker
from mirror.scan import scan

PORT = 8000
INSIGHT_THRESHOLD = 0.5

app = FastAPI()
app.include_router(gauges.router)
app.include_router(feed.router)
app.include_router(memory.router)
stacker = Stacker.load()


class Draft(BaseModel):
    text: str


class CommentedDraft(BaseModel):
    text: str
    comment: str
    """The coworker's open comment, or empty; a sentence can be a reply to it."""
    breaks: list[int]
    """Offsets where a sentence must end even without punctuation: the end of the open thread's anchor."""
    goal: str
    """The Goal header, or empty; it conditions the reaction questions."""


class Question(BaseModel):
    sentence: str
    paragraph: str
    """The text from the start of the question's line or paragraph through `sentence`."""
    question: str
    goal: str


class Reply(BaseModel):
    sentence: str
    comment: str
    reply: str
    proposed: str = ""
    """A rewrite the comment already showed, checked instead of a fresh one."""


class ComponentNote(BaseModel):
    span: str
    question: str
    """`span` without the pasted reference."""
    component: str
    """The clicked component's data and visible text, as the page snapshotted them."""
    sentence: str
    paragraph: str
    goal: str


class TypedDraft(BaseModel):
    text: str
    limit: int
    """Offset where the sentence being typed starts; only text before it is edited."""
    goal: str = ""
    """The Goal header, or empty; its gauge categories drive the goal suggestions."""


@app.get("/")
async def page() -> FileResponse:
    return FileResponse(Path(__file__).with_name("index.html"), headers={"Cache-Control": "no-store"})


@app.post("/insights")
async def insights(draft: Draft) -> dict:
    """Every sentence at or above `INSIGHT_THRESHOLD` with its P(reads as AI) and reason labels."""
    result = await scan(draft.text, stacker)
    return {
        "insights": [
            {"start": finding.start, "end": finding.end, "probability": finding.probability, "reasons": [reason.label for reason in finding.reasons], "ask": next((reason.ask for reason in finding.reasons if reason.ask), "")}
            for finding in result.findings
            if finding.probability >= INSIGHT_THRESHOLD and finding.reasons
        ]
    }


@app.middleware("http")
async def log_decisions(request: Request, call_next):
    """Tag the request's Jev calls for the feed, and print how many Jev questions it asked and how long it took."""
    if request.method == "POST":
        feed.enter(request)
    decisions, began = feed.decisions, time.perf_counter()
    response = await call_next(request)
    if request.method == "POST":
        print(f"{request.url.path}: {feed.decisions - decisions} Jev decisions in {time.perf_counter() - began:.2f} s", flush=True)
    return response


@app.post("/notes")
async def notes(draft: CommentedDraft) -> dict:
    found, timing = await asyncio.gather(decide.find_notes(draft.text, draft.comment, draft.breaks, draft.goal), decide.timing(draft.text))
    return {"notes": [asdict(note) for note in found], "timing": asdict(timing), "reactions": [asdict(reaction) for reaction in reactions.find(decide.line_sentences(draft.text, draft.breaks), draft.goal, found)]}


@app.post("/answer")
async def answer(question: Question) -> dict:
    return asdict(await decide.answer(question.sentence, question.paragraph, question.question, question.goal, memory.context()))


@app.post("/revise")
async def revise(reply: Reply) -> dict:
    return {"replacement": await decide.revise(reply.sentence, reply.comment, reply.reply, reply.proposed, memory.context())}


@app.post("/component")
async def component(note: ComponentNote) -> dict:
    return asdict(await decide.about_component(note.span, note.question, note.component, note.sentence, note.paragraph, note.goal, memory.context()))


class Probe(BaseModel):
    question: str
    text: str


@app.post("/probe")
async def probe(probe: Probe) -> dict:
    answers = await decide.probe(probe.question, probe.text)
    feed.act(answers, "shown", "answer")
    return {"probability": answers["answer"]}


@app.post("/fixes")
async def fixes(draft: TypedDraft) -> dict:
    goal, categories = draft.goal.strip(), []
    if goal:
        try:
            categories = await gauges.categories(goal, draft.text)
        except Exception as error:
            print(f"/fixes: no goal categories ({error!r}); corrections only", flush=True)
    fixes, suggestions = await decide.find_fixes(draft.text, draft.limit, goal if categories else "", categories)
    return {"fixes": [asdict(fix) for fix in fixes], "suggestions": [asdict(suggestion) for suggestion in suggestions]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
