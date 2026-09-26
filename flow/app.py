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

from flow import decide, gauges
from mirror.model import Stacker
from mirror.scan import scan

PORT = 8000
INSIGHT_THRESHOLD = 0.5

app = FastAPI()
app.include_router(gauges.router)
stacker = Stacker.load()


class Draft(BaseModel):
    text: str


class CommentedDraft(BaseModel):
    text: str
    comment: str
    """The coworker's open comment, or empty; a sentence can be a reply to it."""
    breaks: list[int]
    """Offsets where a sentence must end even without punctuation: the end of the open thread's anchor."""


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


class TypedDraft(BaseModel):
    text: str
    limit: int
    """Offset where the sentence being typed starts; only text before it is edited."""


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
    """Print how many Jev questions each request asked and how long it took; /insights questions go through mirror.jev and are not counted."""
    decisions, began = decide.decisions, time.perf_counter()
    response = await call_next(request)
    if request.method == "POST":
        print(f"{request.url.path}: {decide.decisions - decisions} Jev decisions in {time.perf_counter() - began:.2f} s", flush=True)
    return response


@app.post("/notes")
async def notes(draft: CommentedDraft) -> dict:
    found, timing = await asyncio.gather(decide.find_notes(draft.text, draft.comment, draft.breaks), decide.timing(draft.text))
    return {"notes": [asdict(note) for note in found], "timing": asdict(timing)}


@app.post("/answer")
async def answer(question: Question) -> dict:
    return asdict(await decide.answer(question.sentence, question.paragraph, question.question, question.goal))


@app.post("/revise")
async def revise(reply: Reply) -> dict:
    return {"replacement": await decide.revise(reply.sentence, reply.comment, reply.reply)}


@app.post("/fixes")
async def fixes(draft: TypedDraft) -> dict:
    return {"fixes": [asdict(fix) for fix in await decide.find_fixes(draft.text, draft.limit)]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
