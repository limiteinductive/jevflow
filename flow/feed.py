"""The live feed of Jev decisions: every Jev request goes through `ask`, which keeps it in an in-memory ring buffer the page polls.

Nothing is written to disk; the buffer holds the last `CAPACITY` requests since the server started.
"""

import contextvars
import itertools
import statistics
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel
from pydantic_ai import Agent

CAPACITY = 400
PRICE_PER_MILLION_INPUT_TOKENS = 0.042

Kind = Literal["yes/no", "score", "choice"]
Action = Literal["shown", "applied", "dropped", "silent"]


@dataclass(frozen=True)
class Origin:
    pause: int
    """Ordinal of the writer's typing pause, shared by every request the page sends in it."""
    endpoint: str


@dataclass
class Question:
    name: str
    kind: Kind
    text: str
    answer: float | str
    """P(yes) for a yes/no question, the level for a score, the chosen option for a choice."""
    distribution: dict[str, float]


@dataclass
class Record:
    id: int
    version: int
    pause: int
    endpoint: str
    schema: str
    questions: list[Question]
    started: float
    """Seconds since the server started."""
    seconds: float
    input_tokens: int
    action: Action | None = None
    decisive: list[str] = field(default_factory=list)
    """Names of the questions that decided `action`."""
    cached: bool = False
    """Replayed from a cache: no Jev request was made."""
    answers: dict[str, float | str] = field(default_factory=dict, repr=False)
    """The answers the caller got, matched by identity in `act`."""


OFFLINE = Origin(0, "offline")
origin: contextvars.ContextVar[Origin] = contextvars.ContextVar("origin")
records: deque[Record] = deque(maxlen=CAPACITY)
versions = itertools.count(1)
ids = itertools.count(1)
pauses: dict[str, int] = {}
decisions = 0
"""Jev questions answered since the server started, across every Jev request."""
input_tokens = 0
began = time.perf_counter()
router = APIRouter()


def enter(request: Request) -> None:
    """Tag the Jev requests made while serving `request` with its pause (the page's `X-Pause` header) and endpoint."""
    token = request.headers.get("x-pause") or f"request {len(pauses)}"
    pause = pauses.setdefault(token, len(pauses) + 1)
    origin.set(Origin(pause, request.url.path.strip("/")))


def record(schema: type[BaseModel], distributions: dict[str, dict[str, float]], output: BaseModel | None, scores: dict[str, float], seconds: float, tokens: int) -> dict[str, float | str]:
    """Keep one Jev request in the feed; returns P(yes) for every yes/no field and the chosen option for every other field.

    `output` is None for a cache replay, which holds yes/no fields only.
    """
    global decisions, input_tokens
    questions = [
        Question(
            name,
            "yes/no" if "yes" in distribution else "score" if name in scores else "choice",
            schema.model_fields[name].description or name,
            distribution["yes"] if "yes" in distribution else getattr(output, name),
            distribution,
        )
        for name, distribution in distributions.items()
    ]
    current = origin.get(OFFLINE)
    answers = {question.name: question.answer for question in questions}
    records.append(Record(next(ids), next(versions), current.pause, current.endpoint, schema.__name__, questions, time.perf_counter() - began - seconds, seconds, tokens, cached=output is None, answers=answers))
    decisions += 0 if output is None else len(questions)
    input_tokens += tokens
    return answers


async def ask(agent: Agent, prompt: str, output_type: type[BaseModel] | None = None) -> tuple[object, dict[str, float | str]]:
    """Run one Jev request and keep it in the feed; returns the agent result and the answers `record` returns."""
    started = time.perf_counter()
    result = await agent.run(prompt, output_type=output_type) if output_type else await agent.run(prompt)
    details = result.response.provider_details
    answers = record(output_type or agent.output_type, details["probabilities"], result.output, details.get("scores", {}), time.perf_counter() - started, result.usage.input_tokens)
    return result, answers


def act(answers: dict[str, float | str], action: Action, *decisive: str) -> None:
    """Mark what the page does with `answers` (as returned by `ask`) and which questions decided it."""
    for kept in reversed(records):
        if kept.answers is answers:
            kept.action, kept.decisive, kept.version = action, list(decisive), next(versions)
            return


@router.get("/feed")
async def feed(after: int = 0) -> dict:
    """Records created or acted on since version `after`, and the session totals; cache replays add no decisions, latency or cost."""
    latencies = [kept.seconds for kept in records if not kept.cached]
    return {
        "version": max((kept.version for kept in records), default=after),
        "records": [{**asdict(kept), "answers": None} for kept in records if kept.version > after],
        "decisions": decisions,
        "median_seconds": statistics.median(latencies) if latencies else 0,
        "dollars": input_tokens * PRICE_PER_MILLION_INPUT_TOKENS / 1e6,
    }
