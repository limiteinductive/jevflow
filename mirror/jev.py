"""Ask the question bank about sentences through Jev, with an on-disk cache.

Each sentence costs one Jev request: the state is the sentence with its neighbors, and the
fields are every question in the bank. Answers are keyed by (model, bank, state) in
`data/build/jev.cache.jsonl`, so re-running replays them instead of paying again.
"""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, create_model
from pydantic_ai import Agent

from flow import feed
from mirror.questions import QUESTIONS

MODEL = "typesafe:jev-latest"
PRICE_PER_MILLION_INPUT_TOKENS = 0.042
CACHE_PATH = Path(__file__).parents[1] / "data" / "build" / "jev.cache.jsonl"
CONCURRENCY = 24

Bank = create_model(
    "Bank",
    __doc__="Answer each question about the sentence under 'Sentence:'. The 'Around it:' text is context only.",
    **{question.name: (Literal["yes", "no"], Field(description=question.text)) for question in QUESTIONS},
)
agent = Agent(MODEL, output_type=Bank)
"""Shared across calls: a fresh agent per call opens a new connection and adds about 0.4 s."""
BANK_KEY = hashlib.sha256(json.dumps(Bank.model_json_schema(), sort_keys=True).encode()).hexdigest()

_cache: dict[str, dict] = {}
if CACHE_PATH.exists():
    for line in CACHE_PATH.read_text().splitlines():
        record = json.loads(line)
        _cache[record["key"]] = record


def render(sentence: str, context: str) -> str:
    return f"Around it: {context}\n\nSentence: {sentence}"


async def ask(state: str, semaphore: asyncio.Semaphore) -> dict:
    """Return {"key", "probabilities": {name: P(yes)}, "input_tokens", "seconds"} for one state, or {"error"} on failure (not cached)."""
    key = hashlib.sha256(f"{MODEL}\n{BANK_KEY}\n{state}".encode()).hexdigest()
    if key in _cache:
        feed.record(state, Bank, {name: {"yes": probability, "no": 1 - probability} for name, probability in _cache[key]["probabilities"].items()}, None, {}, 0, 0)
        return _cache[key]
    async with semaphore:
        loop = asyncio.get_running_loop()
        start = loop.time()
        try:
            result, _ = await feed.ask(agent, state)
        except Exception as error:
            return {"key": key, "error": f"{type(error).__name__}: {error}"[:500]}
        seconds = loop.time() - start
    probabilities = result.response.provider_details["probabilities"]
    record = {
        "key": key,
        "probabilities": {question.name: probabilities[question.name]["yes"] for question in QUESTIONS},
        "input_tokens": result.usage.input_tokens,
        "seconds": seconds,
    }
    _cache[key] = record
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    return record


async def ask_many(states: list[str]) -> list[dict]:
    """Ask the bank about every state, `CONCURRENCY` requests at a time, in order."""
    semaphore = asyncio.Semaphore(CONCURRENCY)
    return await asyncio.gather(*(ask(state, semaphore) for state in states))


def dollars(input_tokens: int) -> float:
    return input_tokens * PRICE_PER_MILLION_INPUT_TOKENS / 1e6
