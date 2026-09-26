"""The one Jev agent and the call every module makes through it."""

from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent

from flow import feed

MODEL = "typesafe:jev-latest"

YesNo = Literal["yes", "no"]

agent = Agent(MODEL)
"""One agent for every decision: a fresh agent per call opens a new connection and adds about 0.4 s."""


async def run(output_type: type[BaseModel], prompt: str) -> dict[str, float | str]:
    """P(yes) for every yes/no field, and the chosen option for every other field; `feed.act` marks what the page does with them."""
    _, answers = await feed.ask(agent, prompt, output_type)
    return answers
