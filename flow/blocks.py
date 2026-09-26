"""The blocks a writer's notes lift into, blocks created while writing, and the coworker's ideas.

The page is a stack of blocks: Goal (with Audience and Tone), Content, To do, Ideas, Open questions, and the blocks Jev creates for notes that fit none of them.
"""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field, create_model

from flow import decide, generator

BLOCKS = {
    "goal": "what kind of text the writer is writing, such as 'im writing a blog post'",
    "audience": "who it is for",
    "tone": "how it should sound",
    "to_do": "a task for later: something to add, check or fix",
    "ideas": "something the text could also mention or try",
    "undo": "asks to undo the last edit",
}
"""Block choices for a note, with the description Jev reads; "undo" reverts the last lift. Prose never reaches this choice: the plan question keeps it in Content."""
OTHER = "other"
"""The choice for a note that fits no block; the local model then names a new block."""
KEPT_QUESTION = "Is the sentence a note the writer keeps about the document, such as a list of characters or sources, rather than part of the text itself?"
KEPT_GATE = 0.7
"""A sentence the plan question misses, such as "characters: jonah is her older brother" (0.79 here, 0.06 on plan), is a note when `KEPT_QUESTION` reaches this; prose scores at most 0.07 and "Mara is the narrator of this story" 0.58."""
BLOCK_THRESHOLD = 0.6
MAX_IDEAS = 2
IDEA_THRESHOLD = 0.6
NAME_INSTRUCTIONS = (
    "The user message is a writer's note about their document. Reply with a short name, one or two words, for the part of the document the note belongs to. "
    "When the note starts with a label such as 'sources:', the label is the name."
)
NAME_EXAMPLES = [
    ("characters: mara is the narrator", "Characters"),
    ("timeline: beta in may, launch in june", "Timeline"),
    ("sources: the 2023 postmortem and the design doc", "Sources"),
]
IDEAS_INSTRUCTIONS = (
    "You are a coworker reading a writer's draft. Propose three short ideas for what the draft could add next, one per line, each under ten words. "
    "Reply with the three lines only."
)
IDEAS_EXAMPLES = [
    (
        "Goal: a blog post about our migration to Postgres\nDraft: We moved off MySQL last month. The cutover took four hours.",
        "why we left MySQL\nthe one query that got slower\nwhat we would do differently",
    )
]

YesNo = Literal["yes", "no"]

router = APIRouter()


class IdeasDraft(BaseModel):
    text: str
    goal: str
    known: list[str]
    """Ideas already in the Ideas block; Jev drops proposals that repeat them."""


def choice(names: list[str]) -> tuple[type, Field]:
    """The "which block does this belong to?" choice over the fixed blocks, the blocks created so far (`names`) and "other"."""
    described = "; ".join([*(f"{block}: {description}" for block, description in BLOCKS.items()), *(f"{name}: the writer's {name} block" for name in names), f"{OTHER}: a note that fits none of these, such as the characters of a story or the sources of a report"])
    return Literal[tuple([*BLOCKS, *names, OTHER])], Field(description=f"Which block of the document does the sentence belong to? {described}.")


async def propose_name(sentence: str) -> str:
    """A short block name for `sentence` from the local model, asked for every note in parallel with Jev's gate so a new block costs no extra round."""
    return (await generator.chat(NAME_INSTRUCTIONS, NAME_EXAMPLES, sentence, 8)).strip("'\".:").title()


async def resolve(block: str, sentence: str, name: str, names: list[str]) -> str:
    """`block` unchanged unless it is `OTHER`; then `name` when Jev says it is a distinct part of the document worth its own section, else `OTHER`. A note in a created block is its whole sentence."""
    if block != OTHER:
        return block
    if not name or name.lower() in (*BLOCKS, *(known.lower() for known in names)):
        return OTHER
    BlockCheck = create_model(
        "BlockCheck",
        __doc__=f"A writer keeps notes about their document next to the text. The note: '{sentence}'.",
        distinct=(YesNo, Field(description=f"Is '{name}' a distinct part of the document, worth its own section?")),
    )
    check = await decide.run(BlockCheck, f"Note: {sentence}\n\nBlock: {name}")
    return name if check["distinct"] >= BLOCK_THRESHOLD else OTHER


@router.post("/ideas")
async def ideas(draft: IdeasDraft) -> dict:
    """At most `MAX_IDEAS` of the local model's three content ideas: Jev keeps those that fit the goal and are new relative to the draft and the Ideas block."""
    reply = await generator.chat(IDEAS_INSTRUCTIONS, IDEAS_EXAMPLES, f"Goal: {draft.goal or 'unknown'}\nDraft: {draft.text}", 60, 0.7)
    proposals = list(dict.fromkeys(line.strip("-*•0123456789. ").strip() for line in reply.splitlines() if line.strip()))[:3]
    if not proposals:
        return {"ideas": []}
    fields = {}
    for index, proposal in enumerate(proposals):
        fields[f"fits_{index}"] = (YesNo, Field(description=f"Does '{proposal}' fit the goal of the draft?"))
        fields[f"new_{index}"] = (YesNo, Field(description=f"Is '{proposal}' new relative to the draft and the ideas already noted?"))
    IdeaCheck = create_model("IdeaCheck", __doc__=f"A coworker proposes ideas for a writer's draft. Goal: '{draft.goal or 'unknown'}'.", **fields)
    known = "; ".join(draft.known) or "none"
    check = await decide.run(IdeaCheck, f"Draft: {draft.text}\n\nIdeas already noted: {known}")
    ranked = sorted(((min(check[f"fits_{index}"], check[f"new_{index}"]), proposal) for index, proposal in enumerate(proposals)), reverse=True)
    return {"ideas": [proposal for score, proposal in ranked[:MAX_IDEAS] if score >= IDEA_THRESHOLD]}
