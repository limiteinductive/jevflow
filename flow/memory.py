"""What jevflow knows about the writer: markdown files in a local git repo outside the project, one dated fact per line.

Files are `profile.md` (always loaded), `preferences.md` and `people/<name>.md`; each starts with an `aliases:` line that the keyword pass matches against the draft.
Jev decides every write and every recall; the local model only writes the fact lines. Every write is a commit, so git history keeps the old facts.
"""

import asyncio
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, Field, create_model

from flow import decide, feed, generator

ROOT = Path.home() / ".jevflow" / "memory"
PROFILE = "profile"
KEEP_THRESHOLD = 0.7
"""Every write question must reach this: "a note" and "still true next week" on a sentence, "fair statement", and "rejected this kind of edit" for an undo."""
RELEVANT_THRESHOLD = 0.5
MAX_RECALLED = 4
MAX_DRAFT_CHARS = 600
FACT_INSTRUCTIONS = (
    "The user message is a sentence a writer typed; it holds a lasting fact or preference about the writer. Reply with three lines. "
    "Line 1: where it goes: profile (the writer's life and work), preferences (how they like their writing), or people/<first name in lowercase> (someone in their life). "
    "Line 2: comma-separated keywords a later draft about this would contain. Line 3: the fact as one short casual sentence addressed to the writer as you."
)
FACT_EXAMPLES = [
    ("my brother Tom hates long emails so keep it short", "people/tom\ntom, brother\nyour brother Tom hates long emails"),
    ("i always text in lowercase btw", "preferences\ntext, texting, message\nyou text in lowercase"),
    ("I'm a nurse and I write these after night shifts", "profile\nwork, shift, nurse\nyou're a nurse on night shifts"),
]
UNDO_INSTRUCTIONS = (
    "A writing assistant changed a sentence in the writer's draft and the writer undid the change. "
    "Before is the writer's own sentence; After is the assistant's change, which the writer rejected. Reply with two lines. Line 1: comma-separated keywords a later draft of this kind would contain. "
    "Line 2: what the writer wants instead of the rejected change, as one short casual sentence addressed to the writer as you."
)
UNDO_EXAMPLES = [
    (
        "Draft: quick update for the team slack, we shipped the fix\nBefore: we shipped the fix lol\nAfter: We have successfully deployed the fix.",
        "slack, team, update\nyou keep team Slack updates casual, slang and all",
    ),
    (
        "Draft: Goal: an email to my landlord\nthe heater is broken again, can someone come by this week\nBefore: the heater is broken again, can someone come by this week\nAfter: The heater is broken AGAIN!!! Send someone ASAP.",
        "landlord, email, heater\nyou keep emails to your landlord calm and polite",
    ),
]


@dataclass(frozen=True)
class Fact:
    file: str
    """The file's path under `ROOT` without `.md`, such as "people/lena"."""
    text: str
    """The whole line after "- ", including its date."""
    words: str
    """The line without its date, as the page shows it."""
    day: str
    """The date the line carries, or empty."""


@dataclass(frozen=True)
class Recall:
    fact: Fact
    probability: float
    """Jev's P(the fact is relevant to the draft): the higher of "about what the draft is about" and "followed in this draft"."""


class Draft(BaseModel):
    text: str


class Candidate(BaseModel):
    id: str
    title: str
    line: str
    """The page's line that shares the most words with the question."""


class PageSearch(BaseModel):
    question: str
    open: bool
    """The writer asks to go to the page, rather than about what it says."""
    candidates: list[Candidate]
    """The writer's other pages whose title or text shares a word with the question, found by the page's keyword pass."""


class Undo(BaseModel):
    draft: str
    before: str
    after: str


router = APIRouter()
recalled: list[Recall] = []
"""The facts Jev ranked relevant at the last pause, most relevant first; they feed the coworker's prompts."""
seen: set[str] = set()
"""Sentences already asked the pause questions, so each is decided once."""
writes = asyncio.Lock()
tasks: set[asyncio.Task] = set()


def git(*arguments: str) -> None:
    subprocess.run(["git", "-C", str(ROOT), "-c", "user.name=jevflow", "-c", "user.email=jevflow@localhost", *arguments], check=True, capture_output=True)


def path(file: str) -> Path:
    return ROOT / f"{file}.md"


def read(file: str) -> tuple[list[str], list[str]]:
    """The file's aliases and fact lines; both empty for a missing file."""
    if not path(file).exists():
        return [], []
    lines = path(file).read_text().splitlines()
    aliases = [alias.strip() for alias in lines[0].removeprefix("aliases:").split(",") if alias.strip()] if lines else []
    return aliases, [line[2:] for line in lines if line.startswith("- ")]


def write(file: str, aliases: list[str], lines: list[str], message: str) -> None:
    """Rewrites `file` and commits it with `message`."""
    path(file).parent.mkdir(parents=True, exist_ok=True)
    path(file).write_text(f"aliases: {', '.join(dict.fromkeys(aliases))}\n\n# {file}\n\n" + "".join(f"- {line}\n" for line in lines))
    git("add", "-A")
    git("commit", "-q", "-m", message)


def files() -> list[str]:
    if not (ROOT / ".git").exists():
        ROOT.mkdir(parents=True, exist_ok=True)
        git("init", "-q")
    return sorted(str(file.relative_to(ROOT).with_suffix("")) for file in ROOT.rglob("*.md"))


def facts(file: str) -> list[Fact]:
    return [
        Fact(file, line, re.sub(r"; (stated|corrected) on \d{4}-\d{2}-\d{2}\.$", "", line), next(iter(re.findall(r"\d{4}-\d{2}-\d{2}", line)[-1:]), ""))
        for line in read(file)[1]
    ]


def candidates(text: str) -> list[Fact]:
    """Facts in every file whose aliases appear as whole words in `text`; the profile is not a candidate because it always loads."""
    lowered = text.lower()
    return [
        fact
        for file in files()
        if file != PROFILE and any(re.search(rf"\b{re.escape(alias.lower())}\b", lowered) for alias in read(file)[0])
        for fact in facts(file)
    ]


def context() -> str:
    """The profile and the recalled facts as a prompt preamble, or empty when jevflow knows nothing."""
    lines = [fact.text for fact in facts(PROFILE)] + [recall.fact.text for recall in recalled]
    return "What you know about the writer, who is \"you\" in these lines:\n" + "".join(f"- {line}\n" for line in lines) + "\n" if lines else ""


def spawn(coroutine) -> None:
    task = asyncio.create_task(coroutine)
    tasks.add(task)
    task.add_done_callback(tasks.discard)


def state() -> dict:
    """Every stored fact by file, and the facts recalled at the last pause with Jev's P(relevant)."""
    return {
        "facts": [asdict(fact) for file in files() for fact in facts(file)],
        "recalled": [{"text": recall.fact.text, "probability": recall.probability} for recall in recalled],
    }


async def store(file: str, keywords: list[str], fact: str, gates: dict[str, str], message: str) -> None:
    """Adds `fact` to `file` when every question in `gates` reaches `KEEP_THRESHOLD`, skipping a fact any file already holds and replacing every fact in `file` it contradicts.

    Jev answers the gates, duplicate and contradiction questions in one payload.
    """
    async with writes:
        aliases, lines = read(file)
        stored = [stored_fact.text for stored_file in files() for stored_fact in facts(stored_file)]
        fields = {name: (decide.YesNo, Field(description=question)) for name, question in gates.items()}
        for index, line in enumerate(stored):
            fields[f"same_{index}"] = (decide.YesNo, Field(description=f"Does the stored fact '{line}' already say that {fact}?"))
        for index, line in enumerate(lines):
            fields[f"contradicts_{index}"] = (decide.YesNo, Field(description=f"Does the new fact '{fact}' contradict the stored fact '{line}'?"))
        check = await decide.run(create_model("Store", __doc__="jevflow keeps dated facts about the writer it works with.", **fields), f"New fact: {fact}")
        failed = [name for name in gates if check[name] < KEEP_THRESHOLD]
        duplicates = [f"same_{index}" for index in range(len(stored)) if check[f"same_{index}"] >= KEEP_THRESHOLD]
        if failed or duplicates:
            feed.act(check, "dropped", *(failed or duplicates), component="memory")
            return
        kept = [line for index, line in enumerate(lines) if check[f"contradicts_{index}"] < KEEP_THRESHOLD]
        feed.act(check, "applied", *gates, *(f"contradicts_{index}" for index in range(len(lines)) if check[f"contradicts_{index}"] >= KEEP_THRESHOLD), component="memory")
        corrected = "corrected" if len(kept) < len(lines) else "stated"
        write(file, aliases + keywords, kept + [f"{fact.rstrip('.')}; {corrected} on {date.today()}."], message)


async def remember(sentence: str) -> None:
    """The local model writes the fact line and picks its file; `store` gates it."""
    lines = [line.strip() for line in (await generator.chat(FACT_INSTRUCTIONS, FACT_EXAMPLES, sentence, 60)).splitlines() if line.strip()]
    if len(lines) != 3:
        return
    file = PROFILE if lines[0] == PROFILE else "preferences" if lines[0] == "preferences" else "people/" + re.sub(r"[^a-z0-9-]", "", lines[0].removeprefix("people/").lower())
    if file == "people/":
        return
    fair = {"fair": f"Is '{lines[2]}' a fair statement of what the writer said in '{sentence}'?"}
    keywords = [keyword.strip().lower() for keyword in lines[1].split(",") if keyword.strip()] + [file.removeprefix("people/")] * file.startswith("people/")
    await store(file, keywords, lines[2], fair, f"Remember: {lines[2]}")


async def forget(sentence: str) -> None:
    """Removes every stored fact Jev reads as what `sentence` asks to forget, leaving a dated correction line in its file."""
    stored = [fact for file in files() for fact in facts(file)]
    if not stored:
        return
    fields = {f"match_{index}": (decide.YesNo, Field(description=f"Is '{fact.text}' what the writer asks to forget?")) for index, fact in enumerate(stored)}
    check = await decide.run(create_model("Forget", __doc__=f"The writer typed: '{sentence}'.", **fields), f"Typed: {sentence}")
    matches = [name for name in fields if check[name] >= KEEP_THRESHOLD]
    feed.act(check, "applied" if matches else "dropped", *(matches or fields), component="memory")
    async with writes:
        for file in dict.fromkeys(fact.file for index, fact in enumerate(stored) if check[f"match_{index}"] >= KEEP_THRESHOLD):
            aliases, lines = read(file)
            gone = {fact.text for index, fact in enumerate(stored) if fact.file == file and check[f"match_{index}"] >= KEEP_THRESHOLD}
            write(file, aliases, [line for line in lines if line not in gone] + [f"you asked me to forget this ('{sentence}'); corrected on {date.today()}."], f"Forget: {sentence}")


@router.get("/memory")
async def show() -> dict:
    return state()


@router.post("/pages/search")
async def search_pages(search: PageSearch) -> dict:
    """The candidate Jev ranks highest on "does this page answer the question?" (or "is it the page to open?"), with its P, or an empty reply when none reaches `RELEVANT_THRESHOLD`."""
    if not search.candidates:
        return {}
    asks = "Is the page '{title}' the one the writer asks for in '{question}'?" if search.open else "Does the writer's page '{title}', with the line '{line}', answer '{question}'?"
    fields = {
        f"answers_{index}": (decide.YesNo, Field(description=asks.format(title=candidate.title, line=candidate.line, question=search.question)))
        for index, candidate in enumerate(search.candidates)
    }
    check = await decide.run(create_model("PageSearch", __doc__="A writer asks jevflow about their other pages.", **fields), f"Question: {search.question}")
    best = max(range(len(search.candidates)), key=lambda index: check[f"answers_{index}"])
    probability = check[f"answers_{best}"]
    feed.act(check, "shown" if probability >= RELEVANT_THRESHOLD else "dropped", f"answers_{best}", component="memory")
    return search.candidates[best].model_dump() | {"probability": probability} if probability >= RELEVANT_THRESHOLD else {}


@router.post("/memory/pause")
async def pause(draft: Draft) -> dict:
    """One Jev payload per pause: "a note?", "still true next week?" and "an instruction to forget?" on each new sentence, "about someone or something this draft is about?" and "followed in this draft?" on each alias-matched fact.

    Writes run after the response; the reply lists the sentences Jev read as forget requests, for the page to lift out of the text.
    """
    global recalled
    fresh = list(dict.fromkeys(sentence.text for sentence in decide.line_sentences(draft.text, []) if sentence.text not in seen))
    found = candidates(draft.text)
    fields = {}
    for index, sentence in enumerate(fresh):
        fields[f"note_{index}"] = (decide.YesNo, Field(description=f"Is '{sentence}' a note to the assistant, rather than a line of the text the writer is writing?"))
        fields[f"lasting_{index}"] = (decide.YesNo, Field(description=f"Does '{sentence}' say something about the writer or the people in their life that would still be true next week?"))
        fields[f"forget_{index}"] = (decide.YesNo, Field(description=f"Is '{sentence}' an instruction to forget?"))
    for index, fact in enumerate(found):
        fields[f"relevant_{index}"] = (decide.YesNo, Field(description=f"Is the memory '{fact.text}' about someone or something this draft is about?"))
        fields[f"applies_{index}"] = (decide.YesNo, Field(description=f"Would the writer want the preference or fact '{fact.text}' followed in this draft?"))
    if not fields:
        return state() | {"forgets": []}
    check = await decide.run(create_model("Pause", __doc__=decide.NOTE_CONTEXT, **fields), f"Draft: {draft.text[-MAX_DRAFT_CHARS:]}")
    seen.update(fresh)
    ranked = sorted((Recall(fact, max(check[f"relevant_{index}"], check[f"applies_{index}"])) for index, fact in enumerate(found)), key=lambda recall: recall.probability, reverse=True)
    recalled = [recall for recall in ranked if recall.probability >= RELEVANT_THRESHOLD][:MAX_RECALLED]
    forgets = [sentence for index, sentence in enumerate(fresh) if check[f"forget_{index}"] >= KEEP_THRESHOLD]
    for sentence in forgets:
        spawn(forget(sentence))
    lasting = [index for index, sentence in enumerate(fresh) if min(check[f"note_{index}"], check[f"lasting_{index}"]) >= KEEP_THRESHOLD and sentence not in forgets]
    for index in lasting:
        spawn(remember(fresh[index]))
    shown = [f"{kind}_{index}" for index, fact in enumerate(found) if any(recall.fact == fact for recall in recalled) for kind in ("relevant", "applies")]
    written = [f"forget_{index}" for index, sentence in enumerate(fresh) if sentence in forgets] + [f"{kind}_{index}" for index in lasting for kind in ("note", "lasting")]
    feed.act(check, "applied" if written else "shown" if shown else "silent", *(written or shown), component="memory")
    return state() | {"forgets": forgets}


@router.post("/memory/undo")
async def undo(edit: Undo) -> dict:
    """The local model writes the preference an undone edit shows; Jev keeps it only when the writer rejected this kind of edit, in the same payload as `store`'s questions."""
    lines = [line.strip() for line in (await generator.chat(UNDO_INSTRUCTIONS, UNDO_EXAMPLES, f"Draft: {edit.draft[:MAX_DRAFT_CHARS]}\nBefore: {edit.before}\nAfter: {edit.after}", 60)).splitlines() if line.strip()]
    if len(lines) != 2:
        return state()
    gates = {
        "rejected": f"The writer undid a change from '{edit.before}' to '{edit.after}'. Did the writer reject this kind of edit?",
        "fair": f"The writer undid a change from '{edit.before}' to '{edit.after}'. Does that undo show this about the writer: {lines[1]}?",
    }
    await store("preferences", [keyword.strip().lower() for keyword in lines[0].split(",") if keyword.strip()], lines[1], gates, f"Learn from undo: {lines[1]}")
    return state()
