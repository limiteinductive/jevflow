"""Jev decisions for the writing flow: which sentences hold notes, what they are about, and whether the note the LLM lifted out is right.

Every decision is one Jev request with all its questions in one payload.
"""

import asyncio
import math
from itertools import pairwise
from dataclasses import dataclass, replace
from typing import Literal

from pydantic import BaseModel, Field, create_model
from pydantic_ai import Agent

from flow import generator
from text_processing import Sentence, split_sentences

MODEL = "typesafe:jev-latest"
MAX_NOTE_WORDS = 12
NOTE_THRESHOLD = 0.6
NOTE_GATE = 0.55
"""A sentence holds a note when the plan question reaches this; hand-made notes score 0.89 to 0.98 and prose sentences at most 0.21."""
NUM_DRAFTS = 3
DRAFT_CHECKS = {
    "answers": "Does '{draft}' answer the writer's question?",
    "specific": "Is '{draft}' specific to this draft, not generic advice?",
    "voice": "Does '{draft}' keep the writer's voice and joke?",
    "reads": "Does the reply '{draft}' read the writer's sentence the way the writer meant it?",
    "leads": "Does the reply '{draft}' lead with its answer?",
    "quick": "Can the reply '{draft}' be read in two seconds?",
    "one_question": "Does the reply '{draft}' end with at most one question?",
    "casual": "Does the reply '{draft}' sound like a casual text from a friend?",
}
"""Jev's questions on each drafted reply, all drafts in one payload; a draft ranks by the product of its answers."""
ANSWER_FLOOR = 0.3
"""Below this on "answers the writer's question?" for every draft, the drafts are resampled once."""
FIX_THRESHOLD = 0.7
"""Every fix question must reach this; a real spelling fix scores at least 0.90 on all four and an unchanged sentence 0.00 on `real_mistake`."""
NOTE_CONTEXT = "A writer types their text and, in the same stream, notes to a writing assistant such as 'make this punchier', 'this is for engineers', 'im writing a blog post', 'replying to my boss about friday' or 'undo'."

YesNo = Literal["yes", "no"]

NoteField = Literal["goal", "audience", "tone", "to_do", "undo", "question", "reply"]
"""Header fields, plus "undo" (reverts the last edit), "question" (opens a comment thread on the previous sentence) and "reply" (answers the open comment)."""

Scope = Literal["sentence", "paragraph"]

QuestionKind = Literal["understand", "opinion", "cut_or_keep", "true", "wording", "other"]

HeaderField = Literal["goal", "audience", "tone", "to_do", "undo"]


@dataclass(frozen=True)
class Note:
    start: int
    end: int
    text: str
    probability: float
    field: NoteField
    header: str
    """The note as a header value, such as "an X post" for "ok im gonna write an x post"."""


@dataclass(frozen=True)
class SentenceNote:
    text: str
    probability: float
    field: NoteField
    header: str


@dataclass(frozen=True)
class Fix:
    start: int
    end: int
    text: str
    replacement: str


@dataclass(frozen=True)
class Measure:
    label: str
    """The question Jev answered, as a short phrase such as "readers get it"."""
    probability: float


@dataclass(frozen=True)
class Answer:
    text: str
    """The coworker's words, written by the local model."""
    measures: list[Measure]
    """Jev's probabilities behind the reply, shown beside the words."""
    scope: "Scope"
    """What the question is about; the page anchors the thread on the last sentence or the whole paragraph."""


agent = Agent(MODEL)
"""One agent for every decision: a fresh agent per call opens a new connection and adds about 0.4 s."""

sentence_notes: dict[tuple[str, str], SentenceNote | None] = {}
"""Decisions by sentence text; the page asks about the whole draft on every pause, so each sentence is decided once."""

decisions = 0
"""Jev questions answered since the server started, one per yes/no or choice field."""

sentence_fixes: dict[str, str | None] = {}
"""Jev-accepted replacements by sentence text, decided once per sentence like `sentence_notes`."""


def line_sentences(text: str, breaks: list[int]) -> list[Sentence]:
    """Sentences with offsets in `text`, never spanning a line break or an offset in `breaks`: the page ends a sentence at a newline and at a comment thread's anchor."""
    sentences = []
    positions = [0, *sorted(position for position in breaks if 0 < position < len(text)), len(text)]
    for chunk_start, chunk_end in pairwise(positions):
        line_start = chunk_start
        for line in text[chunk_start:chunk_end].split("\n"):
            sentences += [replace(sentence, start=line_start + sentence.start, end=line_start + sentence.end) for sentence in split_sentences(line)]
            line_start += len(line) + 1
    return sentences


async def run(output_type: type[BaseModel], prompt: str) -> dict[str, float | str]:
    """P(yes) for every yes/no field, and the chosen option for every other field."""
    global decisions
    result = await agent.run(prompt, output_type=output_type)
    probabilities = result.response.provider_details["probabilities"]
    decisions += len(probabilities)
    return {name: probabilities[name]["yes"] if "yes" in probabilities[name] else getattr(result.output, name) for name in probabilities}


async def decide_note(sentence: str, comment: str) -> SentenceNote | None:
    """Jev gates the sentence while the LLM copies out the note words and writes the header (the local model is free, so it runs on every sentence); Jev then checks both.

    When the copied words fail the check, the sentence is a note only if Jev reads all of it as one; a header that fails is replaced by the note words.
    """
    reply = {"reply": (YesNo, Field(description="Is the writer answering the coworker's comment (agreeing, disagreeing, correcting it or asking for the change) rather than writing the text?"))} if comment else {}
    NoteGate = create_model(
        "NoteGate",
        __doc__=NOTE_CONTEXT + (f" The writer's coworker just commented on the draft: '{comment}'" if comment else ""),
        plan=(YesNo, Field(description="Is the sentence the writer's plan or intent for the text, rather than part of the text itself?")),
        asks=(YesNo, Field(description="Is the writer asking the assistant for its opinion or help?")),
        is_question=(YesNo, Field(description="Is the sentence a question?")),
        field=(HeaderField, Field(description="What is the note about? goal: what the writer is writing; audience: who it is for; tone: how it should sound; to_do: something to add, check or change; undo: asks to undo the last edit.")),
        **reply,
    )
    gate, (note, header), question = await asyncio.gather(
        run(NoteGate, f"Sentence: {sentence}"), generator.extract_note(sentence), generator.extract_question(sentence)
    )
    if gate.get("reply", 0) >= NOTE_GATE:
        return SentenceNote(sentence, gate["reply"], "reply", "")
    asks = min(gate["asks"], gate["is_question"])
    field = "question" if asks >= NOTE_GATE else gate["field"]
    probability = max(gate["plan"], asks)
    if probability < NOTE_GATE:
        return None
    if field == "undo":
        return SentenceNote(sentence, probability, "undo", "") if len(sentence.split()) <= MAX_NOTE_WORDS else None
    if field == "question":
        return SentenceNote(await question_span(sentence, question), probability, "question", "")
    NoteCheck = create_model(
        "NoteCheck",
        __doc__=NOTE_CONTEXT + f" The sentence is: '{sentence}'.",
        only_note=(YesNo, Field(description=f"Is '{note}' only about the draft (what it is, who it is for, how it should sound or what to change), with none of the text the writer is writing?")),
        whole_note=(YesNo, Field(description=f"Does '{note}' hold the whole note, leaving no note words out?")),
        sentence_only_note=(YesNo, Field(description="Is the whole sentence only about the draft (what it is, who it is for, how it should sound or what to change), with none of the text the writer is writing?")),
    )
    HeaderCheck = create_model(
        "HeaderCheck",
        __doc__=f"A writer's note to a writing assistant is filed as a short header at the top of the draft. The note: '{note}'. The header: '{header}'.",
        fair=(YesNo, Field(description="Is the header a fair short version of the note?")),
        adds=(YesNo, Field(description="Does the header add something the note does not say?")),
    )
    check, header_check = await asyncio.gather(run(NoteCheck, f"Sentence: {sentence}"), run(HeaderCheck, f"Note: {note}\n\nHeader: {header}"))
    if note not in sentence or min(check["only_note"], check["whole_note"]) < NOTE_THRESHOLD:
        if check["sentence_only_note"] < NOTE_THRESHOLD:
            return None
        note = sentence
    return SentenceNote(note, probability, field, header if min(header_check["fair"], 1 - header_check["adds"]) >= NOTE_THRESHOLD else note)


async def question_span(sentence: str, question: str) -> str:
    """The question words when they end `sentence` and Jev says they hold the whole question, else the whole sentence; the text before them stays in the draft."""
    if not question or question == sentence or not sentence.endswith(question):
        return sentence
    QuestionCheck = create_model(
        "QuestionCheck",
        __doc__=f"A writer typed: '{sentence}'.",
        whole_question=(YesNo, Field(description=f"Does '{question}' hold the whole question, leaving none of its words out?")),
    )
    check = await run(QuestionCheck, f"Typed: {sentence}")
    return question if check["whole_question"] >= NOTE_THRESHOLD else sentence


async def find_notes(text: str, comment: str, breaks: list[int]) -> list[Note]:
    """At most one note per sentence, located by character offsets in `text`; `comment` is the coworker's open comment, or empty."""
    sentences = line_sentences(text, breaks)
    fresh = list(dict.fromkeys(sentence.text.rstrip(".") for sentence in sentences if (sentence.text.rstrip("."), comment) not in sentence_notes))
    for sentence, decision in zip(fresh, await asyncio.gather(*(decide_note(sentence, comment) for sentence in fresh))):
        sentence_notes[sentence, comment] = decision
    notes = []
    for sentence in sentences:
        decision = sentence_notes[sentence.text.rstrip("."), comment]
        if decision is None:
            continue
        start = text.find(decision.text, sentence.start)
        notes.append(Note(start, start + len(decision.text), decision.text, decision.probability, decision.field, decision.header))
    return notes


async def decide_fix(sentence: str) -> str | None:
    """The LLM's correction of `sentence` when Jev says it keeps the meaning and the voice, stays small and fixes a real mistake."""
    replacement = await generator.fix(sentence)
    if replacement in ("", sentence):
        return None
    FixCheck = create_model(
        "FixCheck",
        __doc__=f"A writing assistant proposes a correction to a writer's sentence. Sentence: '{sentence}'. Correction: '{replacement}'.",
        keeps_meaning=(YesNo, Field(description="Does the correction keep every fact and the meaning of the sentence?")),
        same_voice=(YesNo, Field(description="Does the correction still sound like the writer?")),
        small=(YesNo, Field(description="Does the correction only fix mistakes, changing as few words as possible?")),
        real_mistake=(YesNo, Field(description="Does the correction fix a real mistake in the sentence?")),
        on_purpose=(YesNo, Field(description="Does the correction change a spelling or word the writer chose on purpose, for voice or a joke (like 'akshually' or 'gonna')?")),
    )
    check = await run(FixCheck, f"Sentence: {sentence}\n\nCorrection: {replacement}")
    return replacement if min(check["keeps_meaning"], check["same_voice"], check["small"], check["real_mistake"], 1 - check["on_purpose"]) >= FIX_THRESHOLD else None


async def find_fixes(text: str, limit: int) -> list[Fix]:
    """Accepted corrections for the sentences that end at or before `limit`, the start of the sentence being typed."""
    sentences = [
        sentence for sentence in line_sentences(text, []) if sentence.start + len(sentence.text) <= limit and text.startswith(sentence.text, sentence.start)
    ]
    fresh = list(dict.fromkeys(sentence.text for sentence in sentences if sentence.text not in sentence_fixes))
    for sentence, replacement in zip(fresh, await asyncio.gather(*(decide_fix(sentence) for sentence in fresh))):
        sentence_fixes[sentence] = replacement
    return [
        Fix(sentence.start, sentence.start + len(sentence.text), sentence.text, sentence_fixes[sentence.text])
        for sentence in sentences
        if sentence_fixes[sentence.text] is not None
    ]


async def answer(sentence: str, paragraph: str, question: str, goal: str) -> Answer:
    """The coworker's reply to the writer's `question` about `sentence`: the local model drafts `NUM_DRAFTS` in one batch, Jev ranks them on `DRAFT_CHECKS`.

    The top-ranked draft is shown unless every draft fails "answers the question"; then the drafts are resampled once.
    Jev also names the kind of question; when it asks whether readers will understand, P(a reader gets it) leads the measures.
    """
    reader = f"a typical reader of this {goal}" if goal else "a typical reader"
    Meta = create_model(
        "Meta",
        __doc__=f"A writer drafting a {goal or 'text'} asks a coworker a question about the sentence they just wrote.",
        scope=(Scope, Field(description="Is the question about the last sentence, or about the whole paragraph (its length, pace or structure)?")),
        kind=(QuestionKind, Field(description="What is the writer asking? understand: will readers get it; opinion: what do you think; cut_or_keep: should it stay; true: is it accurate; wording: is there a better way to say it; other.")),
        reader_gets=(YesNo, Field(description=f"Would {reader} get what the sentence means?")),
    )
    meta_task = asyncio.create_task(run(Meta, f"Paragraph: {paragraph}\n\nLast sentence: {sentence}\n\nQuestion: {question}"))
    for attempt in range(2):
        drafts = list(dict.fromkeys(await generator.answers(sentence, paragraph, question, goal, NUM_DRAFTS)))
        fields = {
            f"{name}_{index}": (YesNo, Field(description=template.format(draft=draft)))
            for index, draft in enumerate(drafts)
            for name, template in DRAFT_CHECKS.items()
        }
        Pick = create_model("Pick", __doc__=f"A writer drafting a {goal or 'text'} asks a coworker about the sentence they just wrote; the coworker drafted replies.", **fields)
        check = await run(Pick, f"Paragraph: {paragraph}\n\nLast sentence: {sentence}\n\nQuestion: {question}")
        ranked = sorted(((math.prod(check[f"{name}_{index}"] for name in DRAFT_CHECKS), check[f"answers_{index}"], draft) for index, draft in enumerate(drafts)), reverse=True)
        if max(answers for _, answers, _ in ranked) >= ANSWER_FLOOR:
            break
    meta = await meta_task
    measures = [Measure("answers your question", ranked[0][1])]
    if meta["kind"] == "understand":
        measures.insert(0, Measure("readers get it", meta["reader_gets"]))
    return Answer(ranked[0][2], measures, meta["scope"])


async def revise(sentence: str, comment: str, reply: str) -> str | None:
    """`sentence` rewritten as the writer's `reply` to the coworker's `comment` asks, when Jev says the writer wants a change and the rewrite does it, keeps every fact and the voice."""
    Wants = create_model(
        "Wants",
        __doc__=f"A coworker commented on a writer's sentence. Sentence: '{sentence}'. Comment: '{comment}'. The writer replied: '{reply}'.",
        wants_change=(YesNo, Field(description="Does the writer's reply ask for the sentence to be changed?")),
    )
    wants, replacement = await asyncio.gather(run(Wants, f"Reply: {reply}"), generator.revise(sentence, comment, reply))
    if wants["wants_change"] < NOTE_GATE:
        return None
    ReviseCheck = create_model(
        "ReviseCheck",
        __doc__=f"A coworker commented on a writer's sentence, the writer replied, and the coworker rewrote it. Sentence: '{sentence}'. Comment: '{comment}'. Reply: '{reply}'. Rewrite: '{replacement}'.",
        follows=(YesNo, Field(description="Does the rewrite do what the writer's reply asks, following the comment where the reply agrees with it?")),
        keeps_facts=(YesNo, Field(description="Does the rewrite keep every fact the sentence states? Adding what the comment asks for is fine.")),
        same_voice=(YesNo, Field(description="Does the rewrite still sound like the writer?")),
        only_text=(YesNo, Field(description="Is the rewrite only text the writer would send, with no advice, commentary or formatting marks in it?")),
    )
    check = await run(ReviseCheck, f"Sentence: {sentence}\n\nRewrite: {replacement}")
    return replacement if replacement and min(check.values()) >= FIX_THRESHOLD else None
