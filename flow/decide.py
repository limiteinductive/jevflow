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

from flow import feed, generator, reactions
from text_processing import Sentence, split_sentences

MODEL = "typesafe:jev-latest"
MAX_NOTE_WORDS = 12
NOTE_THRESHOLD = 0.6
NOTE_GATE = 0.55
"""A sentence holds a note when the plan question reaches this; hand-made notes score 0.89 to 0.98 and prose sentences at most 0.21."""
OWN_DRAFT_GATE = 0.3
"""A question lifts only when "about their own draft?" reaches this: questions to the recipient ('want me to grab pad thai') score 0.00 to 0.04, and the demo's run-on 'wdyt?' 0.43 to 0.49."""
TIMING_CONTEXT_CHARS = 400
MID_THOUGHT_GATE = 0.5
INTERRUPT_GATE = 0.3
"""The page shows unasked suggestions only below `MID_THOUGHT_GATE` on mid_thought and at or above `INTERRUPT_GATE` on interrupt; the feed marks the gate with the same values."""
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
HURTS_THRESHOLD = 0.7
"""A sentence gets a goal suggestion when Jev's P(it makes the text less of a Goal category) reaches this."""
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
    """One word for the measure, such as "understood"."""
    question: str
    """The question Jev answered."""
    probability: float


@dataclass(frozen=True)
class Answer:
    text: str
    """The coworker's words, written by the local model."""
    measures: list[Measure]
    """Jev's probabilities behind the reply, shown beside the words."""
    scope: "Scope"
    """What the question is about; the page anchors the thread on the last sentence or the whole paragraph."""


@dataclass(frozen=True)
class Timing:
    mid_thought: float
    interrupt: float
    """P(a suggestion about the finished text is welcome now)."""


@dataclass(frozen=True)
class GoalSuggestion:
    replacement: str
    comment: str
    """Why the sentence should change, such as "For a LinkedIn post, this sentence works against 'concrete'."."""
    measures: list[Measure]


@dataclass(frozen=True)
class Suggestion:
    start: int
    end: int
    text: str
    replacement: str
    comment: str
    measures: list[Measure]


agent = Agent(MODEL)
"""One agent for every decision: a fresh agent per call opens a new connection and adds about 0.4 s."""

sentence_notes: dict[tuple[str, str, str], SentenceNote | None] = {}
"""Decisions by sentence text; the page asks about the whole draft on every pause, so each sentence is decided once."""

sentence_fixes: dict[tuple[str, str], tuple[str | None, GoalSuggestion | None]] = {}
"""Jev-accepted correction and goal suggestion by sentence text and Goal, decided once per pair like `sentence_notes`."""


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
    """P(yes) for every yes/no field, and the chosen option for every other field; `feed.act` marks what the page does with them."""
    _, answers = await feed.ask(agent, prompt, output_type)
    return answers


async def decide_note(sentence: str, comment: str, goal: str) -> SentenceNote | None:
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
        **reactions.fields(goal),
    )
    gate, (note, header), question = await asyncio.gather(
        run(NoteGate, f"Sentence: {sentence}"), generator.extract_note(sentence), generator.extract_question(sentence)
    )
    reactions.record(sentence, goal, gate)
    if gate.get("reply", 0) >= NOTE_GATE:
        feed.act(gate, "applied", "reply")
        return SentenceNote(sentence, gate["reply"], "reply", "")
    asks = min(gate["asks"], gate["is_question"])
    field = "question" if asks >= NOTE_GATE else gate["field"]
    probability = max(gate["plan"], asks)
    if probability < NOTE_GATE:
        feed.act(gate, "silent", "plan", "asks")
        return None
    feed.act(gate, "applied", "asks" if field == "question" else "plan", "field")
    if field == "undo":
        return SentenceNote(sentence, probability, "undo", "") if len(sentence.split()) <= MAX_NOTE_WORDS else None
    if field == "question":
        span = await question_span(sentence, question)
        return SentenceNote(span, probability, "question", "") if span else None
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
            feed.act(check, "dropped", "sentence_only_note")
            return None
        note = sentence
    feed.act(check, "applied", "only_note", "whole_note")
    feed.act(header_check, "applied" if min(header_check["fair"], 1 - header_check["adds"]) >= NOTE_THRESHOLD else "dropped", "fair", "adds")
    return SentenceNote(note, probability, field, header if min(header_check["fair"], 1 - header_check["adds"]) >= NOTE_THRESHOLD else note)


async def question_span(sentence: str, question: str) -> str | None:
    """The span to lift as a question to the coworker, or None when Jev says it asks the person the text is for.

    The span is the copied question words when they end `sentence` and Jev says they hold the whole question, else the whole
    sentence; the text before them stays in the draft. "About their own draft" is asked of the span alone, since a run-on
    sentence is mostly the writer's text.
    """
    span = question if question and question != sentence and sentence.endswith(question) else sentence
    QuestionCheck = create_model(
        "QuestionCheck",
        __doc__=NOTE_CONTEXT + f" The writer typed: '{sentence}'.",
        whole_question=(YesNo, Field(description=f"Does '{span}' hold the whole question, leaving none of its words out?")),
        about_own_draft=(YesNo, Field(description=f"In '{span}', is the writer asking about their own draft (how it reads, whether it works), rather than asking the person the text is for?")),
    )
    check = await run(QuestionCheck, f"Typed: {sentence}")
    if check["about_own_draft"] < OWN_DRAFT_GATE:
        feed.act(check, "dropped", "about_own_draft")
        return None
    feed.act(check, "applied", "about_own_draft", "whole_question")
    return span if span == sentence or check["whole_question"] >= NOTE_THRESHOLD else sentence


async def timing(text: str) -> Timing:
    """Whether now is a moment to show the writer anything unasked: every comment, popup and fix waits for it (the page's shortcut until a single per-pause decision motor exists)."""
    Moment = create_model(
        "Moment",
        __doc__=f"A writer is typing a draft; the end of it so far is: '{text[-TIMING_CONTEXT_CHARS:]}'.",
        mid_thought=(YesNo, Field(description="Is the writer in the middle of a thought, likely to keep typing the same idea right now?")),
        interrupt=(YesNo, Field(description="Would a suggestion about what they already wrote be welcome now, rather than breaking their flow?")),
    )
    moment = await run(Moment, f"Draft end: {text[-TIMING_CONTEXT_CHARS:]}")
    feed.act(moment, "applied" if moment["mid_thought"] < MID_THOUGHT_GATE and moment["interrupt"] >= INTERRUPT_GATE else "silent", "mid_thought", "interrupt")
    return Timing(moment["mid_thought"], moment["interrupt"])


async def find_notes(text: str, comment: str, breaks: list[int], goal: str) -> list[Note]:
    """At most one note per sentence, located by character offsets in `text`; `comment` is the coworker's open comment and `goal` the Goal header, each possibly empty."""
    sentences = line_sentences(text, breaks)
    fresh = list(dict.fromkeys(sentence.text.rstrip(".") for sentence in sentences if (sentence.text.rstrip("."), comment, goal) not in sentence_notes))
    for sentence, decision in zip(fresh, await asyncio.gather(*(decide_note(sentence, comment, goal) for sentence in fresh))):
        sentence_notes[sentence, comment, goal] = decision
    notes = []
    for sentence in sentences:
        decision = sentence_notes[sentence.text.rstrip("."), comment, goal]
        if decision is None:
            continue
        start = text.find(decision.text, sentence.start)
        notes.append(Note(start, start + len(decision.text), decision.text, decision.probability, decision.field, decision.header))
    return notes


async def decide_fix(sentence: str, goal: str, categories: list[str]) -> tuple[str | None, GoalSuggestion | None]:
    """The LLM's correction of `sentence` when Jev says it keeps the meaning and the voice, stays small and fixes a real mistake.

    Without an accepted correction, one question per Goal category (the gauges' categories) rides in the same request.
    When Jev says the sentence makes the text less of a category, the LLM rewrites it and `revise` gates the rewrite.
    """
    replacement = await generator.fix(sentence)
    corrects = replacement not in ("", sentence)
    if not corrects and not categories:
        return None, None
    fields = {}
    if corrects:
        fields.update(
            keeps_meaning=(YesNo, Field(description="Does the correction keep every fact and the meaning of the sentence?")),
            same_voice=(YesNo, Field(description="Does the correction still sound like the writer?")),
            small=(YesNo, Field(description="Does the correction only fix mistakes, changing as few words as possible?")),
            real_mistake=(YesNo, Field(description="Does the correction fix a real mistake in the sentence?")),
            on_purpose=(YesNo, Field(description="Does the correction change a spelling or word the writer chose on purpose, for voice or a joke (like 'akshually' or 'gonna')?")),
        )
    fields.update({f"hurts_{index}": (YesNo, Field(description=f"Does the sentence make the text less {category}?")) for index, category in enumerate(categories)})
    context = f"A writer drafting {goal} wrote a sentence." if categories else "A writer wrote a sentence."
    correction = f" A writing assistant proposes a correction. Correction: '{replacement}'." if corrects else ""
    FixCheck = create_model("FixCheck", __doc__=f"{context} Sentence: '{sentence}'.{correction}", **fields)
    check = await run(FixCheck, f"Sentence: {sentence}" + (f"\n\nCorrection: {replacement}" if corrects else ""))
    if corrects and min(check["keeps_meaning"], check["same_voice"], check["small"], check["real_mistake"], 1 - check["on_purpose"]) >= FIX_THRESHOLD:
        feed.act(check, "applied", "keeps_meaning", "same_voice", "small", "real_mistake", "on_purpose")
        return replacement, None
    hurt = [(check[f"hurts_{index}"], category) for index, category in enumerate(categories) if check[f"hurts_{index}"] >= HURTS_THRESHOLD]
    hurts = [f"hurts_{index}" for index in range(len(categories)) if check[f"hurts_{index}"] >= HURTS_THRESHOLD]
    if not hurt:
        feed.act(check, "dropped" if corrects else "silent", *check)
        return None, None
    against = " and ".join(f"'{category}'" for _, category in hurt)
    comment = f"For {goal}, this sentence works against {against}."
    rewrite = await revise(sentence, comment, "ok")
    if rewrite is None or rewrite == sentence:
        feed.act(check, "dropped", *hurts)
        return None, None
    feed.act(check, "shown", *hurts)
    return None, GoalSuggestion(rewrite, comment, [Measure(category, f"1 minus P(yes) for: Does the sentence make the text less {category}?", 1 - probability) for probability, category in hurt])


async def find_fixes(text: str, limit: int, goal: str, categories: list[str]) -> tuple[list[Fix], list[Suggestion]]:
    """Accepted corrections and goal suggestions for the sentences that end at or before `limit`, the start of the sentence being typed."""
    sentences = [
        sentence for sentence in line_sentences(text, []) if sentence.start + len(sentence.text) <= limit and text.startswith(sentence.text, sentence.start)
    ]
    fresh = list(dict.fromkeys(sentence.text for sentence in sentences if (sentence.text, goal) not in sentence_fixes))
    for sentence, decision in zip(fresh, await asyncio.gather(*(decide_fix(sentence, goal, categories) for sentence in fresh))):
        sentence_fixes[sentence, goal] = decision
    fixes, suggestions = [], []
    for sentence in sentences:
        replacement, suggestion = sentence_fixes[sentence.text, goal]
        end = sentence.start + len(sentence.text)
        if replacement is not None:
            fixes.append(Fix(sentence.start, end, sentence.text, replacement))
        if suggestion is not None:
            suggestions.append(Suggestion(sentence.start, end, sentence.text, suggestion.replacement, suggestion.comment, suggestion.measures))
    return fixes, suggestions


async def answer(sentence: str, paragraph: str, question: str, goal: str, memory: str) -> Answer:
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
        drafts = list(dict.fromkeys(await generator.answers(sentence, paragraph, question, goal, NUM_DRAFTS, memory)))
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
    feed.act(check, "shown", *(f"{name}_{drafts.index(ranked[0][2])}" for name in ("answers", "specific", "voice")))
    feed.act(meta, "shown" if meta["kind"] == "understand" else "silent", "kind", "reader_gets")
    measures = [Measure("answers", "Does the reply answer the writer's question?", ranked[0][1])]
    if meta["kind"] == "understand":
        measures.insert(0, Measure("understood", f"Would {reader} get what the sentence means?", meta["reader_gets"]))
    return Answer(ranked[0][2], measures, meta["scope"])


async def revise(sentence: str, comment: str, reply: str, proposed: str, memory: str) -> str | None:
    """`sentence` rewritten as the writer's `reply` to the coworker's `comment` asks, when Jev says the writer wants a change and the rewrite does it, keeps every fact and the voice.

    `proposed` is a rewrite the comment already showed (a goal suggestion); it is checked instead of a fresh one, so the writer gets the text they saw.
    """
    Wants = create_model(
        "Wants",
        __doc__=f"A coworker commented on a writer's sentence. Sentence: '{sentence}'. Comment: '{comment}'. The writer replied: '{reply}'.",
        wants_change=(YesNo, Field(description="Does the writer's reply ask for the sentence to be changed?")),
    )
    rewrite = asyncio.sleep(0, proposed) if proposed else generator.revise(sentence, comment, reply, memory)
    wants, replacement = await asyncio.gather(run(Wants, f"Reply: {reply}"), rewrite)
    if wants["wants_change"] < NOTE_GATE:
        return None
    ReviseCheck = create_model(
        "ReviseCheck",
        __doc__=f"A coworker commented on a writer's sentence, the writer replied, and the coworker rewrote it. Sentence: '{sentence}'. Comment: '{comment}'. Reply: '{reply}'. Rewrite: '{replacement}'.",
        follows=(YesNo, Field(description="Does the rewrite do what the writer's reply asks, following the comment where the reply agrees with it?")),
        keeps_facts=(YesNo, Field(description="Does the rewrite keep every fact the sentence states? Adding what the comment asks for is fine.")),
        same_voice=(YesNo, Field(description="Does the rewrite still sound like the writer?")),
        only_text=(YesNo, Field(description="Is the rewrite only text the writer would send, with no advice, commentary or formatting marks in it?")),
        keeps_meaning=(YesNo, Field(description="Does the rewrite keep the writer's meaning?")),
        keeps_style=(YesNo, Field(description="Does the rewrite keep the writer's casing and style?")),
    )
    check = await run(ReviseCheck, f"Sentence: {sentence}\n\nRewrite: {replacement}")
    feed.act(check, "applied" if replacement and min(check.values()) >= FIX_THRESHOLD else "dropped", min(check, key=check.get))
    return replacement if replacement and min(check.values()) >= FIX_THRESHOLD else None
