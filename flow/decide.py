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

from flow import claims, components, feed, generator, reactions
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
ON_PURPOSE_CHECK = "Does '{draft}' call a word or spelling the writer chose on purpose, for voice or a joke (like 'akshually' or 'gonna'), a mistake, or offer to fix it?"
"""Asked of each draft beside `DRAFT_CHECKS`; a draft at or above `NOTE_GATE` ranks below every other, so the reply never offers a fix the rewrite gate's on_purpose check would refuse."""
ANSWER_FLOOR = 0.3
"""Below this on "answers the writer's question?" for every draft, the drafts are resampled once."""
FIX_THRESHOLD = 0.7
"""Every fix question must reach this; a real spelling fix scores at least 0.90 on all four and an unchanged sentence 0.00 on `real_mistake`."""
HURTS_THRESHOLD = 0.7
"""A sentence gets a goal suggestion when Jev's P(it makes the text less of a Goal category) reaches this."""
OPTION_INSTRUCTIONS = "A writer asks an open question about a part of their draft. Reply with 2 to 4 short possible answers, one per line, each 1 to 4 plain words. Reply with the list only."
OPTION_EXAMPLES = [("Text: the launch slipped again, as expected\nQuestion: how does this come across?", "neutral\nfrustrated\npassive-aggressive")]
MAX_OPTIONS = 4
OPTION_FIT = 0.5
"""The options are shown only when Jev's P(one of them answers the question) reaches this; otherwise the question gets a one-line reply."""
NOTE_CONTEXT = "A writer types their text and, in the same stream, notes to a writing assistant such as 'make this punchier', 'this is for engineers', 'im writing a blog post', 'replying to my boss about friday' or 'undo'."

YesNo = Literal["yes", "no"]

NoteField = Literal["new_page", "goal", "audience", "tone", "to_do", "undo", "question", "reply", "find", "open"]
"""Header fields, plus "undo" (reverts the last edit), "question" (opens a comment thread on the previous sentence), "reply" (answers the open comment), and "find" and "open" (search the writer's other pages, and switch to the page found)."""

Scope = Literal["sentence", "paragraph"]

QuestionKind = Literal["understand", "opinion", "cut_or_keep", "true", "wording", "other"]

ComponentIntent = Literal["yes_no", "question", "instruction", "turn_off", "turn_on", "change", "mention"]

HeaderField = Literal["goal", "audience", "tone", "to_do", "undo"]

@dataclass(frozen=True)
class Header:
    field: HeaderField
    text: str
    """The header value, such as "an X post" for "ok im gonna write an x post"."""


@dataclass(frozen=True)
class Note:
    start: int
    end: int
    text: str
    probability: float
    field: NoteField
    headers: list[Header]
    """The headers the note files, such as Goal and Audience for "writing a blog post for engineers"; a `new_page` note's headers seed the new page."""


@dataclass(frozen=True)
class SentenceNote:
    text: str
    probability: float
    field: NoteField
    headers: list[Header]


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
class Rewrite:
    text: str
    keeps_meaning: float
    uses_pattern: float
    """P(the rewrite still reads as the flagged pattern)."""


@dataclass(frozen=True)
class Answer:
    text: str
    """The coworker's words, written by the local model."""
    measures: list[Measure]
    """Jev's probabilities behind the reply, shown beside the words."""
    scope: "Scope"
    """What the question is about; the page anchors the thread on the last sentence or the whole paragraph."""


@dataclass(frozen=True)
class ComponentReply:
    intent: "ComponentIntent"
    answer: Answer | None
    """The coworker's reply when the writer asked an open question about the component."""
    probability: float | None
    """Jev's P(yes) on the writer's own question when it is a yes/no question."""
    options: list[str] | None
    """The local model's short answers to an open question about a copied span, when Jev says one of them fits."""
    pick: str | None
    """The option Jev chose."""


@dataclass(frozen=True)
class Timing:
    mid_thought: float
    interrupt: float
    """P(a suggestion about the finished text is welcome now)."""


@dataclass(frozen=True)
class GoalSuggestion:
    replacement: str
    comment: str
    """Why the sentence should change, such as "heads up: for a LinkedIn post, this one works against 'concrete'."."""
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

sentence_notes: dict[tuple[str, str, str, frozenset[str]], SentenceNote | None] = {}
"""Decisions by sentence text; the page asks about the whole draft on every pause, so each sentence is decided once."""

meta_sentences: dict[str, float] = {}
"""P(the sentence is addressed to the assistant rather than part of the text), by sentence; filled by the note gate."""

sentence_fixes: dict[tuple[str, str], tuple[str | None, GoalSuggestion | None]] = {}
"""Jev-accepted correction and goal suggestion by sentence text and Goal, decided once per pair like `sentence_notes`."""


def line_sentences(text: str, breaks: list[int]) -> list[Sentence]:
    """Sentences with offsets in `text`, never spanning a line break or an offset in `breaks`: the page ends a sentence at a newline and at a comment thread's anchor."""
    positions = [0, *sorted(position for position in breaks if 0 < position < len(text)), len(text)]
    return [
        replace(sentence, start=chunk_start + sentence.start, end=chunk_start + sentence.end)
        for chunk_start, chunk_end in pairwise(positions)
        for sentence in split_sentences(text[chunk_start:chunk_end])
    ]


async def run(output_type: type[BaseModel], prompt: str) -> dict[str, float | str]:
    """P(yes) for every yes/no field, and the chosen option for every other field; `feed.act` marks what the page does with them."""
    _, answers = await feed.ask(agent, prompt, output_type)
    return answers


async def decide_note(sentence: str, comment: str, goal: str, enabled: frozenset[str]) -> SentenceNote | None:
    """Jev gates the sentence while the LLM copies out the note words and writes the header (the local model is free, so it runs on every sentence); Jev then checks both.

    When the copied words fail the check, the sentence is a note only if Jev reads all of it as one; a header that fails is replaced by the note words.
    With a `goal`, a sentence Jev reads as starting something other than it is a `new_page` note: the whole sentence leaves, and its header is the new page's Goal.
    A sentence that asks to turn a component on or off is recorded by `components` and is no note; reaction questions ride only when reactions are `enabled`.
    A goal or `new_page` note that Jev reads as also naming who the text is for files the audience the local model copied as a second header.
    """
    reply = {"reply": (YesNo, Field(description="Is the writer answering the coworker's comment (agreeing, disagreeing, correcting it or asking for the change) rather than writing the text?"))} if comment else {}
    new_piece = {"new_piece": (YesNo, Field(description=f"Does the writer say they are now writing something other than the {goal}?"))} if goal and "pages" in enabled else {}
    NoteGate = create_model(
        "NoteGate",
        __doc__=NOTE_CONTEXT + " The writer keeps several pages, one per piece of writing." + (f" The writer's coworker just commented on the draft: '{comment}'" if comment else ""),
        plan=(YesNo, Field(description="Is the sentence the writer's plan or intent for the text, rather than part of the text itself?")),
        asks=(YesNo, Field(description="Is the writer asking the assistant for its opinion or help?")),
        is_question=(YesNo, Field(description="Is the sentence a question?")),
        field=(HeaderField, Field(description="What is the note about? goal: what the writer is writing; audience: who it is for; tone: how it should sound; to_do: something to add, check or change; undo: asks to undo the last edit.")),
        for_whom=(YesNo, Field(description="Does the sentence also say who the text is for?")),
        for_assistant=(YesNo, Field(description="Is the sentence addressed to the writing assistant (a comment, reply, question or instruction to it), rather than part of the text the writer is writing?")),
        find_page=(YesNo, Field(description="Does the writer ask about something they wrote before, rather than about this text?")),
        open_page=(YesNo, Field(description="Is the sentence a command to switch to another page, like 'open the tacos one' or 'take me to my essay'?")),
        **reply,
        **new_piece,
        **components.fields(),
        **(reactions.fields(goal) if "reactions" in enabled else {}),
        **(claims.fields(goal) if "claims" in enabled else {}),
    )
    gate, (note, header), question, audience = await asyncio.gather(
        run(NoteGate, f"Sentence: {sentence}"), generator.extract_note(sentence), generator.extract_question(sentence), generator.extract_audience(sentence)
    )
    if "reactions" in enabled:
        reactions.record(sentence, goal, gate)
    meta_sentences[sentence] = gate["for_assistant"]
    if "claims" in enabled:
        claims.record(sentence, goal, gate)
    if components.record(sentence, gate):
        return None
    if gate.get("reply", 0) >= NOTE_GATE:
        feed.act(gate, "applied", "reply")
        return SentenceNote(sentence, gate["reply"], "reply", [])
    searches = max(gate["find_page"], gate["open_page"])
    if searches >= NOTE_GATE:
        feed.act(gate, "applied", "find_page", "open_page")
        return SentenceNote(sentence, searches, "open" if gate["open_page"] >= gate["find_page"] else "find", [])
    asks = min(gate["asks"], gate["is_question"])
    new_page = gate.get("new_piece", 0)
    field = "question" if asks >= NOTE_GATE else "new_page" if new_page >= NOTE_GATE else gate["field"]
    probability = max(gate["plan"], asks, new_page)
    if probability < NOTE_GATE:
        feed.act(gate, "silent", "plan", "asks")
        return None
    feed.act(gate, "applied", {"question": "asks", "new_page": "new_piece"}.get(field, "plan"), "field")
    if field == "undo":
        return SentenceNote(sentence, probability, "undo", []) if len(sentence.split()) <= MAX_NOTE_WORDS else None
    if field == "question":
        span = await question_span(sentence, question)
        return SentenceNote(span, probability, "question", []) if span else None
    NoteCheck = create_model(
        "NoteCheck",
        __doc__=NOTE_CONTEXT + f" The sentence is: '{sentence}'.",
        only_note=(YesNo, Field(description=f"Is '{note}' only about the draft (what it is, who it is for, how it should sound or what to change), with none of the text the writer is writing?")),
        whole_note=(YesNo, Field(description=f"Does '{note}' hold the whole note, leaving no note words out?")),
        sentence_only_note=(YesNo, Field(description="Is the whole sentence only about the draft (what it is, who it is for, how it should sound or what to change), with none of the text the writer is writing?")),
    )
    names_audience = field in ("goal", "new_page") and gate["for_whom"] >= NOTE_GATE and audience != "" and audience in sentence
    HeaderCheck = create_model(
        "HeaderCheck",
        __doc__=f"A writer's note to a writing assistant is filed as a short header at the top of the draft. The note: '{note}'. The header: '{header}'.",
        fair=(YesNo, Field(description="Is the header a fair short version of the note?")),
        adds=(YesNo, Field(description="Does the header add something the note does not say?")),
        **({"audience": (YesNo, Field(description=f"Does the note say the text is for '{audience}'?"))} if names_audience else {}),
    )
    decisive = ("fair", "adds", "audience") if names_audience else ("fair", "adds")

    def headers(field: HeaderField, fallback: str) -> list[Header]:
        fair = min(header_check["fair"], 1 - header_check["adds"]) >= NOTE_THRESHOLD
        feed.act(header_check, "applied" if fair else "dropped", *decisive)
        return [Header(field, header if fair else fallback)] + ([Header("audience", audience)] if names_audience and header_check["audience"] >= NOTE_THRESHOLD else [])

    if field == "new_page":
        header_check = await run(HeaderCheck, f"Note: {note}\n\nHeader: {header}")
        seeded = headers("goal", sentence)
        print(f"new_page: {sentence!r} P={probability:.2f} goal={goal!r} seeds={[header.text for header in seeded]}", flush=True)
        if any(header.field == "goal" and header.text.strip().lower() == goal.strip().lower() for header in seeded):
            return None
        return SentenceNote(sentence, probability, field, seeded)
    check, header_check = await asyncio.gather(run(NoteCheck, f"Sentence: {sentence}"), run(HeaderCheck, f"Note: {note}\n\nHeader: {header}"))
    if note not in sentence or min(check["only_note"], check["whole_note"]) < NOTE_THRESHOLD:
        if check["sentence_only_note"] < NOTE_THRESHOLD:
            feed.act(check, "dropped", "sentence_only_note")
            return None
        note = sentence
    feed.act(check, "applied", "only_note", "whole_note")
    return SentenceNote(note, probability, field, headers(field, note))


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
        about_assistant=(YesNo, Field(description=f"In '{span}', is the writer asking the writing assistant about itself (what it can do, what it is doing, why it did something)?")),
    )
    check = await run(QuestionCheck, f"Typed: {sentence}")
    if max(check["about_own_draft"], check["about_assistant"]) < OWN_DRAFT_GATE:
        feed.act(check, "dropped", "about_own_draft", "about_assistant")
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


def meta_spans(sentences: list[Sentence]) -> list[Sentence]:
    """The sentences Jev reads as addressed to the assistant, which every judge of the text skips."""
    return [sentence for sentence in sentences if meta_sentences.get(sentence.text.rstrip("."), 0) >= NOTE_GATE]


async def find_notes(text: str, comment: str, breaks: list[int], goal: str, enabled: frozenset[str]) -> list[Note]:
    """At most one note per sentence, located by character offsets in `text`; `comment` is the coworker's open comment and `goal` the Goal header, each possibly empty."""
    sentences = line_sentences(text, breaks)
    fresh = list(dict.fromkeys(sentence.text.rstrip(".") for sentence in sentences if (sentence.text.rstrip("."), comment, goal, enabled) not in sentence_notes))
    for sentence, decision in zip(fresh, await asyncio.gather(*(decide_note(sentence, comment, goal, enabled) for sentence in fresh))):
        sentence_notes[sentence, comment, goal, enabled] = decision
    notes = []
    for sentence in sentences:
        decision = sentence_notes[sentence.text.rstrip("."), comment, goal, enabled]
        if decision is None:
            continue
        start = text.find(decision.text, sentence.start)
        notes.append(Note(start, start + len(decision.text), decision.text, decision.probability, decision.field, decision.headers))
    return notes


async def decide_fix(sentence: str, draft: str, goal: str, categories: list[str]) -> tuple[str | None, GoalSuggestion | None]:
    """The LLM's correction of `sentence` when Jev says it keeps the meaning and the voice, stays small, fixes a real mistake, and does not capitalize a `draft` kept lowercase on purpose.

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
            lowercase=(YesNo, Field(description="Is the draft written in lowercase on purpose, like casual texting, with sentences starting lowercase?")),
            capitalizes=(YesNo, Field(description="Does the correction capitalize a letter the sentence had in lowercase?")),
        )
    fields.update({f"hurts_{index}": (YesNo, Field(description=f"Does the sentence make the text less {category}?")) for index, category in enumerate(categories)})
    context = f"A writer drafting {goal} wrote a sentence." if categories else "A writer wrote a sentence."
    correction = f" Draft so far: '{draft}'. A writing assistant proposes a correction. Correction: '{replacement}'." if corrects else ""
    FixCheck = create_model("FixCheck", __doc__=f"{context} Sentence: '{sentence}'.{correction}", **fields)
    check = await run(FixCheck, f"Sentence: {sentence}" + (f"\n\nCorrection: {replacement}" if corrects else ""))
    keeps_register = corrects and min(check["lowercase"], check["capitalizes"]) < NOTE_GATE
    if keeps_register and min(check["keeps_meaning"], check["same_voice"], check["small"], check["real_mistake"], 1 - check["on_purpose"]) >= FIX_THRESHOLD:
        feed.act(check, "applied", "keeps_meaning", "same_voice", "small", "real_mistake", "on_purpose", "lowercase", "capitalizes")
        return replacement, None
    hurt = [(check[f"hurts_{index}"], category) for index, category in enumerate(categories) if check[f"hurts_{index}"] >= HURTS_THRESHOLD]
    hurts = [f"hurts_{index}" for index in range(len(categories)) if check[f"hurts_{index}"] >= HURTS_THRESHOLD]
    if not hurt:
        feed.act(check, "dropped" if corrects else "silent", *check)
        return None, None
    against = " and ".join(f"'{category}'" for _, category in hurt)
    comment = f"heads up: for {goal}, this one works against {against}."
    rewrite = await revise(sentence, comment, "ok", "", "")
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
    ends = {sentence.text: sentence.start + len(sentence.text) for sentence in sentences}
    decisions = await asyncio.gather(*(decide_fix(sentence, text[max(0, ends[sentence] - TIMING_CONTEXT_CHARS):ends[sentence]], goal, categories) for sentence in fresh))
    for sentence, decision in zip(fresh, decisions):
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


async def answer(sentence: str, paragraph: str, question: str, goal: str, memory: str, features: str) -> Answer:
    """The coworker's reply to the writer's `question` about `sentence`: the local model drafts `NUM_DRAFTS` in one batch, Jev ranks them on `DRAFT_CHECKS`.

    Drafts that offer to fix an on-purpose word rank last, then drafts that fail `ANSWER_FLOOR` on "answers the question"; when every draft fails it the drafts are resampled once, and the text is empty when they fail again.
    Jev also names the kind of question; when it asks whether readers will understand, P(a reader gets it) leads the measures.
    When Jev says the question is about jevflow itself, the drafts are redone with `features`, the components' on/off state.
    """
    reader = f"a typical reader of this {goal}" if goal else "a typical reader"
    Meta = create_model(
        "Meta",
        __doc__=f"A writer drafting a {goal or 'text'} asks a coworker a question about the sentence they just wrote.",
        scope=(Scope, Field(description="Is the question about the last sentence, or about the whole paragraph (its length, pace or structure)?")),
        kind=(QuestionKind, Field(description="What is the writer asking? understand: will readers get it; opinion: what do you think; cut_or_keep: should it stay; true: is it accurate; wording: is there a better way to say it; other.")),
        reader_gets=(YesNo, Field(description=f"Would {reader} get what the sentence means?")),
        about_assistant=(YesNo, Field(description="Is the question about the writing assistant itself, such as what it can do, rather than a request about the draft?")),
    )
    meta_task = asyncio.create_task(run(Meta, f"Paragraph: {paragraph}\n\nLast sentence: {sentence}\n\nQuestion: {question}"))
    shown_features, retries = "", 1
    while True:
        drafts = list(dict.fromkeys(await generator.answers(sentence, paragraph, question, goal, NUM_DRAFTS, memory, shown_features)))
        fields = {
            f"{name}_{index}": (YesNo, Field(description=template.format(draft=draft)))
            for index, draft in enumerate(drafts)
            for name, template in DRAFT_CHECKS.items()
        } | {f"on_purpose_{index}": (YesNo, Field(description=ON_PURPOSE_CHECK.format(draft=draft))) for index, draft in enumerate(drafts)}
        Pick = create_model("Pick", __doc__=f"A writer drafting a {goal or 'text'} asks a coworker about the sentence they just wrote; the coworker drafted replies.", **fields)
        check = await run(Pick, f"Paragraph: {paragraph}\n\nLast sentence: {sentence}\n\nQuestion: {question}")
        on_purpose = {draft: check[f"on_purpose_{index}"] for index, draft in enumerate(drafts)}
        ranked = sorted(
            ((math.prod(check[f"{name}_{index}"] for name in DRAFT_CHECKS), check[f"answers_{index}"], draft) for index, draft in enumerate(drafts)),
            key=lambda row: (on_purpose[row[2]] < NOTE_GATE, row[1] >= ANSWER_FLOOR, row[0]),
            reverse=True,
        )
        meta = await meta_task
        if features and not shown_features and meta["about_assistant"] >= NOTE_GATE:
            shown_features = features
        elif max(answers for _, answers, _ in ranked) >= ANSWER_FLOOR:
            break
        elif retries:
            retries -= 1
        else:
            feed.act(check, "dropped", *(f"answers_{index}" for index in range(len(drafts))))
            return Answer("", [], meta["scope"])
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


async def rewrite(sentence: str, pattern: str) -> Rewrite | None:
    """The local model's best of `NUM_DRAFTS` rewrites of `sentence` without `pattern`, all scored in one Jev payload; None when none keeps the meaning and drops the pattern."""
    drafts = list(dict.fromkeys(draft for draft in await generator.revisions(sentence, f"It reads as {pattern}. Say what it is as a plain statement, with no contrast against what it is not.", NUM_DRAFTS) if draft and draft != sentence))
    if not drafts:
        return None
    fields = {}
    for index, draft in enumerate(drafts):
        fields[f"keeps_meaning_{index}"] = (YesNo, Field(description=f"Does '{draft}' keep the meaning of the sentence?"))
        fields[f"uses_pattern_{index}"] = (YesNo, Field(description=f"Does '{draft}' still read as {pattern}?"))
    check = await run(create_model("RewriteCheck", __doc__=f"A writer's sentence reads as {pattern}. Sentence: '{sentence}'. A coworker drafted rewrites.", **fields), f"Sentence: {sentence}")
    passing = [
        (check[f"keeps_meaning_{index}"] * (1 - check[f"uses_pattern_{index}"]), index)
        for index in range(len(drafts))
        if check[f"keeps_meaning_{index}"] >= FIX_THRESHOLD and 1 - check[f"uses_pattern_{index}"] >= FIX_THRESHOLD
    ]
    if not passing:
        feed.act(check, "dropped", *check)
        return None
    _, best = max(passing)
    feed.act(check, "applied", f"keeps_meaning_{best}", f"uses_pattern_{best}")
    return Rewrite(drafts[best], check[f"keeps_meaning_{best}"], check[f"uses_pattern_{best}"])


async def about_component(span: str, question: str, component: str, text: str, sentence: str, paragraph: str, goal: str, memory: str) -> ComponentReply:
    """What the writer meant by typing `span` with a pasted reference to a page component, and the coworker's answer when it is a question.

    `question` is `span` without the reference; `text` is the copied draft text when the reference is a span, else empty. Only a span gets the yes/no probe; a yes/no question
    about any other component gets the coworker's answer. The answer and the probe run while Jev decides the intent, so neither costs an extra round.
    """
    Intent = create_model(
        "Intent",
        __doc__=f"A writer pasted a reference to a part of their writing assistant's page into their draft. That part: {component}. They typed: '{span}'.",
        intent=(
            ComponentIntent,
            Field(
                description="yes_no: they ask a yes or no question about that part; question: they ask an open question about it; instruction: they ask to drop, forget or mark done an item; "
                "turn_off: they ask to remove, hide or turn off the whole part; turn_on: they ask to bring it back or turn it on; change: they ask to change what it measures or how it works; "
                "mention: it is part of the text they are writing."
            ),
        ),
    )
    intent, reply, probed, picked = await asyncio.gather(
        run(Intent, f"Typed: {span}"),
        answer(sentence or paragraph, paragraph, f"{span} ({component})", goal, memory, ""),
        probe(question, text) if text else asyncio.sleep(0, None),
        pick_option(question, text) if text else asyncio.sleep(0, None),
    )
    probing = intent["intent"] == "yes_no" and probed is not None
    feed.act(intent, "applied", "intent")
    if probed is not None:
        feed.act(probed, "shown" if probing else "silent", "answer")
    choosing = intent["intent"] == "question" and picked is not None
    asks = intent["intent"] in ("question", "yes_no") and not probing and not choosing
    options, pick = picked if choosing else (None, None)
    return ComponentReply(intent["intent"], reply if asks else None, probed["answer"] if probing else None, options, pick)


async def pick_option(question: str, text: str) -> tuple[list[str], str] | None:
    """The local model's short answers to the writer's open `question` about `text` and the one Jev picks, or None when Jev says none of them answers it."""
    reply = await generator.chat(OPTION_INSTRUCTIONS, OPTION_EXAMPLES, f"Text: {text}\nQuestion: {question}", 40)
    options = list(dict.fromkeys(filter(None, (line.strip(" -*\u2022.").lower() for line in reply.splitlines()))))[:MAX_OPTIONS]
    if len(options) < 2:
        return None
    OptionPick = create_model(
        "OptionPick",
        __doc__=f"A writer asks an open question about a part of their draft: {text}",
        pick=(Literal[tuple(options)], Field(description=question)),
        fits=(YesNo, Field(description=f"Does one of these answer '{question}': {'; '.join(options)}?")),
    )
    check = await run(OptionPick, f"Text: {text}\n\nQuestion: {question}")
    fits = check["fits"] >= OPTION_FIT
    feed.act(check, "shown" if fits else "dropped", "pick", "fits")
    return (options, check["pick"]) if fits else None


async def probe(question: str, text: str) -> dict[str, float | str]:
    """Jev's answers with P(yes) on the writer's own yes/no `question` about `text` under "answer"."""
    Probe = create_model("Probe", __doc__=f"A writer asks a yes or no question about a part of their draft: {text}", answer=(YesNo, Field(description=question)))
    return await run(Probe, f"Text: {text}\n\nQuestion: {question}")
