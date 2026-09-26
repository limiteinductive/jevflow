"""Text for the flow from a local MLX server (OpenAI-compatible, `mlx_lm.server`).

Jev makes every decision; these calls only write text.
"""

import asyncio
import re

import httpx

MLX_URL = "http://127.0.0.1:8085/v1"
MODEL = "Qwen/Qwen3-4B-MLX-4bit"
NOTE_INSTRUCTIONS = (
    "The user message is one sentence a writer typed. It holds a note to a writing assistant, and it may also hold the writer's own text. "
    "Reply with two lines. Line 1: the note words, copied exactly. Line 2: the note as a short header value."
)
NOTE_EXAMPLES = [
    ("im writing a blog post about our launch", "im writing a blog post about our launch\na blog post about our launch"),
    ("We shipped on Friday make it sound calmer", "make it sound calmer\ncalmer"),
    ("add the price here The plan costs less than lunch", "add the price here\nadd the price"),
]
"""Worked turns sent before the sentence; with only the instructions, the 1.7B model copies the instruction's example into every header."""
AUDIENCE_INSTRUCTIONS = (
    "The user message is one sentence a writer typed about what they are writing. "
    "Reply with who the text is for, copied from the sentence. If the sentence does not say who it is for, reply with none."
)
AUDIENCE_EXAMPLES = [
    ("ok writing a blog post for engineers about our CI", "engineers"),
    ("replying to my boss about friday", "my boss"),
    ("im writing a blog post about our launch", "none"),
]
FIX_INSTRUCTIONS = (
    "Fix spelling, grammar and punctuation mistakes in the user's sentence. Change nothing else: keep the writer's words, slang and tone. "
    "If nothing is wrong, reply with the sentence unchanged. Reply with the sentence only."
)
FIX_EXAMPLES = [
    ("we shiped it friday and nobody noticed", "we shipped it Friday and nobody noticed"),
    ("Honestly the new build is way faster lol", "Honestly the new build is way faster lol"),
    ("their going to love this feature, its so fast", "they're going to love this feature, it's so fast"),
]
VOICE = (
    "You are a writing coworker who texts the writer like a sharp friend. "
    "Write one short lowercase line: the verdict and the one fix. "
    "Your first words carry the answer: no preamble, no praise of the question, no restating it. "
    "Give the exact words to use, then at most a two word question that offers to make the change. "
    "Say plainly what you are not sure of and what you leave alone. A note the writer did not ask for starts with 'heads up:'. "
    "Match the writer's register: loose on a casual post, calm and warm on a personal note. "
    "Use at most one emoji, usually none, and no dashes. "
    "Talk to the writer as 'you', never 'the writer' or 'the user'."
)
"""The coworker's voice; every prompt whose output the writer reads as the coworker's words starts with it."""
ANSWER_INSTRUCTIONS = VOICE + (
    " The writer asks you for something about what they just wrote: feedback, a rewrite, an explanation or ideas. "
    "Do what they ask about the last sentence, or the whole paragraph if the request is about it. "
    "For feedback, say what a reader of this kind of text might miss and give one concrete fix. Keep their joke."
)
ANSWER_EXAMPLES = [
    (
        "Goal: a tweet\nParagraph: turns out my cat was right about the vacuum all along\nLast sentence: turns out my cat was right about the vacuum all along\nQuestion: will people get this?",
        "1. they won't know what she thought: \"she hid from it for years.\" add it?\n2. most will. \"right all along\" does the work, leave it.\n3. lead with the vacuum: \"the vacuum lost. my cat called it.\" swap it?",
    ),
    (
        "Goal: a blog post\nParagraph: In this post we will talk about caching.\nLast sentence: In this post we will talk about caching.\nQuestion: how do i make this less boring?",
        "1. open on the problem: \"we cached the wrong thing for a year.\" swap it?\n2. give a number: \"our p99 was 4 seconds.\" add it?\n3. cut it. start with the second sentence.",
    ),
    (
        "Goal: a text to my wife\nParagraph: sorry i was short with you this morning, the deploy was a mess\nLast sentence: sorry i was short with you this morning, the deploy was a mess\nQuestion: does this sound ok?",
        "1. warm, owns it. end on her: \"you didn't deserve that.\" add it?\n2. it's fine as is. sending it matters more than the words.\n3. drop the deploy: \"sorry i was short with you this morning.\" swap it?",
    ),
    (
        "Goal: an x post\nParagraph: our smol team shipped it anyway\nLast sentence: our smol team shipped it anyway\nQuestion: does this land?",
        "1. lands. \"smol\" sells the joke, leaving it as is.\n2. lands. add what \"it\" is: \"shipped the app anyway.\" add it?\n3. lands. end on a beat: \"shipped it anyway. on a friday.\" add it?",
    ),
]
FEATURES_EXAMPLE = (
    (
        "Your features that are on: notes: files what you say about the draft as headers; replies: answers your questions in the margin; gauges: scores the draft for your Goal\nYour features that are off: corrections, reactions\n"
        "Goal: a cover letter\nParagraph: I led the migration to Postgres.\nLast sentence: I led the migration to Postgres.\nQuestion: what can you do?"
    ),
    "1. i file your notes up top, answer you here and score it for your goal. corrections and reactions are off, just ask to turn them on.\n2. notes, replies and goal scores are on. want corrections or reactions too?\n3. i sort your notes, answer questions like this one and score the draft for your goal.",
)
"""Sent after `ANSWER_EXAMPLES` only with the components' on/off state; without them, a memory preamble makes the 4B copy this reply word for word."""
REVISE_INSTRUCTIONS = (
    "Rewrite the writer's sentence to do what the writer's reply asks, following the coworker's comment where the reply agrees with it. "
    "Keep every fact, the writer's words where possible, their slang and their joke. "
    "Reply with the rewritten text only, as the writer would send it: no advice, no quotes, no markdown."
)
REVISE_EXAMPLES = [
    (
        "Sentence: turns out my cat was right about the vacuum all along\nComment: Readers don't know what your cat thought: say it hid from the vacuum for years, then land the joke.\nWriter's reply: ok add that",
        "my cat hid from the vacuum for years. turns out she was right all along",
    )
]
QUESTION_INSTRUCTIONS = (
    "The user message is text a writer typed. It ends with a question the writer asks their coworker about the text, and it may start with the writer's own text. "
    "Reply with the question words only, copied exactly, leaving out the writer's own text."
)
QUESTION_EXAMPLES = [
    ("We shipped on friday lol does this sound too smug?", "does this sound too smug?"),
    ("my cat hates the vacuum ok wait is that even funny", "ok wait is that even funny"),
    ("should I cut this?", "should I cut this?"),
]

CLAIM_INSTRUCTIONS = (
    "The user's sentence states facts about the world, and some are wrong. Rewrite it with every wrong detail corrected (who, where, when, how many), changing no other words. "
    "Keep the writer's words, slang, casing and tone. Reply with the sentence only."
)
CLAIM_EXAMPLES = [
    ("the eiffel tower is in rome lol", "the eiffel tower is in paris lol"),
    ("btw python was made by linus torvalds in 2005", "btw python was made by guido van rossum in 1991"),
]

mlx = httpx.AsyncClient(base_url=MLX_URL, timeout=30)


def undash(text: str) -> str:
    """`text` with each em dash turned into a comma; the 4B writes them despite the voice rule."""
    return text.replace(" — ", ", ").replace("—", ", ")


def messages(instructions: str, examples: list[tuple[str, str]], text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": instructions},
        *({"role": role, "content": content} for example in examples for role, content in zip(("user", "assistant"), example)),
        {"role": "user", "content": text},
    ]


async def chat(instructions: str, examples: list[tuple[str, str]], text: str, max_tokens: int) -> str:
    """Completion with Qwen3's thinking mode off, which otherwise spends the token budget and returns empty text."""
    response = await mlx.post(
        "/chat/completions",
        json={
            "model": MODEL,
            "messages": messages(instructions, examples, text),
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    response.raise_for_status()
    return (response.json()["choices"][0]["message"]["content"] or "").strip()


async def extract_note(sentence: str) -> tuple[str, str]:
    """The note words copied from `sentence`, and the note as a header value; both empty when the reply is not two lines."""
    lines = [line.strip().strip("'\"") for line in (await chat(NOTE_INSTRUCTIONS, NOTE_EXAMPLES, sentence, 40)).splitlines() if line.strip()]
    return (lines[0], lines[1]) if len(lines) == 2 else ("", "")


async def extract_audience(sentence: str) -> str:
    """Who `sentence` says the text is for, copied from it; empty when the local model reads no audience."""
    audience = (await chat(AUDIENCE_INSTRUCTIONS, AUDIENCE_EXAMPLES, sentence, 20)).strip().strip("'\"")
    return "" if audience.lower() == "none" else audience


async def fix(sentence: str) -> str:
    return await chat(FIX_INSTRUCTIONS, FIX_EXAMPLES, sentence, 2 * len(sentence.split()) + 16)


async def sample(instructions: str, examples: list[tuple[str, str]], text: str, count: int, max_tokens: int, temperature: float) -> list[str]:
    """`count` sampled replies from one batched completion; the prompt is Qwen3's chat template with thinking off, written out because /completions takes raw text.

    mlx_lm.server ignores `n` and returns one reply.
    """
    prompt = "".join(f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n" for turn in messages(instructions, examples, text)) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    response = await mlx.post("/completions", json={"model": MODEL, "prompt": prompt, "n": count, "max_tokens": max_tokens, "temperature": temperature})
    response.raise_for_status()
    return [undash(choice["text"].strip()) for choice in response.json()["choices"]]


NUMBERED = re.compile(r"^\s*\d+[.)]\s*")


async def candidates(instructions: str, examples: list[tuple[str, str]], text: str, count: int, max_tokens: int) -> list[str]:
    """Up to `count` distinct replies from one completion, which lists them numbered one per line; `examples` replies use that numbered form.

    One list, instead of `count` samples, makes the replies differ by construction: the 4B's samples at temperature 0.9 are often the same reply.
    """
    reply = await chat(instructions + f" Reply with {count} candidates that each take a different approach, numbered 1 to {count}, one per line.", examples, text, count * max_tokens)
    return list(dict.fromkeys(undash(stripped) for line in reply.splitlines() if (stripped := NUMBERED.sub("", line).strip())))[:count]


async def answers(sentence: str, paragraph: str, question: str, goal: str, count: int, memory: str, features: str) -> list[str]:
    """`count` distinct replies; `memory` is `flow.memory.context()`, `features` is `flow.components.context()` or empty."""
    examples = [*ANSWER_EXAMPLES, FEATURES_EXAMPLE] if features else ANSWER_EXAMPLES
    return await candidates(ANSWER_INSTRUCTIONS, examples, f"{memory}{features}Goal: {goal}\nParagraph: {paragraph}\nLast sentence: {sentence}\nQuestion: {question}", count, 60)


async def revise(sentence: str, comment: str, reply: str, memory: str) -> str:
    return undash(await chat(REVISE_INSTRUCTIONS, REVISE_EXAMPLES, f"{memory}Sentence: {sentence}\nComment: {comment}\nWriter's reply: {reply}", 2 * len(sentence.split()) + 40))


async def revisions(sentence: str, comment: str, count: int) -> list[str]:
    """`count` sampled rewrites of `sentence` as `comment` asks, one request each since the server returns one reply per request."""
    batches = await asyncio.gather(*(sample(REVISE_INSTRUCTIONS, REVISE_EXAMPLES, f"Sentence: {sentence}\nComment: {comment}\nWriter's reply: rewrite it", 1, 2 * len(sentence.split()) + 40, 0.9) for _ in range(count)))
    return [rewrite for batch in batches for rewrite in batch]


async def extract_question(sentence: str) -> str:
    """The question words at the end of `sentence`, copied as the local model reads them."""
    return (await chat(QUESTION_INSTRUCTIONS, QUESTION_EXAMPLES, sentence, 40)).strip().strip("'\"")


async def correct_claims(sentence: str, num_drafts: int) -> list[str]:
    """`num_drafts` sampled rewrites of `sentence` with its wrong facts corrected."""
    return await sample(CLAIM_INSTRUCTIONS, CLAIM_EXAMPLES, sentence, num_drafts, 2 * len(sentence.split()) + 16, 0.8)
