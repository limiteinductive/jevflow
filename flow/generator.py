"""Text for the flow from a local MLX server (OpenAI-compatible, `mlx_lm.server`).

Jev makes every decision; these calls only write text.
"""

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
    " The writer asks you about what they just wrote: the last sentence, or the whole paragraph if the question is about it. "
    "Say what a reader of this kind of text might miss, and give one concrete fix. Keep their joke. "
    "If they ask about you (what you can do, what you are doing), answer from your features and whether each is on."
)
ANSWER_EXAMPLES = [
    (
        "Goal: a tweet\nParagraph: turns out my cat was right about the vacuum all along\nLast sentence: turns out my cat was right about the vacuum all along\nQuestion: will people get this?",
        "they won't know what she thought: \"she hid from it for years.\" add it?",
    ),
    (
        "Goal: a blog post\nParagraph: In this post we will talk about caching.\nLast sentence: In this post we will talk about caching.\nQuestion: how do i make this less boring?",
        "open on the problem: \"we cached the wrong thing for a year.\" swap it?",
    ),
    (
        "Goal: a text to my wife\nParagraph: sorry i was short with you this morning, the deploy was a mess\nLast sentence: sorry i was short with you this morning, the deploy was a mess\nQuestion: does this sound ok?",
        "warm, owns it. end on her: \"you didn't deserve that.\" add it?",
    ),
    (
        "Goal: an x post\nParagraph: our smol team shipped it anyway\nLast sentence: our smol team shipped it anyway\nQuestion: does this land?",
        "lands. \"smol\" sells the joke, leaving it as is.",
    ),
]
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


async def answers(sentence: str, paragraph: str, question: str, goal: str, count: int, context: str) -> list[str]:
    """`count` sampled replies from one batched completion; the prompt is Qwen3's chat template with thinking off, written out because /completions takes raw text.

    mlx_lm.server ignores `n` and returns one reply. `context` is `flow.memory.context()` and `flow.components.context()`.
    """
    turns = messages(ANSWER_INSTRUCTIONS, ANSWER_EXAMPLES, f"{context}Goal: {goal}\nParagraph: {paragraph}\nLast sentence: {sentence}\nQuestion: {question}")
    prompt = "".join(f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n" for turn in turns) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    response = await mlx.post("/completions", json={"model": MODEL, "prompt": prompt, "n": count, "max_tokens": 60, "temperature": 0.9})
    response.raise_for_status()
    return [undash(choice["text"].strip()) for choice in response.json()["choices"]]


async def revise(sentence: str, comment: str, reply: str, memory: str) -> str:
    return undash(await chat(REVISE_INSTRUCTIONS, REVISE_EXAMPLES, f"{memory}Sentence: {sentence}\nComment: {comment}\nWriter's reply: {reply}", 2 * len(sentence.split()) + 40))


async def extract_question(sentence: str) -> str:
    """The question words at the end of `sentence`, copied as the local model reads them."""
    return (await chat(QUESTION_INSTRUCTIONS, QUESTION_EXAMPLES, sentence, 40)).strip().strip("'\"")
