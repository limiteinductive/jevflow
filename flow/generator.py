"""Text for the flow from a local MLX server (OpenAI-compatible, `mlx_lm.server`).

Jev makes every decision; these calls only write text.
"""

import httpx

MLX_URL = "http://127.0.0.1:8085/v1"
MODEL = "Qwen/Qwen3-4B-MLX-8bit"
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
FIX_INSTRUCTIONS = (
    "Fix spelling, grammar and punctuation mistakes in the user's sentence. Change nothing else: keep the writer's words, slang and tone. "
    "If nothing is wrong, reply with the sentence unchanged. Reply with the sentence only."
)
FIX_EXAMPLES = [
    ("we shiped it friday and nobody noticed", "we shipped it Friday and nobody noticed"),
    ("Honestly the new build is way faster lol", "Honestly the new build is way faster lol"),
    ("their going to love this feature, its so fast", "they're going to love this feature, it's so fast"),
]
ANSWER_INSTRUCTIONS = (
    "You are a sharp coworker reading a writer's draft over their shoulder. The writer asks you about what they just wrote: the last sentence, or the whole paragraph if the question is about it. "
    "Answer like a coworker in one or two short sentences: say what a reader of this kind of text might miss in that sentence, and give one concrete fix. "
    "Keep their joke."
)
ANSWER_EXAMPLES = [
    (
        "Goal: a tweet\nParagraph: turns out my cat was right about the vacuum all along\nLast sentence: turns out my cat was right about the vacuum all along\nQuestion: will people get this?",
        "People won't know what the cat's stance was. Add that she hid under the bed every time it ran.",
    )
]
REVISE_INSTRUCTIONS = (
    "Rewrite the writer's sentence to do what the coworker's comment suggests. "
    "Keep every fact, the writer's words where possible, their slang and their joke. "
    "Reply with the rewritten text only, as the writer would send it: no advice, no quotes, no markdown."
)
REVISE_EXAMPLES = [
    (
        "Sentence: turns out my cat was right about the vacuum all along\nComment: Readers don't know what your cat thought: say it hid from the vacuum for years, then land the joke.",
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


async def chat(instructions: str, examples: list[tuple[str, str]], text: str, max_tokens: int, temperature: float = 0, seed: int = 0) -> str:
    """Completion with Qwen3's thinking mode off, which otherwise spends the token budget and returns empty text."""
    response = await mlx.post(
        "/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": instructions},
                *({"role": role, "content": content} for example in examples for role, content in zip(("user", "assistant"), example)),
                {"role": "user", "content": text},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    response.raise_for_status()
    return (response.json()["choices"][0]["message"]["content"] or "").strip()


async def extract_note(sentence: str) -> tuple[str, str]:
    """The note words copied from `sentence`, and the note as a header value; both empty when the reply is not two lines."""
    lines = [line.strip().strip("'\"") for line in (await chat(NOTE_INSTRUCTIONS, NOTE_EXAMPLES, sentence, 40)).splitlines() if line.strip()]
    return (lines[0], lines[1]) if len(lines) == 2 else ("", "")


async def fix(sentence: str) -> str:
    return await chat(FIX_INSTRUCTIONS, FIX_EXAMPLES, sentence, 2 * len(sentence.split()) + 16)


async def answer(sentence: str, paragraph: str, question: str, goal: str, seed: int) -> str:
    """One sampled draft; the server ignores `n`, so each draft is its own request with its own seed."""
    return await chat(ANSWER_INSTRUCTIONS, ANSWER_EXAMPLES, f"Goal: {goal}\nParagraph: {paragraph}\nLast sentence: {sentence}\nQuestion: {question}", 60, 0.9, seed)


async def revise(sentence: str, comment: str) -> str:
    return await chat(REVISE_INSTRUCTIONS, REVISE_EXAMPLES, f"Sentence: {sentence}\nComment: {comment}", 2 * len(sentence.split()) + 40)


async def extract_question(sentence: str) -> str:
    """The question words at the end of `sentence`, copied as the local model reads them."""
    return (await chat(QUESTION_INSTRUCTIONS, QUESTION_EXAMPLES, sentence, 40)).strip().strip("'\"")
