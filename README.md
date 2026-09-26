# jevflow

jevflow is a writing coworker that sits in your document while you write. It reads each sentence as you type it, and it knows what you're writing and who it's for.

You never switch modes. You type the text, your notes and your questions in one stream, the way you'd think out loud next to a colleague. jevflow sorts them. Notes go to the top of the page, questions get an answer in the margin, and the text stays yours.

Every decision it makes is a question to Jev, a judge model that answers in half a second. Is this a note or part of the text? Does this line land? Will readers get the joke? Does this fix keep your meaning? That speed is what lets it keep up with you.

Open jevflow and start typing.

## Run it

Needs an Apple Silicon Mac (the text model runs locally on MLX) and your own TypeSafe API key.

1. Install: `uv sync`
2. Set your key in the shell, never in a file in the repo: `export TYPESAFE_API_KEY=<your key>`
3. Start the local model on port 8085: `uv run autocomplete/mlx_server.py --model Qwen/Qwen3-4B-MLX-4bit --port 8085`. The same server with `--model Qwen/Qwen3-1.7B-MLX-8bit --port 8083` returns `n` continuations from one batched call.
4. Start the app: `uv run python -m flow.app`
5. Open http://127.0.0.1:8000 and type.

Tests run offline, with Jev and the local model faked: `uv run --with pytest pytest`.

## What it does

- A sentence about the draft (what it is, who it is for, how it should sound) lifts into the Goal, Audience, Tone or To do header.
- A question to the coworker ("wdyt?") moves into a comment in the margin, and Jev picks the answer. Typing a reply ("ok add that") applies the suggested rewrite if Jev accepts it.
- Finished sentences get small corrections that Jev accepts only if they keep the meaning and the voice.
- Sentences that read as AI-written get an amber highlight and a question.
- Cmd+Z, or typing "undo", reverts the last edit, and a reverted edit is not reapplied.
