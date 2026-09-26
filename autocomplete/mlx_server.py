# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx-lm", "fastapi", "uvicorn[standard]"]
# [tool.uv]
# exclude-newer = "2026-09-22T00:00:00Z"
# ///
"""OpenAI-compatible completions server for one local MLX model; `n` continuations come from one batched call.

POST /v1/completions takes `prompt`, `max_tokens` (default 16), `n` (default 8) and `temperature`
(default 1.0), and returns `n` choices whose text keeps its leading whitespace. POST /v1/chat/completions
returns `n` replies (default 1) with thinking off. Requests are served one at a time.
Usage: `uv run autocomplete/mlx_server.py --model Qwen/Qwen3-1.7B-MLX-8bit --port 8083` (Apple silicon only).
"""

import argparse
import threading
import time
from typing import Literal

import uvicorn
from fastapi import FastAPI
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler
from pydantic import BaseModel, Field

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--port", type=int, required=True)
arguments = parser.parse_args()

model, tokenizer = load(arguments.model)
generation_lock = threading.Lock()
app = FastAPI()


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(16, ge=1, le=512)
    n: int = Field(8, ge=1, le=32)
    temperature: float = Field(1.0, ge=0.0, le=2.0)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(512, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    n: int = Field(1, ge=1, le=32)


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: Literal["stop", "length"]


class CompletionResponse(BaseModel):
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatResponse(BaseModel):
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]


class ModelCard(BaseModel):
    id: str
    object: str = "model"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


def generate(tokens: list[int], n: int, max_tokens: int, temperature: float) -> list[tuple[str, Literal["stop", "length"]]]:
    """`n` sampled continuations of `tokens` from one batched call, each with its finish reason."""
    with generation_lock:
        generator = BatchGenerator(model, stop_tokens=[[token] for token in tokenizer.eos_token_ids], sampler=make_sampler(temp=temperature))
        uids = generator.insert([tokens] * n, [max_tokens] * n)
        generated = {uid: [] for uid in uids}
        finish_reasons = {}
        while responses := generator.next_generated():
            for response in responses:
                if response.finish_reason != "stop":
                    generated[response.uid].append(response.token)
                if response.finish_reason is not None:
                    finish_reasons[response.uid] = response.finish_reason
        generator.close()
    return [(tokenizer.decode(generated[uid]), finish_reasons[uid]) for uid in uids]


@app.post("/v1/completions")
def completions(request: CompletionRequest) -> CompletionResponse:
    continuations = generate(tokenizer.encode(request.prompt), request.n, request.max_tokens, request.temperature)
    return CompletionResponse(
        created=int(time.time()),
        model=arguments.model,
        choices=[CompletionChoice(index=index, text=text, finish_reason=finish_reason) for index, (text, finish_reason) in enumerate(continuations)],
    )


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest) -> ChatResponse:
    tokens = tokenizer.apply_chat_template([message.model_dump() for message in request.messages], add_generation_prompt=True, enable_thinking=False)
    replies = generate(tokens, request.n, request.max_tokens, request.temperature)
    return ChatResponse(
        created=int(time.time()),
        model=arguments.model,
        choices=[
            ChatChoice(index=index, message=ChatMessage(role="assistant", content=text), finish_reason=finish_reason)
            for index, (text, finish_reason) in enumerate(replies)
        ],
    )


@app.get("/v1/models")
def models() -> ModelList:
    return ModelList(data=[ModelCard(id=arguments.model)])


uvicorn.run(app, host="127.0.0.1", port=arguments.port)
