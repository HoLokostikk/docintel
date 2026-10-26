from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    thought_tokens: int
    model_version: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.thought_tokens


REQUEST_TIMEOUT_MS = 120_000


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    load_dotenv()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set; put it in .env at the repo root")
    return genai.Client(
        api_key=key,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )


def usage_of(response: types.GenerateContentResponse) -> Usage:
    meta = response.usage_metadata
    return Usage(
        input_tokens=(meta.prompt_token_count or 0) if meta else 0,
        output_tokens=(meta.candidates_token_count or 0) if meta else 0,
        thought_tokens=(meta.thoughts_token_count or 0) if meta else 0,
        model_version=response.model_version or "unknown",
    )


MODEL = "gemini-3.6-flash"

TEMPERATURE = 0.0

RETRYABLE_CODES = frozenset({429, 500, 502, 503, 504})

MAX_ATTEMPTS = 5
BASE_DELAY = 1.0
MAX_DELAY = 30.0


def generate(
    contents: object,
    *,
    config: types.GenerateContentConfig,
    model: str = MODEL,
    max_attempts: int = MAX_ATTEMPTS,
) -> types.GenerateContentResponse:
    last: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return get_client().models.generate_content(
                model=model, contents=contents, config=config
            )
        except errors.APIError as exc:
            if exc.code not in RETRYABLE_CODES:
                raise
            last = exc
            if attempt == max_attempts - 1:
                break
            delay = min(BASE_DELAY * 2**attempt, MAX_DELAY) * (0.5 + random.random())
            print(f"  {exc.code} {exc.status}; retrying in {delay:.1f}s")
            time.sleep(delay)

    raise RuntimeError(f"{model} unavailable after {max_attempts} attempts") from last


def extraction_config(schema: type) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        temperature=TEMPERATURE,
        response_mime_type="application/json",
        response_schema=schema,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )


LLM_CACHE_DIR = Path("data/cache/llm")

T = TypeVar("T", bound=BaseModel)


def _fingerprint(contents: object) -> str:
    digest = hashlib.sha256()

    def feed(item: object) -> None:
        if isinstance(item, str):
            digest.update(b"s")
            digest.update(item.encode("utf-8"))
        elif isinstance(item, bytes):
            digest.update(b"b")
            digest.update(item)
        elif isinstance(item, types.Part):
            digest.update(b"p")
            if item.text:
                digest.update(item.text.encode("utf-8"))
            if item.inline_data is not None and item.inline_data.data:
                digest.update(item.inline_data.data)
                digest.update((item.inline_data.mime_type or "").encode("utf-8"))
        elif isinstance(item, (list, tuple)):
            for sub in item:
                feed(sub)
        else:
            digest.update(repr(item).encode("utf-8"))

    feed(contents)
    return digest.hexdigest()


def generate_structured(
    contents: object,
    schema: type[T],
    *,
    model: str = MODEL,
    refresh: bool = False,
) -> tuple[T, Usage, bool]:
    schema_text = json.dumps(schema.model_json_schema(), sort_keys=True)
    key = hashlib.sha256(
        f"{model}|{TEMPERATURE}|{schema_text}|{_fingerprint(contents)}".encode()
    ).hexdigest()

    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    entry = LLM_CACHE_DIR / f"{model}__{key}.json"

    if entry.exists() and not refresh:
        stored = json.loads(entry.read_text(encoding="utf-8"))
        return schema.model_validate(stored["value"]), Usage(**stored["usage"]), True

    response = generate(contents, config=extraction_config(schema), model=model)
    parsed = response.parsed
    if not isinstance(parsed, schema):
        raise RuntimeError(f"model did not return {schema.__name__}: {response.text!r}")

    usage = usage_of(response)
    entry.write_text(
        json.dumps(
            {"value": parsed.model_dump(mode="json"), "usage": usage.__dict__},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return parsed, usage, False
