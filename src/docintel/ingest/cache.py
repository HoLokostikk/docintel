from __future__ import annotations

import hashlib
from pathlib import Path

from docintel.ingest.base import Parser
from docintel.ingest.models import Document

CACHE_DIR = Path("data/cache")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_cached(parser: Parser, path: Path, *, refresh: bool = False) -> Document:

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{parser.name.replace('/', '_')}__{file_hash(path)}.json"
    entry = CACHE_DIR / key

    if entry.exists() and not refresh:
        return Document.model_validate_json(entry.read_text(encoding="utf-8"))

    document = parser.parse(path)
    entry.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    return document
