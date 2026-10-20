from pathlib import Path
from typing import Protocol

from docintel.ingest.models import Document


class Parser(Protocol):
    name: str

    def parse(self, path: Path) -> Document: ...
