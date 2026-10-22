from __future__ import annotations

import argparse
from pathlib import Path

from docintel.ingest.cache import parse_cached
from docintel.ingest.models import Document
from docintel.ingest.native import NativeParser


def format_summary(doc: Document) -> str:
    total_chars = sum(page.char_count for page in doc.pages)
    lines = [
        f"{doc.parser}   {len(doc.pages)} pages   {doc.parse_seconds:.2f} s",
        f"chars: {total_chars}   "
        f"density: {doc.quality.char_density:.4f}   "
        f"scanned: {doc.quality.scanned_pages}",
    ]
    for page in doc.pages:
        mark = "SCAN" if page.is_scanned else "text"
        lines.append(
            f"  p.{page.number:<3d} {mark:5s}  "
            f"{len(page.blocks):3d} blocks  "
            f"{len(page.tables):2d} tables  "
            f"{page.char_count:6d} chars"
        )
    lines.extend(f"  ! {warning}" for warning in doc.quality.warnings)
    return "\n".join(lines)


def main() -> None:
    arg_parser = argparse.ArgumentParser(
        description="Parse PDFs and print the extracted document structure."
    )
    arg_parser.add_argument("paths", nargs="+", type=Path, help="paths to PDF files")
    arg_parser.add_argument("--refresh", action="store_true", help="ignore cached parse results")
    args = arg_parser.parse_args()

    native = NativeParser()
    for path in args.paths:
        print(f"\n=== {path.name} ===")
        if not path.exists():
            print("  file not found")
            continue
        try:
            doc = parse_cached(native, path, refresh=args.refresh)
        except Exception as exc:
            print(f"  parse failed: {type(exc).__name__}: {exc}")
            continue
        print(format_summary(doc))
