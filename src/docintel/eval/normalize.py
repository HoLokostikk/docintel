from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

CURRENCY_SYMBOLS = "$\u20ac\u00a3\u00a5\u20b4"

LEADER_DOTS = re.compile(r"(?:[.\u00b7\u2024]\s*){3,}")
WHITESPACE = re.compile(r"\s+")
EUROPEAN_FORMAT = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?")
PLAIN_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def normalize_text(value: str) -> str:

    cleaned = LEADER_DOTS.sub(" ", value)
    cleaned = cleaned.replace("\u00a0", " ")
    return WHITESPACE.sub(" ", cleaned).strip()


def parse_number(value: str) -> Decimal | None:
    text = normalize_text(value)
    if not text:
        return None

    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()

    for symbol in CURRENCY_SYMBOLS:
        text = text.replace(symbol, "")
    text = text.replace("%", "").replace("\u2212", "-").strip()

    if text.startswith("-"):
        negative = True
        text = text[1:].strip()
    elif text.startswith("+"):
        text = text[1:].strip()

    text = text.replace(" ", "").replace("'", "")
    if not text:
        return None

    if EUROPEAN_FORMAT.fullmatch(text):
        text = text.replace(".", "").replace(",", ".")
    else:
        text = text.replace(",", "")

    if not PLAIN_NUMBER.fullmatch(text):
        return None

    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    return -number if negative else number


def format_number(number: Decimal) -> str:

    return f"{number.normalize():f}"


def normalize_cell(value: str) -> str:

    number = parse_number(value)
    if number is not None:
        return format_number(number)
    return normalize_text(value).casefold()
