.PHONY: dev fmt lint type test check

dev:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check


type:
	uv run mypy

test:
	uv run pytest


check: lint type test

fmt:
	uv run ruff format .
	uv run ruff check --fix
