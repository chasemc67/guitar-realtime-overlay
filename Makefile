.PHONY: install test lint fmt anchors demo

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

anchors:
	uv run python scripts/hand_anchors.py --debug

demo:
	uv run python scripts/render_demo.py --resnap
