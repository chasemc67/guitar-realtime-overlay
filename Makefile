.PHONY: install test lint fmt anchors demo review eval-auto

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

review:
	uv run python scripts/review_sheet.py samples/anchors --batch batch-000-hand

eval-auto:
	uv run python scripts/eval_auto.py --json eval/auto_eval_ai.json
	uv run python scripts/eval_auto.py --no-ai --json eval/auto_eval_noai.json
