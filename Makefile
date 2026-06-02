.PHONY: install test lint serve

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check graphctx/ tests/
	ruff format --check graphctx/ tests/

serve:
	graphctx serve --db /tmp/graphctx.db --port 8420
