.PHONY: build test lint format typecheck clean

build:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

lint:
	ruff check mlinter/ tests/
	ruff format --check mlinter/ tests/

format:
	ruff check --fix mlinter/ tests/
	ruff format mlinter/ tests/

typecheck:
	ty check mlinter/

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
