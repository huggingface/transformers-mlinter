.PHONY: build build-release test lint format typecheck clean codex claude

build:
	pip install -e ".[dev]"

build-release:
	rm -rf build/ dist/ *.egg-info
	python -m build

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

codex:
	mkdir -p .agents
	rm -rf .agents/skills
	ln -snf ../.ai/skills .agents/skills

claude:
	mkdir -p .claude
	rm -rf .claude/skills
	ln -snf ../.ai/skills .claude/skills
