.PHONY: build build-release test lint format typecheck clean codex claude docs docs-serve docs-rules

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
	rm -rf docs/_site docs/.jekyll-cache docs/rules

# Regenerate the rule reference from mlinter/rules.toml. The output is git-ignored; the site build and
# the Pages workflow both run this first, so the docs cannot describe a rule that no longer exists.
docs-rules:
	python scripts/build_docs.py

# Full site build, including the internal link check the Pages workflow runs. Needs the Ruby toolchain:
# `cd docs && bundle install` once.
docs: docs-rules
	cd docs && bundle exec jekyll build --strict_front_matter --trace
	cd docs && bundle exec htmlproofer _site --disable-external --allow-hash-href --ignore-empty-alt \
		--swap-urls "^$$(bundle exec ruby -ryaml -e 'print YAML.load_file("_config.yml")["baseurl"].to_s'):"

# Live preview on http://localhost:4000/transformers-mlinter/. Editing a rule needs `make docs-rules`
# to show up; the hand-written pages reload on save.
docs-serve: docs-rules
	cd docs && bundle exec jekyll serve --livereload

codex:
	mkdir -p .agents
	rm -rf .agents/skills
	ln -snf ../.ai/skills .agents/skills

claude:
	mkdir -p .claude
	rm -rf .claude/skills
	ln -snf ../.ai/skills .claude/skills
