## Useful commands
- `make format`: auto-fixes style issues with Ruff.
- `make lint`: runs Ruff checks and format verification.
- `make test`: runs the test suite under `tests/`.
- `make typecheck`: runs `ty` on `mlinter/`.
- `make docs`: regenerates the rule pages, builds the docs site, and checks internal links.
- `make docs-serve`: live preview of the docs site.

Run `make format` or `make lint` and `make test` before wrapping up a change.

## Repo structure

- Rule modules live in `mlinter/trf*.py`.
- Rule metadata lives in `mlinter/rules.toml`.
- Focused tests live in `tests/test_mlinter.py`.
- Skills live under `.ai/skills/`.
- The docs site source lives in `docs/` (Jekyll + just-the-docs), published to
  <https://huggingface.github.io/transformers-mlinter/> by `.github/workflows/pages.yml`.

## Docs site

- `docs/rules/` is **generated** by `scripts/build_docs.py` from `rules.toml` and is git-ignored.
  Never hand-edit it, and never commit it.
- Adding a rule to `rules.toml` therefore documents it: there is no page to write. The `description`
  and `explanation` fields are the published prose, so write them for a contributor reading a failing
  CI job.
- Only the hand-written pages (`index.md`, `usage.md`, `suppressing.md`, `contributing.md`,
  `release.md`) are committed under `docs/`. `docs/rules/`, `docs/changelog.md` (copied from the root
  `CHANGELOG.md`) and `docs/_data/mlinter.yml` (version from `pyproject.toml`) are all generated.
- Rule prose may contain `code spans`; the generator escapes Markdown metacharacters around them, so
  `<Model>Config` and `modeling_*.py` render literally. Don't pre-escape them in the TOML.
- The Ruby toolchain needs `cd docs && bundle install` once before `make docs` works.

## Mlinter rules

- Rules must use static analysis only with Python's `ast` module.
- Rule modules must expose `check(tree, file_path, source_lines) -> list[Violation]`.
- Rule specs use `what_it_does`, `why_bad`, and `diff`; keep examples in a single diff block.
- Cross-file rules may need to read companion `configuration_*.py` files from disk.
