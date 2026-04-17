## Useful commands
- `make format`: auto-fixes style issues with Ruff.
- `make lint`: runs Ruff checks and format verification.
- `make test`: runs the test suite under `tests/`.
- `make typecheck`: runs `ty` on `mlinter/`.

Run `make format` or `make lint` and `make test` before wrapping up a change.

## Repo structure

- Rule modules live in `mlinter/trf*.py`.
- Rule metadata lives in `mlinter/rules.toml`.
- Focused tests live in `tests/test_mlinter.py`.
- Skills live under `.ai/skills/`.

## Mlinter rules

- Rules must use static analysis only with Python's `ast` module.
- Rule modules must expose `check(tree, file_path, source_lines) -> list[Violation]`.
- Rule specs use `what_it_does`, `why_bad`, and `diff`; keep examples in a single diff block.
- Cross-file rules may need to read companion `configuration_*.py` files from disk.
