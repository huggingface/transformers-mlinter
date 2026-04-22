# mlinter

A standalone linter for [Hugging Face Transformers](https://github.com/huggingface/transformers) modeling files. It enforces structural conventions on every `modeling_*.py`, `modular_*.py`, and `configuration_*.py` file under `src/transformers/models`.

## Installation

```bash
pip install git+https://github.com/huggingface/transformers-mlinter@main
```

When working on the transformers repo, mlinter is included in the `quality` extras:

```bash
pip install -e ".[quality]"
```

## How rule registration works

- Rule metadata lives in `mlinter/rules.toml`.
- The TOML schema is versioned with a top-level `version = 1`. Custom files passed with `--rules-toml` must use the same schema version.
- Executable TRF rules are auto-discovered from `trf*.py` modules in the `mlinter/` package.
- Each module must define a `check(tree, file_path, source_lines) -> list[Violation]` function.
- The module name determines the rule id: `trf003.py` → `TRF003`.
- A `RULE_ID` module-level constant is set automatically by the discovery mechanism.
- Every discovered rule must have a matching entry in the TOML file, and every TOML rule must have a matching module. Import-time validation fails if either side is missing.
- Suppressions use `# trf-ignore: TRFXXX` on the same line or the line immediately above the flagged construct.

## How to add a new TRF rule

1. Add a `[rules.TRFXXX]` entry to `mlinter/rules.toml`.
2. Fill in `description`, `default_enabled`, `explanation.what_it_does`, `explanation.why_bad`, and `explanation.diff`. Optional model-level exceptions go in `allowlist_models`.
3. Create a new module `mlinter/trfXXX.py` with a `check(tree, file_path, source_lines) -> list[Violation]` function.
4. Use the `RULE_ID` module constant instead of hardcoding `"TRFXXX"` inside the check.
5. Add or update focused tests in `tests/`.

## CLI usage

Run from the root of a transformers checkout:

```bash
# Check all modeling, modular, and configuration files
mlinter

# Only check files changed against a git base ref
mlinter --changed-only --base-ref origin/main

# List all available TRF rules and their default state
mlinter --list-rules

# Use a custom rules TOML instead of the bundled mlinter/rules.toml
mlinter --rules-toml /path/to/custom-rules.toml

# Show the installed mlinter version
mlinter --version

# Show detailed documentation for one rule
mlinter --rule TRF001

# Enable additional rules on top of the defaults
mlinter --enable-rules TRF003

# Enable every TRF rule, including ones disabled by default
mlinter --enable-all-trf-rules

# Emit GitHub Actions error annotations
mlinter --github-annotations
```

When installed from a git checkout or a `git+https://...` URL, `mlinter --version` includes a short commit hash suffix such as `0.1.1+gabcdef1`.

You can also invoke it as a Python module:

```bash
python -m mlinter
```

The lint cache is stored in the user cache directory instead of next to the installed package:
`$XDG_CACHE_HOME/mlinter/.mlinter_cache.json` on Linux, `~/Library/Caches/mlinter/.mlinter_cache.json` on macOS, and `%LOCALAPPDATA%\mlinter\.mlinter_cache.json` on Windows.

## Python API

Import the supported Python API from the package root:

```python
from mlinter import TRF001, analyze_file, model_dir_name, render_rules_reference
```

`mlinter.mlinter` and `mlinter._helpers` are implementation modules and may change without a compatibility promise.

## Development

```bash
git clone https://github.com/huggingface/transformers-mlinter
cd transformers-mlinter
pip install -e ".[dev]"
```

Run the tests:

```bash
pytest tests/
```

## Releasing

See [docs/release.md](docs/release.md) for the current release process.
