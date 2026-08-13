<!-- Absolute URL on purpose: this README is also the PyPI long_description, and PyPI does not resolve
     repo-relative image paths. -->
<p align="center">
  <img src="https://raw.githubusercontent.com/huggingface/transformers-mlinter/main/docs/assets/images/mlinter-logo.png"
       alt="mlinter" width="360">
</p>

# mlinter

A standalone linter for [Hugging Face Transformers](https://github.com/huggingface/transformers) model
integration files — `modeling_*.py`, `modular_*.py`, `configuration_*.py`, `processing_*.py`,
`image_processing_*.py`, `video_processing_*.py` and `feature_extraction_*.py` under
`src/transformers/models/`, plus `test_tokenization_*.py` under `tests/models/`. It enforces the
structural conventions that keep hundreds of model implementations consistent with each other.

**📖 Documentation: <https://huggingface.github.io/transformers-mlinter/>**

The docs site is generated from `mlinter/rules.toml`, so its
[rule reference](https://huggingface.github.io/transformers-mlinter/rules/) is always in step with the
installed rules.

## Installation

```bash
pip install transformers-mlinter
```

When working on the transformers repo, mlinter is included in the `quality` extras:

```bash
pip install -e ".[quality]"
```

## Quick start

Run from the root of a transformers checkout:

```bash
mlinter                                     # check every model integration file
mlinter --changed-only --base-ref origin/main   # only what you changed
mlinter --list-rules                        # list rules and their default state
mlinter --rule TRF001                       # explain one rule
```

See the [CLI reference](https://huggingface.github.io/transformers-mlinter/usage/) for every flag, the
Python API, and cache locations.

## Documentation map

| Page | What's there |
|:-----|:-------------|
| [Home](https://huggingface.github.io/transformers-mlinter/) | What mlinter checks and why, installation, how rule registration works |
| [Rules](https://huggingface.github.io/transformers-mlinter/rules/) | All rules, filterable, one page each with examples and exemptions |
| [CLI usage](https://huggingface.github.io/transformers-mlinter/usage/) | Every flag, output formats, cache, Python API |
| [Suppressing rules](https://huggingface.github.io/transformers-mlinter/suppressing/) | `# trf-ignore`, whole-file directives, cutoff dates, allowlists |
| [Contributing a rule](https://huggingface.github.io/transformers-mlinter/contributing/) | Adding a rule, the `add-mlinter-rule` skill, constraints on a rule |
| [Releasing](https://huggingface.github.io/transformers-mlinter/release/) | The tag-driven release process |

## Development

```bash
git clone https://github.com/huggingface/transformers-mlinter
cd transformers-mlinter
pip install -e ".[dev]"
```

```bash
make test        # pytest under tests/
make lint        # ruff check + format --check
make format      # auto-fix style
make typecheck   # ty on mlinter/
```

### Building the docs site

The rule pages are generated and git-ignored. Building needs the Ruby toolchain once:

```bash
cd docs && bundle install && cd ..
make docs         # regenerate rule pages, build the site, check internal links
make docs-serve   # live preview on http://localhost:4000/transformers-mlinter/
```

## License

Apache-2.0. See [LICENSE](LICENSE).
