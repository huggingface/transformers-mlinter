---
layout: default
title: CLI usage
nav_order: 2
description: "Every mlinter command-line flag, the supported Python API, and where the lint cache lives."
---

# CLI usage
{: .no_toc }

## On this page
{: .no_toc .text-delta }

- TOC
{:toc}

---

With no path argument, mlinter resolves paths relative to the current directory, so run it from the
root of a transformers checkout. Pass a path to check any other directory — see
[Checking a repository outside transformers](#checking-a-repository-outside-transformers).

## Checking files

```bash
# Check all model integration files
mlinter

# Only check files changed against a git base ref
mlinter --changed-only --base-ref origin/main
```

`--changed-only` is what CI uses on pull requests. It diffs against the base ref and lints only the
matching files, which keeps a run proportional to the size of the change rather than the size of the
library.

## Checking a repository outside transformers

Model code shipped on the Hub with `trust_remote_code` lives in a flat repository rather than under
`src/transformers/models/`, but it has to honour the same conventions: a model that skips `post_init`
or hardcodes a dimension breaks `AutoModel.from_pretrained` in ways that are hard to trace back. Give
mlinter the files or directories to check and it stops assuming the transformers layout:

```bash
# Check a cloned Hub model repository
mlinter ~/models/LLaDA-8B-Instruct

# Check one file, or several paths at once
mlinter ~/models/LLaDA-8B-Instruct/modeling_llada.py
mlinter src/transformers/models/llama tests/models/llama
```

A directory is searched recursively for model integration files (`modeling_*.py`, `modular_*.py`,
`configuration_*.py`, `processing_*.py`, `image_processing_*.py`, `video_processing_*.py`,
`feature_extraction_*.py`, `test_tokenization_*.py`); a file named explicitly is checked as given.
Since the search is recursive and takes several paths at once, the layout does not matter: a GitHub
project that keeps its model files in some directory of its own is checked by naming that directory,
or by naming each one when they are scattered. Rules gate on the file name, so a file named something
else — `model.py`, say — runs no rules, and mlinter says so rather than reporting a clean run. A path
that does not exist is an error: the run stops with exit code 2 and names it, rather than quietly
checking the rest. Files generated from a `modular_*.py` source are skipped here as they are in a
checkout, so the modular file is the one reported on. `--changed-only` composes with paths: it narrows
the git diff to the paths you passed.

`test_tokenization_*.py` is the only test file discovered, in a checkout or out of one: [TRF042](rules/trf042.md)
is the only rule that reads a test file. `test_modeling_*.py` and `test_processing_*.py` are walked
once a rule targets them.

Rules that resolve other models (a sibling `configuration_*.py`, a test file under `tests/models/`,
another model's directory) find nothing outside a transformers checkout and stay quiet, as do
per-model allowlists and cutoff dates, which are keyed on a `src/transformers/models/<model>/` path.
Everything a single file can be judged on still applies.

No transformers checkout is needed either way: the rules ship inside the package, so
`pip install -U transformers-mlinter` is all it takes to check against the current rule set.

## Selecting rules

```bash
# List all available TRF rules and their default state
mlinter --list-rules

# Enable additional rules on top of the defaults
mlinter --enable-rules TRF003

# Enable every TRF rule, including ones disabled by default
mlinter --enable-all-trf-rules

# Use a custom rules TOML instead of the bundled mlinter/rules.toml
mlinter --rules-toml /path/to/custom-rules.toml
```

A custom rules file must declare the same top-level `version = 1` as the bundled one, and every rule it
names must have a matching `trf*.py` module in the installed package.

Rules that have been removed from mlinter are the one exception: they must be marked `deprecated = true`
(or dropped from the file). mlinter then ignores them. A custom file that still lists a removed rule as
active fails with exit code 2 and names the rule, rather than silently linting nothing under that id.

## Reading the docs from the terminal

```bash
# Show detailed documentation for one rule
mlinter --rule TRF001
```

This prints the same "what it does", "why is this bad", and diff example that the
[rule pages](rules/index.md) on this site are generated from.

## Output formats

```bash
# Emit GitHub Actions error annotations
mlinter --github-annotations
```

With `--github-annotations`, violations are printed as `::error` workflow commands, so they appear
inline on the diff of a pull request instead of only in the job log.

```bash
# Also write every finding to a JSON file
mlinter --output-json findings.json
```

`--output-json` writes the findings alongside the normal output, including the description and diff of
each rule that fired. The file is written even when there are no violations, so a consumer never has to
distinguish "clean run" from "no file produced", and the exit code is unaffected. The transformers CI
uses it to upload findings as an artifact and post them as inline review comments.

Progress output is animated when attached to a terminal; `--no-progress` disables it, which is what you
want in a log.

## Version

```bash
mlinter --version
```

When installed from a git checkout or a `git+https://...` URL, the version includes a short commit hash
suffix such as `1.2.3+g1a2b3c4`.

## Running as a module

```bash
python -m mlinter
```

Equivalent to the `mlinter` entry point, and useful when the script directory is not on `PATH`.

## Cache

The lint cache is stored in the user cache directory rather than next to the installed package:

| Platform | Location |
|:---------|:---------|
| Linux | `$XDG_CACHE_HOME/mlinter/.mlinter_cache.json` |
| macOS | `~/Library/Caches/mlinter/.mlinter_cache.json` |
| Windows | `%LOCALAPPDATA%\mlinter\.mlinter_cache.json` |

A file is skipped when its content hash, the set of enabled rules, the hash of the rules TOML, and the
contents of any companion `configuration_*.py` all match the cached entry. Editing a config file
therefore re-checks the modeling files that read it.

One cache serves every repository you check, so entries are keyed on the absolute path: two model
repositories that each hold a `modeling_llada.py` get an entry apiece rather than one shadowing the
other.

Disable the cache with `--no-cache`.

## Python API

Import the supported API from the package root:

```python
from mlinter import TRF001, analyze_file, model_dir_name, render_rules_reference
```

File discovery is part of that API, so a tool that lints a repository of its own does not have to
reimplement the patterns:

```python
from pathlib import Path

from mlinter import analyze_file, iter_modeling_files

for path in iter_modeling_files(search_paths=[Path("~/models/LLaDA-8B-Instruct").expanduser()]):
    violations = analyze_file(path, path.read_text(encoding="utf-8"))
```

`search_paths` takes the same files and directories as the command line; without it, discovery walks
`src/transformers/models` and `tests/models` relative to the current directory. `resolve_search_paths`
validates a list of paths the way the CLI does, raising `ValueError` naming any that do not exist.

`mlinter.mlinter` and `mlinter._helpers` are implementation modules and may change without a
compatibility promise.
