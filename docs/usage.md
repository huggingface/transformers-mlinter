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

mlinter resolves paths relative to the current directory, so run it from the root of a transformers
checkout.

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

Disable the cache with `--no-cache`.

## Python API

Import the supported API from the package root:

```python
from mlinter import TRF001, analyze_file, model_dir_name, render_rules_reference
```

`mlinter.mlinter` and `mlinter._helpers` are implementation modules and may change without a
compatibility promise.
