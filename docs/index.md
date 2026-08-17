---
layout: default
title: Home
nav_order: 1
description: "mlinter is a linter for Hugging Face Transformers model integration files, enforcing the structural conventions that keep modeling code consistent across the library."
permalink: /
---

# mlinter <span class="version-badge">v{{ site.data.mlinter.version }}</span>
{: .no_toc }

A standalone linter for [Hugging Face Transformers](https://github.com/huggingface/transformers) model
integration files. It enforces the structural conventions that keep hundreds of model implementations
consistent with each other.
{: .fs-6 .fw-300 }

[Browse the rules](rules/index.md){: .btn .btn-primary .mr-2 }
[View on GitHub](https://github.com/huggingface/transformers-mlinter){: .btn }

Latest release **v{{ site.data.mlinter.version }}** — see the [changelog](changelog.md) for what
changed. This site is built from `main`, so it may describe rules that are not in the release yet.
{: .fs-3 .fw-300 }

---

## Why it exists

Adding a model to Transformers means writing files that look like every other model's files. The
conventions are real and they are documented — there are simply too many of them to hold in your head
at once: which class attribute has to match which config name, which decorator replaces a hand-rolled
`return_dict` branch, which base class a tokenizer test has to inherit. So they get checked in review,
one pull request at a time. That works, and it is a lot to ask: it spends reviewer attention on
mechanical details that a machine can check, when the interesting question is whether the model itself
is right.

mlinter turns those conventions into checks. Each rule is a small static analysis over the file's
syntax tree — it never imports the model, downloads weights, or runs a forward pass, so a full sweep
of the library is a few seconds of parsing.

## What it checks

Every file under `src/transformers/models/` matching one of these patterns:

`modeling_*.py`, `modular_*.py`, `configuration_*.py`, `processing_*.py`, `image_processing_*.py`,
`video_processing_*.py`, `feature_extraction_*.py`

plus `test_tokenization_*.py` under `tests/models/`.

The [rule reference](rules/index.md) lists every check, whether it runs by default, and which models
are exempt from it.

## Installation

```bash
pip install transformers-mlinter
```

Or straight from the repo:

```bash
pip install git+https://github.com/huggingface/transformers-mlinter@main
```

When working on the transformers repo itself, mlinter comes with the `quality` extras:

```bash
pip install -e ".[quality]"
```

## Quick start

Run it from the root of a transformers checkout:

```bash
# Check every model integration file
mlinter

# Only the files you changed
mlinter --changed-only --base-ref origin/main

# Explain one rule
mlinter --rule TRF001
```

See [CLI usage](usage.md) for the full set of flags, the Python API, and cache locations.

## How rules are organised

Rules are named `TRF001`, `TRF002`, and so on. There are no categories or prefixes to learn — the
[rule index](rules/index.md) is one filterable table.

Two mechanisms keep a new rule from retroactively failing the whole library:

- **Cutoff dates.** A rule that encodes a convention introduced at a point in time only applies to
  models contributed on or after that date. Older models are grandfathered without needing to be
  listed anywhere.
- **Model allowlists.** Individual models that predate a convention and cannot change without breaking
  backward compatibility are exempted by name in `rules.toml`. Each rule page lists its own.

Both are visible on every rule page, along with a link to the rule's source module.

## How rule registration works

- Rule metadata lives in [`mlinter/rules.toml`](https://github.com/huggingface/transformers-mlinter/blob/main/mlinter/rules.toml).
- The TOML schema is versioned with a top-level `version = 1`. Custom files passed with `--rules-toml`
  must use the same schema version.
- Executable TRF rules are auto-discovered from `trf*.py` modules in the `mlinter/` package.
- Each module must define a `check(tree, file_path, source_lines) -> list[Violation]` function.
- The module name determines the rule id: `trf003.py` → `TRF003`.
- A `RULE_ID` module-level constant is set automatically by the discovery mechanism.
- Every discovered rule must have a matching entry in the TOML file, and every TOML rule must have a
  matching module. Import-time validation fails if either side is missing.
- A retired rule keeps a tombstone entry — `deprecated = true` and nothing else that matters — and loses
  its module. mlinter then ignores that id everywhere: it disappears from `--list-rules`, from the rule
  pages, and from the default set, and asking for it (`--enable-rules`, `--rule`) is an error. A rules
  TOML that still describes it as a live rule is rejected outright, so a project cannot keep failing CI
  on a rule whose code is gone.

That last point is why this site cannot drift: the rule pages are generated from `rules.toml` on every
build, and `rules.toml` cannot describe a rule that has no code behind it.

## Suppressing a rule

Use `# trf-ignore: TRFXXX` on the flagged line or the line directly above it. Some rules also honour a
module-level directive that exempts named subjects for a whole file. See
[Suppressing rules](suppressing.md).

## Contributing

New rules are welcome — the repo ships a skill that walks an agent through the whole process, from
duplicate detection to running the candidate rule against every model in the library. See
[Contributing a rule](contributing.md).
