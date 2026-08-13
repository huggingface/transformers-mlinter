---
layout: default
title: Contributing a rule
nav_order: 5
description: "How to add a new TRF rule to mlinter: the skill-driven path, the manual steps, and the constraints every rule has to satisfy."
---

# Contributing a rule
{: .no_toc }

## On this page
{: .no_toc .text-delta }

- TOC
{:toc}

---

## Development setup

```bash
git clone https://github.com/huggingface/transformers-mlinter
cd transformers-mlinter
pip install -e ".[dev]"
```

The repo's own checks:

```bash
make test        # pytest under tests/
make lint        # ruff check + format --check
make format      # auto-fix style
make typecheck   # ty on mlinter/
make docs        # regenerate the rule pages and build this site
```

## The guided path

The repo ships an `add-mlinter-rule` skill that walks an agent through duplicate detection, numbering,
module creation, running the candidate against every model in the library, and deciding whether the
findings mean "fix the models" or "allowlist them". Enable it for your agent:

```bash
make claude   # symlinks .claude/skills -> .ai/skills (for Claude Code)
make codex    # symlinks .agents/skills -> .ai/skills (for Codex)
```

Then invoke `/add-mlinter-rule` in a new session.

The step that matters most is the one that is easy to skip: **run the candidate rule against the whole
library before proposing it.** A rule that looks obviously correct routinely turns up dozens of
pre-existing violations, and how many there are decides whether the rule needs a cutoff date, an
allowlist, or a rethink.

## The manual steps

1. Add a `[rules.TRFXXX]` entry to
   [`mlinter/rules.toml`](https://github.com/huggingface/transformers-mlinter/blob/main/mlinter/rules.toml).
2. Fill in `description`, `default_enabled`, `explanation.what_it_does`, `explanation.why_bad`, and
   `explanation.diff`. Optional: `allowlist_models` for per-model exemptions and `cutoff_date` to scope
   the rule to newer models.
3. Create `mlinter/trfXXX.py` with a `check(tree, file_path, source_lines) -> list[Violation]` function.
4. Use the `RULE_ID` module constant instead of hardcoding `"TRFXXX"` inside the check.
5. Add or update focused tests in `tests/`.

Registration is automatic from there: rule modules are discovered by filename, and import-time
validation fails if a module has no TOML entry or a TOML entry has no module. There is no registry list
to update.

## Constraints on a rule

- **Static analysis only.** Use Python's `ast` module. A rule must never import the model, download
  weights, or execute the file under inspection.
- **Gate on the filename.** Rules are handed every file kind mlinter discovers, so a rule that only
  makes sense for `modeling_*.py` has to check the prefix itself. Widening file discovery must not
  expose an existing rule to a file type it was never written for.
- **One `check` signature.** `check(tree, file_path, source_lines) -> list[Violation]`.
- **Honour suppressions.** Call the shared suppression helper rather than reimplementing the comment
  scan, unless the rule deliberately supports no suppression — in which case say so in a comment, as
  `TRF038` does.
- **Cross-file reads are allowed but must be cheap.** Some rules read the companion
  `configuration_*.py` from disk. The cache accounts for this: a modeling file is re-checked when its
  companion config changes.

## Writing the explanation

The three `explanation` fields are what a contributor sees when a rule fires on their pull request, both
in `mlinter --rule TRFXXX` and on this site's [rule pages](rules/index.md), which are generated from
them.

- `what_it_does` — the mechanical description. What construct is detected.
- `why_bad` — the consequence. What actually breaks, or what a reader gets wrong. "It is inconsistent"
  is not a consequence; "it can break weight loading key mapping" is.
- `diff` — a single diff block with `-` for the flagged form and `+` for the fix. Keep it to the few
  lines that carry the difference. Use `Acme` as the model name, matching the existing rules.

Prose may contain `code spans`; the site generator escapes everything around them, so `<Model>Config`
and `modeling_*.py` render as written rather than being eaten as markup.

## Documentation

This site's rule reference is generated, so **adding a rule to `rules.toml` documents it** — there is no
separate page to write. Run `make docs` to see how it renders. Only the hand-written pages (home, CLI
usage, suppressing, this page) live as files under `docs/`.
