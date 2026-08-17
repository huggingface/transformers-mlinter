---
name: remove-mlinter-rule
description: Retire a TRF rule from the mlinter. Marks the rule deprecated in rules.toml, deletes its module and tests, and records the removal in the CHANGELOG. Use when asked to remove, retire, drop, or delete a rule.
---

# Remove Mlinter Rule

## Input

- `<rule id>`: the rule to retire, e.g. `TRF054`. Accept a bare number (`54`) and normalise it to
  `TRFXXX`, zero-padded to three digits.
- Optional: the reason for the removal. Ask for one if none is given — it becomes the CHANGELOG entry
  and the tombstone description, and it is the only record contributors will have of why the rule went
  away.

## Why a tombstone and not a deletion

Rule ids are referenced from outside this repo: `# trf-ignore: TRFXXX` comments in the transformers
tree, CI configs, and copies of `rules.toml` passed via `--rules-toml`. So a removal keeps a tombstone
entry in `rules.toml` instead of deleting the table:

- The engine drops a deprecated id from `TRF_RULES`, `TRF_RULE_SPECS`, `TRF_RULE_CHECKS`,
  `DEFAULT_ENABLED_TRF_RULES`, the `mlinter.TRFXXX` public constants, and `--list-rules`. Projects that
  still suppress or configure the id are silently unaffected.
- The docs site is the exception: it keeps publishing a page for the id, built from the tombstone and
  listed under `Removed rules` on the rule index. A number that used to fire has to stay findable by
  whoever meets it in an old CI log.
- Asking for it explicitly (`--enable-rules TRFXXX`, `--rule TRFXXX`) fails with exit code 2.
- A rules TOML that still lists the id as an active rule fails the whole run with exit code 2. The
  bundled `rules.toml` is the authority here, so this holds for custom files too.
- Leaving `trfXXX.py` on disk while the id is deprecated is also an error, which is what makes a
  half-finished removal impossible to ship.

Never reuse a retired number for a new rule: the tombstone stays forever, and `add-mlinter-rule` picks
the next number after the highest `trf*.py`.

## Workflow

1. Confirm the rule exists and read it.
   - `grep -n "\[rules.TRFXXX\]" mlinter/rules.toml` and read the whole table plus `mlinter/trfXXX.py`.
   - If the id is already marked `deprecated = true`, stop and report that it is already retired.
   - Summarise for the user what the rule checks and any `allowlist_models` / `cutoff_date` it carries,
     so they can confirm this is the rule they mean before anything is deleted.

2. Replace the TOML table with a tombstone in `mlinter/rules.toml`.
   - Keep the table in place at its original position — ordering in this file is by rule id.
   - Delete `default_enabled`, `allowlist_models`, `cutoff_date`, and the whole
     `[rules.TRFXXX.explanation]` table.
   - Leave exactly:
     ```toml
     [rules.TRFXXX]
     deprecated = true
     description = "Removed in <version>: <one-line reason>."
     ```
   - `<version>` is the version under development in `pyproject.toml`. The description **is** published:
     it is the whole of the retired rule's docs page and its row in the `Removed rules` table, so write
     it for someone who just hit the id in a CI log, not as an internal note.

3. Delete the rule module.
   ```bash
   git rm mlinter/trfXXX.py
   ```

4. Delete the tests.
   - Remove the rule's own test block in `tests/test_mlinter.py` — the section under the
     `# --- TRFXXX: ... ---` comment, up to the next `# ---` marker.
   - Remove the id from the public-API tests: the `assertEqual(public_api.TRFXXX, "TRFXXX")` line and the
     `assertIn("TRFXXX", public_api.__all__)` line.
   - `grep -rn "TRFXXX" tests/ mlinter/` must come back empty except for the tombstone in `rules.toml`.
     Watch for helper fixtures or shared source strings that only that rule used, and for any other rule's
     test that happened to reference it.
   - No new test is needed for the removal itself: `test_bundled_deprecated_rules_are_fully_retired` loops
     over every tombstone in the bundled `rules.toml` and asserts the module and the public constant are
     gone, so it starts covering the new id automatically.

5. Update `CHANGELOG.md`.
   - Under `## [Unreleased]`, in a `### Removed` section (create it if absent), state the id, what it
     checked, and why it is gone. Add the migration note that projects need no change: mlinter ignores the
     id, existing `# trf-ignore: TRFXXX` comments are harmless, but a rules TOML of their own must mark the
     rule `deprecated = true` or drop it.
   - If the removed rule was mentioned in `README.md`, `docs/index.md`, or any other hand-written doc page,
     update those too. `docs/rules/` is generated and git-ignored — there is nothing to delete there.

6. Verify.
   ```bash
   make lint
   make test
   make typecheck
   python -m mlinter --list-rules | grep TRFXXX   # must print nothing
   python -m mlinter --rule TRFXXX                # must exit 2 and say the rule is deprecated
   python -m mlinter --enable-all-trf-rules       # must not error on the missing module
   ```
   - `make docs-rules` if the docs toolchain is available, to confirm the generator emits a `Removed`
     page for the id and lists it under `Removed rules` on the index — not a live rule page, and not
     nothing at all.

7. Report.
   - The tombstone, the deleted files, the CHANGELOG entry, and what a consumer of mlinter has to do
     (nothing, unless they ship their own `rules.toml`).

## Reference

- Rule metadata and tombstones: `mlinter/rules.toml`
- Deprecation handling: `_load_rule_specs`, `_build_rule_checks`, `_validate_rule_ids`, and
  `BUNDLED_DEPRECATED_TRF_RULES` in `mlinter/mlinter.py`
- Public API surface: `mlinter/__init__.py`
- Tests: `tests/test_mlinter.py` (engine behaviour under "Deprecated rules")
- The inverse skill: `.ai/skills/add-mlinter-rule/SKILL.md`
