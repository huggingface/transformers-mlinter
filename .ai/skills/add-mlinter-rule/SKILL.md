---
name: add-mlinter-rule
description: Add a new TRF rule to the mlinter. Checks for duplicates, creates the rule module and TOML entry, runs against all models, and handles violations (fix or allowlist).
---

# Add Mlinter Rule

## Input

- `<description>`: Natural-language description of what the rule should detect.
- Optional: specific AST pattern or a before/after diff showing the pattern.

## Constraints

- Rules MUST use static analysis only with Python's `ast` module. NEVER import runtime libraries like `torch` or `tensorflow`.
- Rules MUST follow the `check(tree, file_path, source_lines) -> list[Violation]` interface.
- Use the module-level `RULE_ID` constant instead of hardcoding the rule ID string.

## Workflow

1. Check for duplicate coverage in `mlinter/rules.toml`.
   - Read the full TOML file and review existing rule descriptions and explanations.
   - If an existing rule already covers the same concern, stop and ask whether to proceed, extend the existing rule, or abort.

2. Determine the next rule number.
   - List all `mlinter/trf*.py` files and find the highest number.
   - The new rule gets that number + 1, zero-padded to three digits.

3. Add the TOML entry to `mlinter/rules.toml`.
   - Append a new `[rules.TRFXXX]` section at the end of the file with:
     - `description`
     - `default_enabled = true`
     - `allowlist_models = []`
     - `[rules.TRFXXX.explanation]` with `what_it_does`, `why_bad`, and `diff`
   - Follow the exact formatting style of existing entries.

4. Create the rule module at `mlinter/trfXXX.py`.
   - Start with the Apache 2.0 license header copied from an existing `trf*.py` file.
   - Add a module docstring: `"""TRFXXX: <short description>."""`
   - Import `ast`, `Path`, and any needed helpers from `._helpers`.
   - Define `RULE_ID = ""  # Set by discovery`.
   - Implement `def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:`.
   - Refer to existing rules in `mlinter/trf*.py` for patterns and helpers.

5. Run the rule against all models.
   ```bash
   python -m mlinter --enable-rules TRFXXX
   ```
   - If the run errors, fix the rule code and re-run.

6. Handle violations.
   - Present the list of violations to the user.
   - Ask whether to fix the models or add them to `allowlist_models` in `mlinter/rules.toml`.
   - If fixing, apply the fixes and re-run the rule to confirm zero violations.
   - If allowlisting, extract the model directory names from the violation file paths and add them to `allowlist_models`.

7. Add tests in `tests/test_mlinter.py`.
   - Add at least one positive test and one negative test.
   - Follow the existing pattern: create source strings, call `mlinter.analyze_file()`, and assert on violations.
   - For cross-file rules, use `tempfile.TemporaryDirectory` to create real file structures.
   - If the rule maps a modeling class to a specific config class, add a regression where another config class in the same file would otherwise cause a false positive or false negative.
   - Run the focused tests:
   ```bash
   pytest tests/test_mlinter.py -x -v -k "trfXXX"
   ```

8. Update documentation.
   - Add an entry under the `## [Unreleased]` section of `CHANGELOG.md` (create that section above the latest released version if it does not yet exist) describing the new rule. Mention any incidental changes shipped with it (e.g. expanding `MODELING_PATTERNS` to cover new file types), since those affect every rule.
   - If the rule applies to file types not already documented in `README.md`, update the README accordingly.

9. Final validation.
   ```bash
   make lint
   make test
   ```

## Model architecture knowledge

The mlinter processes files one at a time via `analyze_file(file_path, text, enabled_rules)`. When a rule needs cross-file information, the rule module must read the other file from disk. Watch for these patterns:

### Multi-config directories

Some model directories contain multiple configuration files. Match by suffix first:
`modeling_foo_text.py` -> `configuration_foo_text.py`.
Only fall back to a generic `configuration_*.py` pick when there is no exact suffix match.

### Multi-class configuration files

A single `configuration_*.py` file can define multiple config classes. If the rule is checking a property that belongs to one specific config class, do not accept the first matching class in the file. Resolve the modeling class's target config class first:

- Prefer `config_class` from the model class, following local modeling inheritance if a parent `*PreTrainedModel` declares it.
- If there is no explicit `config_class`, infer the best match from class names, usually by longest shared prefix.

Then validate only that config class.

### Inherited configs

Some config classes inherit from another model config rather than directly from `PreTrainedConfig`. If the base class is not `PreTrainedConfig` or `PretrainedConfig` and still ends with `Config`, assume the field may be inherited and skip the violation unless the rule specifically needs stricter handling.

### `tie_word_embeddings` is not in `PreTrainedConfig`

The base `PreTrainedConfig` does not define `tie_word_embeddings`. When a rule needs it, the model config must declare it explicitly, either as a class attribute or through `self.tie_word_embeddings = ...` in initialization code.

## Reference

- Rule modules: `mlinter/trf*.py`
- Rule config: `mlinter/rules.toml`
- Helpers: `mlinter/_helpers.py`
- Tests: `tests/test_mlinter.py`
- README: `README.md`
