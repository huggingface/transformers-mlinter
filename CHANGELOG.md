# Changelog

All notable changes to `transformers-mlinter` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `TRF016`, which flags `do_*` boolean flags declared on image/video processor classes that are not referenced
  by an overridden `preprocess` / `_preprocess` method (default disabled).
- Added `TRF018`, which flags `_init_weights` overrides on `PreTrainedModel` subclasses that do not chain via
  `super()._init_weights(...)` (or the modular-file equivalent `<Class>._init_weights(self, ...)`). Models that
  intentionally fully override initialization can suppress with `# trf-ignore: TRF018`. Modular files using the
  `raise AttributeError(...)` delete-sentinel are skipped. See
  https://github.com/huggingface/transformers/pull/45597 for the bug class this catches.
- Expanded the set of files the linter targets to include `image_processing_*.py` and `video_processing_*.py` in
  addition to `modeling_*.py`, `modular_*.py`, and `configuration_*.py`. This affects file discovery for every rule,
  not just `TRF016`.

## [0.1.1] - 2026-04-22

### Added

- Added `--rules-toml` so the CLI can load rule metadata from a custom TOML file instead of the bundled
  `mlinter/rules.toml`.
- Added schema version validation for rule-spec TOML files and included the active rule-spec hash in the lint cache so
  custom rule sets do not reuse stale cache entries.

### Fixed

- Fixed `TRF005` so modular files may use `AttributeError()` as the sentinel for removing `_no_split_modules` during
  generated-code cleanup, while `modeling_*.py` files still require a list or tuple of non-empty strings.

## [0.1.0] - 2026-04-21

### Added

- Initial release of `transformers-mlinter`.
