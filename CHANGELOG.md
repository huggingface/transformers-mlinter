# Changelog

All notable changes to `transformers-mlinter` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
