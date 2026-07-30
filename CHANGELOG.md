# Changelog

All notable changes to `transformers-mlinter` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `TRF020`, which enforces that Multi-head Latent Attention (MLA) models — those whose configuration declares
  `kv_lora_rank` — isolate the KV LoRA expansion (conventionally `kv_b_proj`, or any `nn.Linear(config.kv_lora_rank, ...)`)
  in a dedicated method (e.g. `expand_kv`) that `forward()` calls, rather than applying it inline inside `forward()`.
  This gives external backends (vLLM/SGLang) a single method to override so they can store and consume the compressed
  KV cache directly instead of materializing the full key/value states. The MLA gate reads companion
  `configuration_*.py` files to detect the `kv_lora_rank` field; models that intentionally deviate can suppress with
  `# trf-ignore: TRF020`.
- Added `TRF021`, which flags `torch.tensor(<scalar>, ..., device=<non-cpu>)` in `modeling_*.py` and `modular_*.py`.
  Building a 0-d tensor that way materializes the value on the host and then issues a host-to-device copy, which CUDA
  graph capture forbids; `torch.full((), <scalar>, dtype=..., device=...)` fills the same tensor directly on-device.
  The rule only fires when the value provably resolves to a Python scalar — numeric literals and arithmetic over them,
  `torch.finfo`/`torch.iinfo` fields, scalar builtins and `math.*` calls, locals bound exactly once, `self.<attr>`
  assigned in the class body, and `self.config.<field>` / `config.<field>` annotated `int`/`float`/`bool` in the
  companion `configuration_*.py`. Fields that may also be sequences (e.g. `eos_token_id: int | list[int] | None`) and
  unresolvable expressions are left alone, as are construction-time methods (`__init__`, `_init_weights`,
  `__post_init__`, `post_init`). Suppress with `# trf-ignore: TRF021`.
- Added `TRF022`, which flags `_no_split_modules` entries naming a class that does not exist in the model. A name is
  resolved against the classes defined in the `modeling_*.py` file, the names it imports, and the classes defined by
  sibling modules of the same model directory. Entries naming another model's
  classes are flagged: `post_init` already collects `_no_split_modules` from child submodels, so hardcoding them
  is redundant. Only `modeling_*.py` files are checked — modular files inherit most of their classes implicitly, so
  their references cannot be resolved statically. Complements `TRF005`, which only validates the shape of the value.
- Shared the companion-config resolution helpers (`_find_config_file`, `_parse_config_classes`,
  `_resolve_config_class_name_from_modeling_class`, `_resolve_target_config_class_name`) by moving them from `TRF015`
  into `mlinter/_helpers.py`, so cross-file rules resolve a modeling class to its target config class the same way.

## [0.1.2] - 2026-07-08

### Added

- Added `TRF016`, which flags `do_*` boolean flags declared on image/video processor classes that are not referenced
  by an overridden `preprocess` / `_preprocess` method.
- Expanded the set of files the linter targets to include `image_processing_*.py` and `video_processing_*.py` in
  addition to `modeling_*.py`, `modular_*.py`, and `configuration_*.py`. This affects file discovery for every rule,
  not just `TRF016`.
- Added `TRF017`, which flags model output classes decorated with both `@auto_docstring` and `@dataclass` where
  `@dataclass` is listed above `@auto_docstring`. Bottom-up decorator application means `@auto_docstring` then runs
  before `@dataclass` synthesizes `__init__`, and ends up modifying the parent class's `__init__.__doc__` instead of
  the subclass's. Mirrors the upstream fix in
  [huggingface/transformers#45702](https://github.com/huggingface/transformers/pull/45702).
- Added `TRF018`, which flags `_init_weights` overrides on `PreTrainedModel` subclasses that do not chain via
  `super()._init_weights(...)` (or the modular-file equivalent `<Class>._init_weights(self, ...)`). Models that
  intentionally fully override initialization can suppress with `# trf-ignore: TRF018`. Modular files using the
  `raise AttributeError(...)` delete-sentinel are skipped. See
  https://github.com/huggingface/transformers/pull/45597 for the bug class this catches.
- Added `TRF019`, which flags non-empty `_defaults` dictionaries on `*ProcessorKwargs` TypedDict classes in
  `processing_*.py` files for models contributed on or after the rule cutoff date. Processor defaults should live in
  `processor_config.json` on the Hub instead of being hardcoded in Python.
- Expanded the set of files the linter targets to include `processing_*.py` files in addition to modeling,
  configuration, modular, image-processing, and video-processing files.

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
