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
- Added `TRF022`, which flags `_no_split_modules` entries naming a class that does not exist in the model. Names are
  resolved against the classes defined or imported in the `modeling_*.py` / `modular_*.py` file and those defined by
  sibling modules of the same model directory. Entries naming another model's classes are flagged too: `post_init`
  already collects `_no_split_modules` from child submodels. Complements `TRF005`, which only validates the shape of
  the value. Suppress with `# trf-ignore: TRF022`.
- Added `TRF023`, which flags config fields declared under an upstream paper's abbreviation instead of the library's
  canonical name: `d_model`/`n_embd` (→ `hidden_size`), `d_ff`/`d_inner`/`ffn_dim`/`ffn_hidden_size`/`expansion_ratio`
  (→ `intermediate_size`), `d_head` (→ `head_dim`), `n_head`/`n_heads` (→ `num_attention_heads`),
  `n_layer`/`n_layers`/`num_blocks` (→ `num_hidden_layers`). Fields are collected from the class body and from
  `__init__`/`__post_init__` assignments and signature defaults, and each legacy name is reported once per class.
  Names that remain idiomatic in parts of the library (`num_heads`, `num_layers`, `embed_dim`, `mlp_ratio`) are
  deliberately not flagged. `cutoff_date` grandfathers the ~78 models that predate the convention; `kosmos2` and
  `openai` are allowlisted because their doc pages (`kosmos-2.md`, `openai-gpt.md`) cannot be derived from the
  directory name, and `qwen3_asr` because its encoder config mirrors Whisper's public `d_model`.
- Added `TRF024`, which flags `torch.nn` layer constructors built with an integer literal greater than 8 in a
  dimension position — positionally or by keyword — in `modeling_*.py` and `modular_*.py`. Covers `Linear`,
  `LazyLinear`, `Bilinear`, `Embedding`, `EmbeddingBag`, `LayerNorm`, `RMSNorm`, `GroupNorm`, `BatchNorm*`,
  `InstanceNorm*`, `Conv*d`, `ConvTranspose*d` and `MultiheadAttention`. Operator-shape arguments (`kernel_size`,
  `stride`, `padding`, `num_groups`) are ignored and literals up to 8 are allowed, so scalar heads, binary
  classifiers and RGB channel counts stay clean. A hardcoded width pins the module to one checkpoint size and splits
  the source of truth away from the config.
- Added `TRF025`, which flags mask factories (`create_causal_mask`, `create_bidirectional_mask`,
  `create_sliding_window_causal_mask`, `create_chunked_causal_mask`, `create_masks_for_generate`, and any
  `create_*_mask` helper) called from a class whose name ends in `Layer`, `Attention` or `Block`. Mask construction
  does not vary per layer, so building it inside the layer repeats quadratic work and leaves each layer owning its
  own mask. Models and encoders that build the mask once and pass it down are not in scope.
- Added `TRF026`, which flags a non-`PreTrainedModel` class that defines only `__init__` and `forward`, assigns
  exactly one `self.<attr>` in `__init__`, and whose `forward` body is exactly `return self.<attr>(...)` for that
  attribute. The wrapper adds a level to every weight name, to `_no_split_modules`, to the parallelism plans and to
  every conversion mapping while contributing no computation. `PreTrainedModel` subclasses are exempt because they
  exist for `from_pretrained` and the auto classes even when the forward only delegates.
- Shared the companion-config resolution helpers (`_find_config_file`, `_parse_config_classes`,
  `_resolve_config_class_name_from_modeling_class`, `_resolve_target_config_class_name`) by moving them from `TRF015`
  into `mlinter/_helpers.py`, so cross-file rules resolve a modeling class to its target config class the same way.
- Moved `model_contribution_date` (and `DOCS_ROOT`) from `TRF019` into `mlinter/_helpers.py` and added
  `is_exempt_by_cutoff`, so every cutoff-gated rule resolves a model's contribution date the same way. The lookup now
  also tries the hyphenated spelling of the model directory (`blenderbot_small` → `blenderbot-small.md`), which
  grandfathers models whose doc page uses hyphens instead of leaving them permanently unexempt.

### Fixed

- Removed a stale `# type: ignore[union-attr]` in `TRF019` that `ty` reported as an unused suppression, so
  `make typecheck` is clean.

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
