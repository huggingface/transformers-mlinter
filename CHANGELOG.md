# Changelog

All notable changes to `transformers-mlinter` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `TRF038`, which checks that every `modeling_*.py`, `processing_*.py`, `image_processing_*.py`,
  `video_processing_*.py` and `feature_extraction_*.py` file has a matching `tests/models/<model>/test_*.py` file
  (e.g. `modeling_acme.py` -> `tests/models/acme/test_modeling_acme.py`). `configuration_*.py` is exempt, since
  config classes are conventionally exercised through `ConfigTester` inside `test_modeling_*.py`.
  `modular_*.py` files are handled by inspecting the classes they define rather than the filename, since one
  modular file can mix modeling, processing, image/video-processor and config classes. This rule has
  no `# trf-ignore: TRF038` suppression: every model can ship at least a minimal test built on a dummy config and
  randomly initialized weights, so exemptions must go through `allowlist_models` instead, where they are visible in
  review.
- Added `TRF039`, which flags imports inside `if is_*_available(): ...` guards (e.g.
  `if is_vision_available(): from PIL import Image`) that are never referenced anywhere else in the file. `ruff`
  does not clean these up on its own, so a leftover import from a refactor silently lingers in `src/transformers`.
  Suppress with `# trf-ignore: TRF039` for genuine false positives (e.g. names only used dynamically).
- Added `TRF040`, which flags methods in `modeling_*.py` / `modular_*.py` decorated with both `@capture_outputs` and
  `@can_return_tuple`. Both decorators pop `return_dict`, so only the outermost one sees the value the caller actually
  passed while the inner one silently falls back to `self.config.return_dict`. `@capture_outputs` already handles the
  `to_tuple` conversion, which makes `@can_return_tuple` redundant. Complements `TRF003`, which covers manual `return_dict`
  branching. Suppress with `# trf-ignore: TRF040`.
- Added `TRF041`, which requires a `# CODEPATH:` comment on every `if`/`elif` statement and conditional
  expression in `modeling_*.py`/`modular_*.py` whose condition reads a `config.*` or `self.config.*` attribute.
  The comment is accepted on the branch line or anywhere in the contiguous comment block above it, so it can head a
  multi-line explanation. Modelled on Rust's `// SAFETY:` convention: the branch stays legal, but the author has to
  write down which checkpoints take which path. Deliberately broad — a branch on a numeric or optional config field
  forks the graph exactly as much as one on a boolean flag, and the library has 1 838 such branches across 330 models
  today. `cutoff_date` grandfathers all of them; the eleven post-cutoff models are allowlisted in the TOML.
  Default coalescing is exempt by shape, not by name: `X if X is not None else fallback`, where the tested field is
  itself one of the results, is `getattr(config, x, default)` spelled long and cannot fork the graph, so it needs no
  note (79 of the 2 674 firings in the library today). Mentioning None is not enough to qualify —
  `config.vision_config is not None` gates a whole extra tower and still has to explain itself.
  Fields that gate no checkpoint divergence — `problem_type` picking a loss, `hidden_act` picking an activation —
  can be exempted for a whole file with a module-level `# trf-ignore: TRF041 config.problem_type, config.hidden_act`
  directive, instead of repeating a per-branch suppression. `self.config.x`, `config.x` and `x` all name the same
  field, and the directive has to name at least one, so a bare `# trf-ignore: TRF041` still means only its own line.
  Exemption is per field: a branch reading several config fields is skipped only when every one of them is exempt.
- Added `TRF042`, which requires a `test_tokenization_*.py` file to define a test class inheriting
  `TokenizerTesterMixin`. `TokenizerTesterMixin` is where encode/decode round-tripping, padding and truncation,
  special-token handling and save/load equivalence are actually checked, so a file that only asserts a couple of
  hand-written id lists looks tested while the tokenizer is broken in every one of those dimensions. Files whose only
  classes are helpers are skipped, and inheritance is followed through local base classes and into another model's
  tokenizer test — `DistilBertTokenizationTest(test_tokenization_bert.BertTokenizationTest)` counts as satisfied
  because the class it derives from carries the mixin. A base the tests tree cannot resolve never counts. Five of the
  six tokenizer tests missing the mixin predate 2026 and are grandfathered by `cutoff_date`; `auto` is allowlisted
  because `test_tokenization_auto.py` tests `AutoTokenizer` resolution rather than one model's tokenizer.
- **Widened file discovery to the tests tree.** `iter_modeling_files` now also walks
  `tests/models/**/test_tokenization_*.py` via a new shared `TESTS_ROOT`, `_model_dir_name` resolves a model name from
  either root, and `--changed-only` accepts those paths. This changes which files the linter walks for *every* rule, so
  it is worth noting even though no existing rule is affected: they all gate on the file-name prefix, and a full scan
  confirms none of `TRF001`-`TRF041` fires on a test file. File count on a current checkout goes from 1 132 to 1 222.
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
- Added `TRF027`, which flags bare `assert` in `modeling_*.py`, `modular_*.py` and `configuration_*.py`. `python -O`
  strips asserts, so a shape or config check written that way silently disappears, and an `AssertionError` tells the
  user nothing actionable.
- Added `TRF028`, which requires a complete license header in the first 25 lines of `modeling_*.py`,
  `modular_*.py`, `configuration_*.py`, `processing_*.py`, `image_processing_*.py` and `video_processing_*.py`.
  Every clause of the warranty paragraph is matched, not just the words `Apache License`, because that is what
  the real defects look like: `bitnet` drops the closing `limitations under the License.`, `tvp` and
  `bridgetower` carry a stray `=` before every comma from a bad search-and-replace, and `minimax_m3_vl` stops
  after the URL. The license name is not checked — `blip` is BSD-3-clause and `sapiens2` uses Meta's own
  license — so a deliberate license choice is not reported.
- Added `TRF029`, which flags an `__init__` accepting `config` alongside an argument that is unambiguously a config
  field (`hidden_size`, `num_attention_heads`, `intermediate_size`, `head_dim`, `embed_dim`, `dropout`, `eps`,
  `patch_size`, `rope_theta`, ...). The value then has two sources of truth and the caller decides which wins.
  `kosmos2` is allowlisted because its doc page (`kosmos-2.md`) is not derivable from the directory name.
- Added `TRF030`, which flags attribute chains rooted at `config`/`self.config` that go three or more levels deep.
  One hop (`config.hidden_size`) and two (`config.text_config.hidden_size`) are the normal sub-config accesses;
  deeper means the module should have been handed a sub-config. Reported once per line.
- Added `TRF031`, which flags a top-level `@dataclass` in a modeling file whose bases carry no `Output` name. A plain
  dataclass does not index like a tuple, does not survive `return_dict=False`, and is invisible to `@auto_docstring`.
  Any `BaseModelOutputWith*` base counts as satisfying the rule.
- Added `TRF032`, which flags `masked_fill`, `masked_fill_`, `full`, `full_like` and `new_full` called with a negated
  literal of magnitude 1e3 or more. A hardcoded `-1e9` overflows to `-inf` in float16 and is not the float32 minimum,
  so the mask behaves differently per dtype; `torch.finfo(dtype).min` is correct in all of them.
- Added `TRF033`, which flags `set_*` methods other than the `PreTrainedModel` contract ones
  (`set_input_embeddings`, `set_output_embeddings`, `set_decoder`, `set_encoder`, `set_attn_implementation`,
  `set_default_language`). A hyperparameter behind a setter is not in the config, so it is not saved, not restored by
  `from_pretrained`, and invisible to device-map and parallelism planning.
- Added `TRF034`, which flags a locally-defined class ending in `Layer`/`Block`, instantiated inside an
  `nn.ModuleList(...)`, that does not reach `GradientCheckpointingLayer` through its local base chain.
  `gradient_checkpointing_enable()` skips plain `nn.Module` layers silently, so training appears to checkpoint and
  still allocates full activations. ModuleLists of experts, heads or projections are out of scope. Ten models are
  allowlisted; the list is in the TOML.
- Added `TRF035`, which flags `# noqa` in `modeling_*.py`, `modular_*.py` and `configuration_*.py`, reporting the
  suppressed codes when they are given. Three models are allowlisted.
- Added `TRF036`, which flags `nn.Sequential(...)` in modeling files. Sequential names its children by position, so
  weights land at `mlp.0.weight`, and every conversion mapping and parallelism plan has to reference indices.
  `x_clip` is allowlisted.
- Added `TRF037`, which flags `einsum` in modeling files and reports the equation when it is a literal.
  **Disabled by default** — einsum is occasionally the clearest way to write a contraction, so this is opt-in via
  `--enable-rules TRF037` rather than a hard convention. `x_clip` is allowlisted.

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
