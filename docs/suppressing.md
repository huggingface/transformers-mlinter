---
layout: default
title: Suppressing rules
nav_order: 4
description: "How to silence a single mlinter finding with trf-ignore, when a whole-file directive applies, and how cutoff dates and model allowlists differ from suppressions."
---

# Suppressing rules
{: .no_toc }

## On this page
{: .no_toc .text-delta }

- TOC
{:toc}

---

There are three separate mechanisms for making a rule not fire, and they exist for different reasons.
Reaching for the wrong one is how a convention quietly stops being enforced.

## Per-line: `# trf-ignore`

The everyday escape hatch. Put the comment on the flagged line, or on the line directly above the
flagged construct:

```python
class AcmePreTrainedModel(PreTrainedModel):
    base_model_prefix = ""  # trf-ignore: TRF002
```

```python
# trf-ignore: TRF018
def _init_weights(self, module):
    ...
```

The comment may also sit **above the decorators** rather than squeezed between the last decorator and
the `def`. mlinter walks upward past decorator lines looking for it, so this works:

```python
# trf-ignore: TRF018
@torch.no_grad()
def _init_weights(self, module):
    ...
```

The search stops at the first line that is neither a decorator nor the directive, so a suppression
can never leak onto a construct further down the file.

{: .note }
> The rule id is matched case-insensitively, but write it uppercase — that is what every existing
> suppression in the library does, and it is what a reader greps for.

## Per-file, per-subject: module-level directives

A few rules flag one construct repeatedly for the same reason. Suppressing each occurrence would mean
repeating an identical comment a dozen times in one file, which is noise that reviewers learn to skip.

Those rules honour a directive at **column 0** naming the subjects to exempt for the whole file:

```python
# trf-ignore: TRF041 problem_type, hidden_act
```

`TRF041` uses this for config fields that gate the same branch in every model — `problem_type` selects
a loss, `hidden_act` looks up an activation — so no checkpoint diverges on them.

Two properties are worth knowing:

- The directive must **name at least one subject**. A bare `# trf-ignore: TRF041` at column 0 is
  treated as an ordinary per-line suppression, so a subject-less directive can never silently widen
  into a whole-file mute.
- Parsing stops at the first word that is not an identifier path, so you can add trailing prose
  explaining why without it being read as another subject.

Not every rule supports this. Check the rule's entry in
[`mlinter/rules.toml`](https://github.com/huggingface/transformers-mlinter/blob/main/mlinter/rules.toml)
before using it — and note that some rules deliberately support no suppression at all, because every
file in scope can satisfy them.

## Repo-wide: cutoff dates and model allowlists

These are not suppressions. They are declarations, in `rules.toml`, about a rule's scope — and they are
the maintainers' tool, not a contributor's.

**Cutoff dates.** A rule that encodes a convention introduced at a point in time carries a
`cutoff_date`. Models contributed to Transformers before that date are grandfathered automatically,
read from the contribution date on the model's doc page. This is what keeps a new rule from having to
ship with a 300-model allowlist. A model whose doc page has no contribution date **is** checked, so a
missing date never silently disables a rule.

**Model allowlists.** Individual models that predate a convention and cannot be fixed without breaking
backward compatibility are listed by name in `allowlist_models`. Each rule page on this site lists its
own allowlist.

Both appear in the **Scope** row of every [rule page](rules/index.md).

## Choosing between them

| Situation | Use |
|:----------|:----|
| One line in one model is a justified exception | `# trf-ignore` on that line |
| The same construct recurs all over one file for one reason, and the rule supports subjects | Module-level directive naming the subjects |
| An existing model cannot comply without a breaking change | `allowlist_models` in `rules.toml` |
| A brand-new convention that older models were never written against | `cutoff_date` in `rules.toml` |

If none of these fit, the rule itself is probably wrong for the case — that is worth an issue rather
than a suppression.
