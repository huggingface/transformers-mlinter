#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run every TRF rule against a transformers checkout and emit a markdown report.

Invoked from the root of a transformers checkout. The CWD must contain
``src/transformers/models``. Outputs go to ``<output-dir>``, which defaults to
the parent of CWD (i.e. the GitHub Actions workspace when this is run from a
nested ``transformers/`` clone).

The report covers every rule, but only the default-enabled ones, after their
allowlists, are what Transformers sees when it adopts this version. Those are
the gate: ``--defaults-only`` writes them as JSON (run it with the base
branch's mlinter installed), and ``--baseline`` fails the scan on any that the
base run did not have. The two runs must scan the same transformers commit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import TypedDict

from mlinter import (
    DEFAULT_ENABLED_TRF_RULES,
    TRF_RULE_SPECS,
    TRF_RULES,
    Violation,
    __version__,
    analyze_file,
    iter_modeling_files,
)


def _rule_id_from_message(message: str) -> str:
    head = message.split(":", 1)[0].strip()
    return head if head.startswith("TRF") else "TRF???"


def _collect(enabled: set[str]) -> tuple[list[Path], list[Violation]]:
    files = list(iter_modeling_files())
    violations: list[Violation] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"  skip {path}: {exc}", file=sys.stderr)
            continue
        violations.extend(analyze_file(path, text, enabled_rules=enabled))
    return files, violations


class Finding(TypedDict):
    rule_id: str
    path: str
    line: int
    message: str


def _default_findings(violations: list[Violation]) -> list[Finding]:
    """The violations Transformers would see on adopting this version, as sortable records."""
    records = [
        Finding(rule_id=rule_id, path=str(v.file_path), line=v.line_number, message=v.message)
        for v in violations
        if (rule_id := v.rule_id or _rule_id_from_message(v.message)) in DEFAULT_ENABLED_TRF_RULES
    ]
    return sorted(records, key=lambda r: (r["path"], r["line"], r["rule_id"]))


def _finding_key(record: Finding) -> tuple[str, str, int]:
    # Both runs scan the same transformers commit, so a line number identifies the finding. The message
    # is left out: rewording a rule's message must not count as a new violation.
    return record["rule_id"], record["path"], record["line"]


def _new_findings(current: list[Finding], baseline: list[Finding]) -> list[Finding]:
    known = {_finding_key(record) for record in baseline}
    return [record for record in current if _finding_key(record) not in known]


def _render_gate(new: list[Finding], current_count: int, baseline_count: int) -> str:
    lines = ["## Gate: new violations under the default rules", ""]
    lines.append(
        f"Default-enabled rules, after allowlists: **{baseline_count}** on the base branch, "
        f"**{current_count}** with this change, **{len(new)}** new."
    )
    lines.append("")
    if not new:
        lines.append("No new violations. Transformers can adopt this change without new failures.")
        return "\n".join(lines) + "\n"
    lines.append(
        "Transformers' quality check would fail on these once it adopts this version. Fix the rule, or, "
        "if the existing code is acceptable, add the model to the rule's `allowlist_models` in `rules.toml`."
    )
    lines.append("")
    for record in new:
        lines.append(f"- `{record['path']}:{record['line']}` — {record['message']}")
    return "\n".join(lines) + "\n"


def _render_report(
    files: list[Path],
    violations: list[Violation],
    enabled: set[str],
    transformers_sha: str | None,
    transformers_ref: str | None,
) -> str:
    by_rule: defaultdict[str, list[Violation]] = defaultdict(list)
    by_file: Counter[str] = Counter()
    for v in violations:
        rule_id = _rule_id_from_message(v.message)
        by_rule[rule_id].append(v)
        by_file[str(v.file_path)] += 1

    lines: list[str] = ["# transformers-mlinter scan", ""]
    lines.append(f"- mlinter version: `{__version__}`")
    if transformers_ref:
        lines.append(f"- transformers ref: `{transformers_ref}`")
    if transformers_sha:
        lines.append(f"- transformers commit: `{transformers_sha}`")
    lines.append(f"- files scanned: **{len(files)}**")
    lines.append(f"- rules enabled: **{len(enabled)}**")
    lines.append(f"- total violations: **{len(violations)}**")
    lines.append("")

    if not violations:
        lines.append("No violations found.")
        return "\n".join(lines).rstrip() + "\n"

    lines.append("## Violations by rule")
    lines.append("")
    lines.append("| Rule | Description | Count |")
    lines.append("|------|-------------|------:|")
    for rule_id in sorted(by_rule, key=lambda r: (-len(by_rule[r]), r)):
        spec = TRF_RULE_SPECS.get(rule_id, {})
        description = spec.get("description", "—")
        lines.append(f"| `{rule_id}` | {description} | {len(by_rule[rule_id])} |")
    lines.append("")

    lines.append("## Top files by violation count")
    lines.append("")
    for path, count in by_file.most_common(20):
        lines.append(f"- `{path}` — {count}")
    lines.append("")

    lines.append("## Sample violations per rule")
    lines.append("")
    for rule_id in sorted(by_rule):
        bucket = by_rule[rule_id]
        lines.append(f"### {rule_id} ({len(bucket)})")
        for v in bucket[:3]:
            lines.append(f"- `{v.file_path}:{v.line_number}` — {v.message}")
        if len(bucket) > 3:
            lines.append(f"- _(+{len(bucket) - 3} more)_")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write scan-report.md and scan-violations.txt (default: parent of CWD).",
    )
    parser.add_argument(
        "--transformers-sha",
        default=os.environ.get("TRANSFORMERS_SHA"),
        help="Commit SHA of the scanned transformers checkout (for the report).",
    )
    parser.add_argument(
        "--transformers-ref",
        default=os.environ.get("TRANSFORMERS_REF"),
        help="Git ref of the scanned transformers checkout (for the report).",
    )
    parser.add_argument(
        "--defaults-only",
        type=Path,
        default=None,
        metavar="PATH",
        help="Only write the default-enabled rules' violations to PATH as JSON; no report.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        metavar="PATH",
        help="JSON from a --defaults-only run on the base branch; exit 1 on any default-rule violation it lacks.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    enabled = set(TRF_RULES)
    print(f"Scanning with {len(enabled)} TRF rule(s)...", file=sys.stderr)

    files, violations = _collect(enabled)
    print(f"Discovered {len(files)} file(s); {len(violations)} violation(s).", file=sys.stderr)
    defaults = _default_findings(violations)

    if args.defaults_only is not None:
        args.defaults_only.write_text(json.dumps(defaults, indent=1) + "\n", encoding="utf-8")
        print(f"Wrote {len(defaults)} default-rule violation(s) to {args.defaults_only}.", file=sys.stderr)
        return 0

    output_dir = (args.output_dir or Path.cwd().parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = _render_report(files, violations, enabled, args.transformers_sha, args.transformers_ref)
    new: list[Finding] = []
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        new = _new_findings(defaults, baseline)
        report = _render_gate(new, len(defaults), len(baseline)) + "\n" + report

    (output_dir / "scan-report.md").write_text(report, encoding="utf-8")
    with (output_dir / "scan-violations.txt").open("w", encoding="utf-8") as f:
        for v in sorted(violations, key=lambda x: (str(x.file_path), x.line_number, x.message)):
            f.write(f"{v.file_path}:{v.line_number}: {v.message}\n")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(report)

    sys.stdout.write(report)
    if new:
        print(f"{len(new)} new default-rule violation(s); see the Gate section above.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
