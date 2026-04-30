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
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from mlinter import (
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = (args.output_dir or Path.cwd().parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    enabled = set(TRF_RULES)
    print(f"Scanning with {len(enabled)} TRF rule(s)...", file=sys.stderr)

    files, violations = _collect(enabled)
    print(f"Discovered {len(files)} file(s); {len(violations)} violation(s).", file=sys.stderr)

    report = _render_report(files, violations, enabled, args.transformers_sha, args.transformers_ref)

    (output_dir / "scan-report.md").write_text(report, encoding="utf-8")
    with (output_dir / "scan-violations.txt").open("w", encoding="utf-8") as f:
        for v in sorted(violations, key=lambda x: (str(x.file_path), x.line_number, x.message)):
            f.write(f"{v.file_path}:{v.line_number}: {v.message}\n")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(report)

    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
