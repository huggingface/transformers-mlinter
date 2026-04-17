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

import json
import subprocess
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 fallback


PACKAGE_NAME = "transformers-mlinter"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_version_from_pyproject() -> str:
    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return data["project"]["version"]


def _installed_distribution():
    try:
        return distribution(PACKAGE_NAME)
    except PackageNotFoundError:
        return None


def _short_commit_id(commit_id: str | None) -> str | None:
    if commit_id is None:
        return None

    commit_id = commit_id.strip()
    if not commit_id:
        return None
    return commit_id[:7]


def _read_git_hash_from_direct_url(dist) -> str | None:
    if dist is None:
        return None

    direct_url = dist.read_text("direct_url.json")
    if not direct_url:
        return None

    try:
        direct_url_data = json.loads(direct_url)
    except json.JSONDecodeError:
        return None

    vcs_info = direct_url_data.get("vcs_info")
    if not isinstance(vcs_info, dict) or vcs_info.get("vcs") != "git":
        return None

    commit_id = vcs_info.get("commit_id")
    if not isinstance(commit_id, str):
        return None
    return _short_commit_id(commit_id)


def _read_git_hash_from_checkout() -> str | None:
    if not (PROJECT_ROOT / ".git").exists():
        return None

    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None

    if result.returncode != 0:
        return None
    return _short_commit_id(result.stdout)


def _append_git_hash(base_version: str, git_hash: str | None) -> str:
    if git_hash is None or "+" in base_version:
        return base_version
    return f"{base_version}+g{git_hash}"


def _resolve_version() -> str:
    dist = _installed_distribution()
    base_version = dist.version if dist is not None else _read_version_from_pyproject()
    git_hash = _read_git_hash_from_direct_url(dist) or _read_git_hash_from_checkout()
    return _append_git_hash(base_version, git_hash)


__version__ = _resolve_version()
