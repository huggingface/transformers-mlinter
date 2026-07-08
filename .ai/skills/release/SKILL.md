---
name: release
description: Cut a new transformers-mlinter release. Bumps the version across all pinned locations, updates the CHANGELOG, runs local checks and a packaged smoke test, then drives the tag-based GitHub Actions publish to PyPI. Use when asked to release, cut a version, or ship X.Y.Z.
---

# Release transformers-mlinter

Cuts a versioned release. The canonical prose reference is `docs/release.md`; this
skill is the operational checklist. Publishing itself is done by CI — pushing the
`vX.Y.Z` tag triggers `.github/workflows/release.yml`, which builds, tests, and
publishes to PyPI via OIDC trusted publishing (no local upload).

## Input

- `<version>`: the target release version `X.Y.Z` (e.g. `0.1.2`). A pre-release
  uses `X.Y.ZrcN`.

## Constraints

- Semantic versioning. Patch releases (`Z` bumps) reuse the existing `vX.Y-release`
  branch; a new minor/major starts a fresh `vX.Y-release` branch.
- Work from a clean checkout on `main`. Never publish uncommitted changes.
- Do NOT run `twine upload` locally — publishing is CI-only via the pushed tag.
- Always confirm with the user before pushing the tag (the tag push is the
  irreversible, publish-triggering step).

## The version lives in one place

`pyproject.toml` `[project].version` is the **single source of truth**. At runtime
`mlinter/_version.py` reads the version from the installed distribution metadata,
falling back to `pyproject.toml`; nothing else hardcodes the release version. So a
release bumps exactly one version field plus the CHANGELOG:

1. `pyproject.toml` — `[project].version` (the only version edit)
2. `CHANGELOG.md` — a new `## [X.Y.Z] - YYYY-MM-DD` section (see step 2)

Do NOT reintroduce hardcoded version strings elsewhere. `DEFAULT_BASE_VERSION` in
`mlinter/_version.py` is a static `0.0.0` sentinel and must stay unbumped; the
`README.md` example suffix and the `tests/test_mlinter.py` version pins use fixed
illustrative values (`1.2.3+g1a2b3c4`, `9.9.9`) decoupled from the real release.

## Workflow

1. **Preflight.** Confirm the branch is `main`, the tree is clean, and it is up to
   date (`git status`, `git fetch && git status`). Confirm the target `<version>` is
   greater than the latest tag (`git tag`).

2. **Update `CHANGELOG.md`.** Add a `## [X.Y.Z] - YYYY-MM-DD` section (today's date)
   above the previous release, following Keep a Changelog with `### Added` /
   `### Fixed` / `### Changed` subsections. Summarize what landed since the last tag
   — `git log vPREV..HEAD --oneline` is the source. Each new `TRFNNN` rule and any
   change to file-discovery globs gets an entry.

3. **Bump the version** — set `[project].version` in `pyproject.toml` to `<version>`.
   That is the only version edit. For a release that added new `TRFNNN` rules, also
   add the corresponding public-API assertions in `tests/test_mlinter.py`
   (`public_api.TRFNNN == "TRFNNN"` and `"TRFNNN"` in `public_api.__all__`) — mirror
   the existing pattern.

4. **Run local checks.** All three must pass:
   ```bash
   make lint
   make test
   make typecheck
   ```

5. **Build and validate the package.**
   ```bash
   make build-release
   python -m twine check --strict dist/*
   ```
   Then smoke-test the wheel in a throwaway venv (adjust the version string):
   ```bash
   python -m venv /tmp/mlinter-release && source /tmp/mlinter-release/bin/activate
   pip install dist/*.whl
   cd /tmp && mlinter --version && python -m mlinter --version
   python -c "import mlinter; print(mlinter.__version__)"
   deactivate
   ```
   `mlinter --version` must print `X.Y.Z` (no `+g...` suffix from an installed wheel).

6. **Commit the bump on `main`.** Match the existing message style:
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "preparing for X.Y.Z"
   git push origin main
   ```
   Add any other files the release genuinely touched (new rule modules, tests, etc.).

7. **Update the release branch.** For a new minor line:
   ```bash
   git switch -c vX.Y-release
   git push -u origin vX.Y-release
   ```
   For a patch on an existing line, fast-forward the branch from `main`:
   ```bash
   git switch vX.Y-release && git merge --ff-only main && git push origin vX.Y-release
   ```
   Pushing the branch runs the `Release` workflow's build+test job (no publish yet).

8. **Tag and publish.** Confirm with the user first, then from `vX.Y-release`:
   ```bash
   git tag vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```
   The tag push triggers the build job again followed by the `upload_package` job,
   which runs in the `pypi-release` environment (may require reviewer approval) and
   publishes via `pypa/gh-action-pypi-publish` with OIDC. Watch the run with
   `gh run watch` or `gh run list --workflow=release.yml`.

9. **Verify from PyPI** once the publish job succeeds:
   ```bash
   python -m pip install -U "transformers-mlinter==X.Y.Z"
   mlinter --version
   ```

## Notes

- No `.dev0` post-release bump is used; releasing is a single bump + tag.
- First-ever PyPI publish requires a pending trusted publisher configured on PyPI
  pointing at `release.yml` with environment `pypi-release` — see `docs/release.md`.
- Optional TestPyPI dry run is documented in `docs/release.md` and is manual.
