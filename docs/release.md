# Releasing `transformers-mlinter`

This repository uses a tag-driven GitHub Actions release workflow similar to `transformers`.

Release branches named `vX.Y-release` build and validate the release artifacts. A pushed tag named `vX.Y.Z` or
`vX.Y.ZrcN` triggers the same build, then publishes the validated artifacts to PyPI through GitHub OIDC trusted
publishing.

No post-release `.dev0` bump is required right now. The release process is a single version bump, validation pass,
release branch push, and tag push.

## Prerequisites

- PyPI maintainer access for `transformers-mlinter`
- A PyPI trusted publisher configured for this GitHub repository and the release workflow
- A GitHub environment named `pypi-release`, ideally protected with required reviewers
- A clean checkout on `main`
- Python 3.10+ with the project dev dependencies installed
- Packaging tools installed locally:

```bash
python -m pip install --upgrade build twine
pip install -e ".[dev]"
```

## First release setup

If `transformers-mlinter` does not exist on PyPI yet, the first release still works with trusted publishing.

Before pushing the first release tag:

1. In PyPI, go to your account-level publishing settings, not a project settings page.
2. Add a GitHub Actions trusted publisher for this repository as a pending publisher.
3. Point it at the `release.yml` workflow in `.github/workflows/`.
4. Set the environment name to `pypi-release` so it matches the workflow.

When the first `vX.Y.Z` tag runs successfully, PyPI creates the project and converts that pending publisher into a
normal project publisher for subsequent releases.

## 1. Update the version

The base version is currently stored in two places:

- `pyproject.toml` under `[project].version`
- `mlinter/_version.py` in `DEFAULT_BASE_VERSION`

Update `CHANGELOG.md` for the target release before tagging it.

Search for version-specific tests before cutting the release:

```bash
rg -n "0\.1\.1|DEFAULT_BASE_VERSION" pyproject.toml mlinter tests README.md CHANGELOG.md
```

Update any tests or examples that intentionally pin the released version string.

## 2. Run the local checks

From the repository root:

```bash
make lint
make test
make typecheck
```

## 3. Build the release artifacts locally

Clean old artifacts, then build both the wheel and source distribution:

```bash
make build-release
```

This should create files under `dist/`.

## 4. Validate the built package

First, ask `twine` to validate the package metadata:

```bash
python -m twine check --strict dist/*
```

Then install the wheel in a fresh virtual environment and verify that the CLI works:

```bash
python -m venv /tmp/transformers-mlinter-release
source /tmp/transformers-mlinter-release/bin/activate
pip install dist/*.whl
cd /tmp
mlinter --version
python -m mlinter --version
python -c "import mlinter; print(mlinter.__version__)"
deactivate
```

## 5. Create the release branch

Create the release branch for a new minor line:

```bash
git switch -c vX.Y-release
```

For patch releases, update the existing `vX.Y-release` branch from `main` with a fast-forward merge:

```bash
git switch vX.Y-release
git merge --ff-only main
```

## 6. Optional TestPyPI smoke test

If you want a dry run before publishing to the real index:

```bash
python -m twine upload --repository testpypi dist/*
```

Then install the uploaded package from TestPyPI:

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "transformers-mlinter==X.Y.Z"
```

This step remains manual for now and is not part of the GitHub Actions workflow.

## 7. Commit the release on the release branch

Once the version bump and validation are done, commit the release on `vX.Y-release` and push that branch:

```bash
git add CHANGELOG.md pyproject.toml mlinter/_version.py tests/test_mlinter.py README.md
git commit -m "Release X.Y.Z"
git push -u origin vX.Y-release
```

The GitHub Actions `Release` workflow will build the distributions, run the tests, reinstall the built wheel, and run
`twine check` on that branch push.

Adjust the `git add` list if the release touched other files.

## 8. Tag the release and publish through trusted publishing

From `vX.Y-release`, create the annotated tag and push it:

```bash
git tag vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

Pushing the `vX.Y.Z` tag triggers the same build job and then the publish job. The publish job runs in the
`pypi-release` GitHub environment and uses `pypa/gh-action-pypi-publish` with `id-token: write`, so no long-lived PyPI
API token is stored in GitHub.

After the environment approval and publish job succeed, verify the package from PyPI:

```bash
python -m pip install -U "transformers-mlinter==X.Y.Z"
mlinter --version
```
