# Changelog

All notable changes for the latest release are documented here. Historical entries have been preserved in docs/DEPRECATED-CHANGELOG.md.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v2.0.0] - 2025-11-17

### Added
- CI workflow flag to optionally test CUDA wheel installation in an isolated virtual environment (workflow_dispatch: gpu=true). The job installs torch/torchvision CUDA 12.8 wheels, prints build/runtime info, installs the package in editable mode, and runs tests.

### Changed
- Project version bumped to v2.0.0 across sources and packaging metadata.
- README versioning section updated to explicitly use semantic versioning and v2.0.0 example.
- Tests modernized to be resilient in CPU-only environments, skipping CUDA extension checks when unavailable.
- Cleaned and standardized CI: non-isolated builds where appropriate, consistent virtualenv usage in jobs.

### Fixed
- Addressed CI build fragility by ensuring PyTorch is not required at build-system level and by using --no-build-isolation for builds in CI, preventing mismatches between build-time and runtime environments.
- Ensured missing module issues (e.g., datasets init) are resolved; import paths validated by tests.

### Documentation
- Legacy changelog moved to docs/DEPRECATED-CHANGELOG.md to declutter the current CHANGELOG.
- Updated auxiliary docs and summaries to reflect v2.0.0.

### Notes
- CUDA extension is optional; tests skip gracefully when not present. Dedicated GPU CI performs a runtime CUDA smoke test in its own venv.
