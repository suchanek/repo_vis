# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-09-24

### Added

- Container image `egsuchanek/pkg-visualizer` for `linux/amd64` and
  `linux/arm64`. By default it runs the Qt GUI on a virtual display served
  over noVNC on port 6080; with `--headless` it writes one file and exits.
- `Makefile` with `build`, `gui`, `export`, `logs`, `stop` and `push`
  targets for Docker (`RUNTIME=docker`, the default) and Apple's `container`
  CLI (`RUNTIME=apple`), running natively on Apple Silicon.
- `docker/docker-compose.yml` for the GUI under Docker.
- `--headless` renders off-screen and writes `.html`, `.png` or `.jpg`,
  chosen by the `--save_path` suffix, without opening a window.
- `--save_elements` (`-e`) writes the parsed classes, methods and functions
  to JSON, in GUI and headless mode.
- `repovis` command, an alias for `pkg-visualizer`.
- `CITATION.cff`, and README badges, banner, Citation and License sections.
- `CHANGELOG.md`.

### Changed

- `pyproject.toml` uses the PEP 621 `[project]` table with the SPDX license
  `GPL-3.0-only` and the `poetry-core>=2` backend, so `pip install` from the
  repository works.
- Requires pyvista 0.49, VTK 9.5 to 9.7 and pyvistaqt 0.13. VTK publishes
  Linux arm64 wheels only from 9.5.
- The scene title separates fields with `-` instead of `|`, which the VTK 9.7
  font does not render.
- `--save_path` fills the GUI **Save Path** box instead of being replaced by
  the package name on load.

### Removed

- The unused `panel` dependency, and with it bokeh, pandas and the Jupyter
  server stack.

### Fixed

- **Save View** to HTML failed in the GUI under pyvista 0.49, whose lazy
  exporter lookup does not reach pyvistaqt's `QtInteractor`.
- `markdown` is declared as a dependency; it was imported but only arrived
  through `panel`.
- `__version__` reported 0.1.0 in the 0.1.1 release.

## [0.1.1] - 2025-07-12

See [RELEASE_NOTES_v0.1.1.md](RELEASE_NOTES_v0.1.1.md).

[Unreleased]: https://github.com/suchanek/repo_vis/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/suchanek/repo_vis/compare/V0.1.1...v0.2.0
[0.1.1]: https://github.com/suchanek/repo_vis/releases/tag/V0.1.1
