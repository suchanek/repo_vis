# Release Notes - v0.2.0

> Released: 2026-09-24

pkg_visualizer now runs in a container as well as directly on your machine. The image `egsuchanek/pkg-visualizer` serves the full Qt GUI to a browser through noVNC, or renders a visualization to a file and exits. It is built for both `linux/amd64` and `linux/arm64`, so it runs natively on Apple Silicon under Docker or Apple's `container` tool, without Rosetta.

## What changed

**Headless export.** The new `--headless` flag builds the scene off-screen and writes an HTML, PNG or JPG file, picked by the `--save_path` suffix. It works on a Mac directly, and in the container on machines with no display. `--save_elements` writes the parsed package structure to JSON in either mode.

**Containers.** A `Makefile` drives both runtimes with the same targets: `make gui` opens the GUI at `http://localhost:6080`, and `make export` writes a file into `./out`. Add `RUNTIME=apple` to use Apple's `container` CLI instead of Docker. The noVNC port is bound to `127.0.0.1` because the VNC server has no password.

**Newer rendering stack.** The arm64 image needed VTK 9.5 or later, which in turn needed pyvista 0.49 and pyvistaqt 0.13. The native install moves to the same versions. Along the way, HTML export from the GUI's **Save View** button was fixed for the new pyvista, and the scene title now uses `-` as its separator because the new VTK font has no `|` glyph.

**Packaging.** `pyproject.toml` now uses standard PEP 621 metadata, so `pip install git+https://github.com/suchanek/repo_vis` works alongside Poetry. The unused `panel` dependency is gone, which drops bokeh, pandas and the Jupyter server stack from every install.

**Fixes.** `--save_path` now fills the GUI's Save Path box instead of being overwritten when the package loads.

## Upgrading

With Poetry, run `poetry install` to pick up the new dependency versions. With pip, reinstall from the repository. Nothing about the command line changes except the new flags. To use the container, run `docker pull egsuchanek/pkg-visualizer` or `make build`, and see the Containers section of the README.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
