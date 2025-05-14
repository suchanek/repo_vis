# repovis

`repovis` is a Python-based application that visualizes the structure of Python repositories in 3D. It provides an interactive and dynamic way to explore the Classes, Methods, and Functions within a repository. Built using Pyvista and pyqt5, the tool is designed for developers and researchers who want to gain insights into codebases visually. The ``examples`` directory has some exported .html files representing some of the most significant Python repositories currently.

## Features

- **Dynamic Visualization**:
  - Automatically adjusts the size of visual elements (e.g., classes and methods) based on the repository size.
  - Uses cubes for functions when their count exceeds 1000 for better performance and clarity.

- **Interactive UI**:
  - Includes sliders to control class and method radii.
  - "Reset Settings" button to restore default slider values and camera position.
  - Selecting a class or method will highlight the object on the screen and pop up the docstring.

- **Customizable Appearance**:
  - Supports dynamic scaling of class and method objects using `class_object_radius` and `method_object_radius`.
  - System theme-aware. The program will track light/dark mode system settings automatically

- **Code Insights**:
  - Displays the hierarchy and relationships between classes, methods, and functions in a repository.

## Recent Enhancements

1. **Dynamic Adjustments**:
   - Introduced logic to scale visual elements dynamically based on repository size.

2. **UI Improvements**:
   - Added a "Reset Settings" button for convenience.

## Installation

- Install poetry

- Build the env:

```console
poetry install
```

- Launch the program

## Usage

To use `repovis`:

```console
source .venv/bin/activate
python repo_vis/repovis.py --repo_path <path to repository source>
```

## General Considerations

I've worked to dynamically reduce the object resolution as a function of the number of classes, functions
and methods, but large repositories can take considerable time to load and render. Minutes on my Macbook Pro M3 max, so prepare to wait a bit. Progress bars should help keep an eye on rendering.
