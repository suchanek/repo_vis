# repovis

`repovis` is a Python-based application that visualizes the structure of Python repositories in 3D. It provides an interactive and dynamic way to explore the Classes, Methods, and Functions within a repository. Built using Panel and other visualization libraries, the tool is designed for developers and researchers who want to gain insights into codebases visually.

## Features

- **Dynamic Visualization**:
  - Automatically adjusts the size of visual elements (e.g., classes and methods) based on the repository size.
  - Uses cubes for functions when their count exceeds 1000 for better performance and clarity.

- **Interactive UI**:
  - Includes sliders to control various visualization parameters.
  - "Reset Settings" button to restore default slider values and camera position.

- **Customizable Appearance**:
  - Supports dynamic scaling of class and method objects using `class_object_radius` and `method_object_radius`.
  - Improved font handling with "Arial" for better rendering.

- **Code Insights**:
  - Displays the hierarchy and relationships between classes, methods, and functions in a repository.

## Recent Enhancements

1. **Dynamic Adjustments**:
   - Introduced logic to scale visual elements dynamically based on repository size.

2. **UI Improvements**:
   - Added a "Reset Settings" button for convenience.

## Usage

To use `repo_vis`, run the `repovis.py` script and follow the on-screen instructions to load and visualize a Python repository. The application provides an intuitive interface to navigate and explore the repository's structure.
