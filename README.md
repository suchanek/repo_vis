# pkg_visualizer

`pkg_visualizer` is a Python-based application that visualizes the structure of Python packages in 3D. It provides an interactive and dynamic way to explore Classes, Methods, and Functions within a package. Built using PyVista and PyQt5, the tool is designed for developers and researchers who want to gain insights into codebases visually. The `examples` directory contains exported HTML snapshots of several popular Python repositories.

## Features

- **Dynamic Visualization**:
  - Adjust class object size at runtime via a slider (`class_radius`).
  - Method and function sizes scale automatically relative to the class and package radii.
  - Adaptive shapes based on element count:
    - Classes rendered as icosahedrons (low-to-moderate count) or cubes (very large count).
    - Functions rendered as cylinders (up to 1000) or cubes (above 1000).
  - Adaptive connections:
    - Class connections drawn as cylinders when total classes < 500.
    - Class connections drawn as lines when classes between 500 and 2000.
    - Connections disabled when classes exceed 2000.
  - Enhanced status display with color-coded, larger text for better visibility.

- **Interactive UI**:
  - **Class Selector**: Multi-select list to include specific classes.
  - **Method Selector**: Multi-select list to include specific methods. Disabled when "Render Methods" is unchecked.
  - **Function Selector**: Multi-select list to include specific functions. Disabled when "Render Functions" is unchecked.
  - **Checkboxes** to enable/disable rendering of functions and methods, which also toggles the corresponding selectors.
  - **Visualize** button to generate or update the 3D scene.

- **Docstring Popups & Picking**:
  - Click on a list entry or mesh to highlight the object and open a popup displaying its docstring (rendered via Markdown).

- **Camera Controls**:
  - **Reset View** button to restore the camera to the default orientation.
  - **Spin Package** button to animate a smooth 360° rotation around the package center.
  - Keyboard and mouse interactions supported by PyVista.

- **Export & Save**:
  - Save the current view to **HTML**, **PNG**, or **JPG** via the **Save View** button.
  - Saved files include the current camera position and scene state.
  - Robust file saving with automatic directory creation and comprehensive error handling.

- **Progress & Metrics**:
  - Simple ASCII progress bars and percentage updates in the status display during rendering.
  - Automatic calculation of total triangle (face) count for classes, methods, functions, and connections.
  - Triangle count is shown in the window title upon completion.

## Installation

1. Clone the repository and navigate into it:

   ```bash
   git clone <repo_url>
   cd repo_vis
   ```

2. Install dependencies (via Poetry):

   ```bash
   poetry install
   ```

3. Activate the virtual environment:

   ```bash
   source .venv/bin/activate
   ```

## Usage

Run the visualization tool by specifying at least the package path. Optionally, provide an output save path.

```bash
python pkg_visualizer/pkg_visualizer.py \
  --package_path /path/to/your/python/package \
  [--save_path /desired/output/path] \
  [--width 1200] \
  [--height 800]
```

- **--package_path**: Path to the root of the Python package to visualize.
- **--save_path**: (Optional) Base save path (without extension). The tool will append `.html`, `.png`, or `.jpg` depending on the chosen format.
- **--width**: (Optional) Width of the visualization window (default: 1200).
- **--height**: (Optional) Height of the visualization window (default: 800).

## General Considerations

- Large repositories (thousands of classes or functions) can take several minutes to parse and render.
- Progress updates and adaptive detail settings help manage performance.
- Adjust the **Class Radius** slider to zoom in on class-level structure or zoom out for an overview.
- Safe file saving with parent directory creation ensures visualizations can be saved anywhere.
- Automatic error handling prevents crashes when dealing with invalid paths or permissions.

## Examples

The `examples` directory contains pre-rendered HTML visualizations of several popular Python projects:

- Flask
- Matplotlib
- ProteusPy
- Requests
- Seaborn
- SymPy
- TensorFlow

Open these files in any modern web browser to explore interactive 3D visualizations without having to generate them yourself.

Happy visualizing! 🚀
