# pkg_visualizer

`pkg_visualizer` is a Python-based application that visualizes the structure of Python packages in 3D. It provides an interactive and dynamic way to explore Classes, Methods, and Functions within a package. Built using PyVista and PyQt5, the tool is designed for developers and researchers who want to gain insights into codebases visually. The `examples` directory contains exported HTML snapshots of several popular Python repositories.

**Version:** 0.1.0  
**Last Updated:** 2025-05-23  
**Author:** Eric G. Suchanek, PhD

## Features

- **Dynamic Visualization**:
  - Adjust class object size at runtime via a slider (`class_radius`).
  - Method and function sizes scale automatically relative to the class and package radii.
  - Adaptive shapes based on element count:
    - Classes rendered as icosahedrons (low-to-moderate count) or cubes (very large count).
    - Functions rendered as cylinders (up to 1000) or cubes (above 1000).
  - Spatially-optimized distribution:
    - Classes distributed on a Fibonacci sphere around the package center.
    - Functions positioned on a separate Fibonacci sphere with a different radius, creating a clear visual distinction.
    - Methods orbit their parent classes in smaller spheres.
  - Adaptive connections:
    - Class connections drawn as cylinders when total classes < 500.
    - Class connections drawn as lines when classes between 500 and 2000.
    - Connections disabled when classes exceed 2000.
  - Enhanced status display with color-coded, larger text for better visibility.

- **Interactive UI**:
  - **Class Selector**: Select specific classes.
  - **Method Selector**: Select to include specific methods. Disabled when "Render Methods" is unchecked.
  - **Function Selector**: Select to include specific functions. Disabled when "Render Functions" is unchecked.
  - **Checkboxes** to enable/disable rendering of functions and methods, which also toggles the corresponding selectors.
  - **Visualize** button to generate or update the 3D scene.

- **Docstring Popups & Picking**:
  - Click on a list entry or mesh to highlight the object and open a popup displaying its docstring (rendered via Markdown).
  - Popups are modeless (non-blocking) and display formatted docstrings with syntax highlighting.
  - Automatic zooming and highlighting of selected objects for better visibility.
  - **Priority Handling**: If a method is selected, its docstring is displayed instead of the class docstring.
  - Closing the popup resets the view and clears the selection.

- **Camera Controls**:
  - **Reset View** button to restore the camera to the default orientation.
  - **Spin Package** button to animate a smooth 360° rotation around the package center.
  - Keyboard and mouse interactions supported by PyVista.

- **Export & Save**:
  - Save the current view to **HTML**, **PNG**, or **JPG** via the **Save View** button.
  - Saved files include the current camera position and scene state.
  - Robust file saving with automatic directory creation and comprehensive error handling.

- **Progress & Metrics**:
  - Progress bars in the status display during rendering.
  - Automatic calculation of total triangle (face) count for classes, methods, functions, and connections.
  - Face count is shown in the window title upon completion.
  - Color-coded status messages for better visibility and user feedback.

## Installation

1. Clone the repository and navigate into it:

   ```bash
   git clone https://github.com/suchanek/repo_vis
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

Run the visualization tool by specifying at least the package path. It is an entrypoint so can be run directly from the env:

```bash
poetry run pkg-visualizer \
  [--package_path /path/to/your/python/package] \
  [--save_path /desired/output/path] \
  [--width 1200] \
  [--height 800]
```

- **--package_path**: (Optional) Path to the root of the Python package to visualize. Defaults to my personal repo.
- **--save_path**: (Optional) Base save path (without extension). The tool will append `.html`, `.png`, or `.jpg` depending on the chosen format.
- **--width**: (Optional) Width of the visualization window (default: 1200).
- **--height**: (Optional) Height of the visualization window (default: 800).

### Picking and interacting with the scene

It's possible to pick itmes from the screen directly by hovering the mouse over the item of interest
and either right-clicking on the object or prressing the **P** key. Picking isn't all that precise.
When the pick succeeds the camera zooms to the object and highlights it and displays the relevent
docstring in the docstring popup window. If you can't see the object spin the camera (click drag the mouse).

### UI Integration

- **PyQt5 and PyVista Integration**: The Qt-based UI is integrated with PyVista's rendering capabilities through the QtInteractor class.

- **Real-time Updates**: Status messages, progress bars, and selection updates are processed through Qt's event loop with `QApplication.processEvents()` to maintain UI responsiveness.

- **Docstring Formatting**: Python docstrings are converted to Markdown for better readability using regex-based parsing.

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

## User Interface Buttons

The `pkg_visualizer` application provides an intuitive user interface with the following buttons and their respective functions:

- **Visualize**: Generates or updates the 3D visualization based on the selected parameters.
- **Reset View**: Resets the camera to the default orientation, providing a clear view of the entire package structure.
- **Spin Package**: Animates a smooth 360° rotation of the visualization around the package center.
- **Save View**: Saves the current visualization to a file. Supported formats include HTML, PNG, and JPG.
- **Class Selector**: Allows users to select specific classes to include in the visualization.
- **Method Selector**: Enables selection of specific methods. This option is disabled when "Render Methods" is unchecked.
- **Function Selector**: Enables selection of specific functions. This option is disabled when "Render Functions" is unchecked.
- **Checkboxes**: Toggle the rendering of functions and methods, which also enables or disables the corresponding selectors.
- **Show Docstring**: Shows the docstring for the selected Class/Method/Function.

These buttons and controls make it easy to customize and interact with the 3D visualization.

## Under the Hood

This section explains the internal workings of the visualization algorithm to provide insight into how the 3D representations are created.

### Memory Management

- **Global Cleanup**: Comprehensive cleanup function registered with `atexit` to ensure proper resource release.
- **Exception Handling**: Custom exception hook to catch and handle specific PyVista/VTK errors during shutdown.

### Package Analysis & Parsing

- **AST-Based Code Parsing**: The tool uses Python's Abstract Syntax Tree (AST) module to parse Python files and extract structural information without executing the code.

- **Element Collection**: The system walks through all `.py` files in the package recursively, identifying:
  - Classes and their methods
  - Standalone functions
  - Docstrings for all elements

- **Duplicates Handling**: A tracking mechanism (`seen_classes` and `seen_functions` sets) ensures that each class and function is only included once in the visualization.

### Visualization Algorithm

#### 3D Space Distribution

- **Fibonacci Sphere Distribution**: Classes are positioned using the Fibonacci sphere algorithm, which creates nearly uniform point distributions on a sphere's surface, ensuring optimal spacing even with large numbers of elements.

- **Fibonacci Sphere for Functions**: Functions are positioned on a separate Fibonacci sphere with radius of 1.25× the package radius (scaled based on function count). This creates a clear visual distinction between functions and classes.

- **Hierarchical Positioning**:
  - Package center serves as the origin (0,0,0)
  - Classes orbit around the package in a spherical arrangement
  - Methods orbit their parent classes in smaller spheres
  - Functions are positioned on a separate sphere with a larger radius than the classes

#### Mesh Generation & Rendering

- **Adaptive Geometry Selection**:
  - For small to moderate packages (< 2000 classes): Icosahedrons represent classes
  - For large packages (> 2000 classes): Simpler cube geometries are used
  - For functions: Cylinders up to 1000, cubes above 1000
  - Methods are always rendered as icosahedrons
  
- **Connection Logic**:
  - Connections are rendered differently based on element count:
    - Cylinders for fewer than 500 classes (high detail)
    - Lines for 500-2000 classes (medium detail)
    - No connections for extremely large packages (> 2000 classes) to improve performance
  - Method connections are only drawn if total methods < 2000

- **Performance Optimization**:
  - MultiBlock collections group similar meshes for efficient rendering
  - Triangle count is carefully tracked and displayed
  - Selective rendering of methods and functions based on user preferences
  - Progress tracking with incremental updates for large packages

#### Picking & Interaction

- **Mesh Mapping System**: Each 3D object (mesh) is assigned a unique ID that maps to its corresponding code element (class, method, or function).

- **Picking Algorithm**: PyVista's picking functionality is enhanced with custom callbacks that:
  1. Identify which mesh was clicked
  2. Retrieve the associated element information
  3. Display docstrings in formatted Markdown popups
  4. Highlight the selected element
  5. Zoom in on the selected object for detailed inspection
  6. Reset view when popup is closed

#### Triangle Count Calculation

- **Face Counting**: The total triangle count is calculated by summing:
  - Faces from class meshes
  - Faces from method meshes
  - Faces from function meshes
  - Faces from all connection geometries

- This provides a performance metric and indicates visualization complexity

### General Considerations

- Large repositories (thousands of classes or functions) can take several minutes to parse and render.
- Progress updates and adaptive detail settings help manage performance.
- Adjust the **Class Radius** slider to zoom in on class-level structure or zoom out for an overview.
- Safe file saving with parent directory creation ensures visualizations can be saved anywhere.
- Automatic error handling prevents crashes when dealing with invalid paths or permissions.
- Memory management optimizations prevent common PyVista/VTK errors during application exit.

Happy visualizing! 🚀
