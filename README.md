# pkg_visualizer

`pkg_visualizer` is a Python-based application that visualizes the structure of Python packages in 3D. It provides an interactive and dynamic way to explore Classes, Methods, and Functions within a package. Built using PyVista and PyQt5, the tool is designed for developers and researchers who want to gain insights into codebases visually. The `examples` directory contains exported HTML snapshots of several popular Python repositories.

## Features

- **Dynamic Visualization**:
  - Adjust class object size at runtime via a slider (`class_radius`).
  - Method and function sizes scale automatically relative to the class and package radii.
  - Adaptive shapes based on element count:
    - Classes rendered as icosahedrons (low-to-moderate count) or cubes (very large count).
    - Functions rendered as cylinders (up to 1000) or cubes (above 1000).
  - Spatially-optimized distribution:
    - Classes distributed on a Fibonacci sphere around the package center.
    - Functions arranged in a Fibonacci annulus (ring) around the package, creating a clear visual distinction.
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

## Under the Hood

This section explains the internal workings of the visualization algorithm to provide insight into how the 3D representations are created.

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

- **Fibonacci Annulus for Functions**: Functions are arranged in a ring-shaped region (annulus) using a Fibonacci spiral algorithm. This creates a clear visual distinction between functions and classes, with functions positioned in a flat disk with inner radius of 1.25× the package radius.

- **Hierarchical Positioning**:
  - Package center serves as the origin (0,0,0)
  - Classes orbit around the package in a spherical arrangement
  - Methods orbit their parent classes in smaller spheres
  - Functions form a spiral pattern in an annular disk around the package

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

#### Triangle Count Calculation

- **Face Counting**: The total triangle count is calculated by summing:
  - Faces from class meshes
  - Faces from method meshes
  - Faces from function meshes
  - Faces from all connection geometries

- This provides a performance metric and indicates visualization complexity

### UI Integration

- **PyQt5 and PyVista Integration**: The Qt-based UI is integrated with PyVista's rendering capabilities through the QtInteractor class.

- **Real-time Updates**: Status messages, progress bars, and selection updates are processed through Qt's event loop with `QApplication.processEvents()` to maintain UI responsiveness.

- **Docstring Formatting**: Python docstrings are converted to Markdown for better readability using regex-based parsing.

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
