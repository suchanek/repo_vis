# pylint: disable=C0301
# pylint: disable=C0116
# pylint: disable=C0115

"""
Module: visualize_repository_panel

This program provides a Panel-based application for visualizing the structure of a Python repository in 3D.
It uses PyVista for 3D rendering and Panel for creating an interactive user interface. The application
analyzes a given repository, extracts classes, methods, and standalone functions, and generates a 3D
visualization of their relationships.

Key Features:
- Parses Python files in a repository to extract classes, methods, and functions.
- Visualizes the repository structure in 3D using spheres, cylinders, and lines to represent classes,
  methods, and their relationships.
- Provides an interactive user interface with widgets for customizing visualization parameters,
  selecting specific classes or functions, and saving the visualization in various formats (HTML, PNG, JPEG).
- Supports off-screen rendering for generating screenshots.

Classes:
- RepositoryVisualizer: A parameterized class that manages repository analysis, visualization, and
  user interactions.

Functions:
- parse_file(file_path): Parses a Python file to extract classes, methods, and functions.
- collect_elements(repo_path): Collects all classes, methods, and functions from a repository.
- fibonacci_sphere(samples, radius, center): Generates points uniformly distributed on a sphere.
- create_3d_visualization_for_panel(elements, save_path, save_format, ...): Creates a 3D visualization
  of the repository structure.

Usage:
Run the script to launch the interactive application. Use the provided widgets to specify the repository
path, customize visualization parameters, and generate the 3D visualization.
 - panel serve <path to this script> --show

Author: Eric G. Suchanek, PhD
Last modified: 2025-05-02 07:54:33

"""

# -*- coding: utf-8 -*-

import ast
import logging
import os
from pathlib import Path

import numpy as np
import panel as pn
import param
import pyvista as pv
from rich.console import Console
from rich.progress import Progress

# Ensure logging level is set to DEBUG and output is directed to the console
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger()

# Log PyVista version
print("PyVista version:", pv.__version__)

# Set PyVista off-screen rendering
pv.OFF_SCREEN = False

print("pv.OFF_SCREEN:", pv.OFF_SCREEN)

pn.extension("vtk", sizing_mode="stretch_both")

# Global plotter
plotter = pv.Plotter(off_screen=pv.OFF_SCREEN)
plotter.clear()
plotter.add_floor(i_resolution=100, j_resolution=100)

ORIGIN = (0, 0, 0)

# Initialize plotter with a placeholder cube
plotter.add_mesh(
    pv.Cube(center=ORIGIN, x_length=1, y_length=1, z_length=1), color="gray"
)

plotter.set_background("white")
plotter.enable_anti_aliasing("msaa")
plotter.enable_parallel_projection()
plotter.reset_camera()

DEFAULT_REP = "/Users/egs/repos/proteusPy"
# Extract default package name from the repository path
DEFAULT_PACKAGE_NAME = os.path.basename(DEFAULT_REP)

logger.info("Starting repository visualization for package: %s", DEFAULT_PACKAGE_NAME)


def parse_file(file_path):
    """Parse a Python file and extract classes, methods, and functions."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            tree = ast.parse(file.read(), filename=file_path)
    except (SyntaxError, UnicodeDecodeError):
        return []

    elements = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            elements.append(
                {
                    "type": "class",
                    "name": node.name,
                    "methods": [
                        n.name for n in node.body if isinstance(n, ast.FunctionDef)
                    ],
                    "lineno": node.lineno,
                }
            )
        elif isinstance(node, ast.FunctionDef) and not any(
            isinstance(parent, ast.ClassDef) for parent in ast.iter_child_nodes(tree)
        ):
            elements.append(
                {"type": "function", "name": node.name, "lineno": node.lineno}
            )
    return elements


def collect_elements(repo_path):
    console = Console()
    with Progress() as progress:
        task = progress.add_task("Collecting elements...", total=1)
    """Collect all classes, methods, and functions from the repository, avoiding duplicate class names."""
    elements = []
    seen_classes = set()
    seen_functions = set()
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                file_elements = parse_file(file_path)
                for elem in file_elements:
                    if elem["type"] == "class":
                        if elem["name"] not in seen_classes:
                            seen_classes.add(elem["name"])
                            elements.append(elem)
                        else:
                            print(
                                f"Skipping duplicate class '{elem['name']}' in {file_path}"
                            )  # Debug
                    elif elem["type"] == "function":
                        if elem["name"] not in seen_functions:
                            seen_functions.add(elem["name"])
                            elements.append(elem)
                        else:
                            print(
                                f"Skipping duplicate function '{elem['name']}' in {file_path}"
                            )  # Debug
    logger.debug(
        f"Collected elements: {[e['name'] for e in elements if e['type'] == 'class']} (classes), {[e['name'] for e in elements if e['type'] == 'function']} (functions)"
    )  # Debug
    return elements


def fibonacci_sphere(samples, radius=1.0, center=None):
    """
    Generate points uniformly distributed on a sphere using the Fibonacci algorithm.

    :param samples: Number of points to generate
    :type samples: int
    :param radius: Radius of the sphere
    :type radius: float
    :param center: Center of the sphere (default: origin)
    :type center: list or numpy.ndarray or None
    :return: Array of 3D points on the sphere
    :rtype: list
    """
    if center is None:
        center = np.array([0, 0, 0])

    # Handle special cases
    if samples <= 0:
        return []

    if samples == 1:
        # For a single point, place it at the top of the sphere
        return [center + radius * np.array([0, 0, 1])]

    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius_at_y = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # Golden angle increment

        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y

        # Scale by radius and shift to center
        points.append(center + radius * np.array([x, y, z]))

    return points


def create_3d_visualization_for_panel(
    elements,
    save_path,
    save_format="html",
    class_radius=4.0,
    member_radius_scale=1.0,
    old_title="",
):
    """Update the global plotter with a 3D visualization of the repository structure and handle screenshots with a separate off-screen plotter."""
    global plotter
    logger.debug("Creating visualization for %s", save_path)

    # Reinitialize global plotter for interactive visualization
    plotter = pv.Plotter(off_screen=pv.OFF_SCREEN)

    plotter.add_floor()
    print("Reinitialized global plotter")
    package_center = np.array([0, 0, 0])
    package_name = Path(save_path).stem
    package_size = 0.8
    package_mesh = pv.Cube(
        center=package_center,
        x_length=package_size,
        y_length=package_size,
        z_length=package_size,
    )
    plotter.add_mesh(package_mesh, color="purple", show_edges=True, smooth_shading=True)
    """
    plotter.add_text(
        package_name,
        position=package_center + [0, 0, package_size * 1.5],
        font_size=14,
        color="black",
    )
    """

    num_classes = len([e for e in elements if e["type"] == "class"])
    print(
        "Rendering",
        num_classes,
        "classes:",
        [e["name"] for e in elements if e["type"] == "class"],
    )
    class_size = 0.75
    class_positions = []
    class_names = []
    sphere_positions = fibonacci_sphere(
        num_classes, radius=class_radius, center=package_center
    )
    class_index = 0
    for _, element in enumerate(elements):
        if element["type"] != "class":
            continue
        print("Processing class:", element["name"])
        pos = sphere_positions[class_index]
        class_index += 1
        class_positions.append(pos)
        class_names.append(element["name"])
        mesh = pv.Dodecahedron(radius=class_size / 2, center=pos)
        plotter.add_mesh(
            mesh,
            color="red",
            show_edges=True,
            smooth_shading=False,
        )
        direction = pos - package_center
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)

        line = pv.Line(package_center, pos)
        plotter.add_mesh(line, color="gray", line_width=4, smooth_shading=True)

    method_size = 0.3 * 0.75
    function_size = 0.3 * 0.5

    # Add standalone functions to the visualization
    num_functions = len([e for e in elements if e["type"] == "function"])
    if num_functions > 0:
        print(f"Rendering {num_functions} functions")
        function_positions = fibonacci_sphere(
            num_functions, radius=class_radius * 0.4, center=package_center
        )

        with Progress() as progress:
            task = progress.add_task("Rendering functions...", total=num_functions)
            for i, element in enumerate(
                [e for e in elements if e["type"] == "function"]
            ):
                pos = function_positions[i]
                mesh = pv.Cylinder(
                    radius=function_size / 2,
                    height=function_size / 2,
                    center=pos,
                    direction=(0, 0, 1),
                )
                plotter.add_mesh(
                    mesh, color="green", show_edges=True, smooth_shading=True
                )

                line = pv.Line(package_center, pos)
                plotter.add_mesh(
                    line, color="lightgray", line_width=1, smooth_shading=True
                )

                progress.update(task, advance=1)

    member_radius = member_radius_scale * class_size
    for class_idx, (class_pos, class_elem) in enumerate(
        zip(class_positions, [e for e in elements if e["type"] == "class"])
    ):
        members = class_elem.get("methods", [])
        num_members = len(members)
        if num_members > 0:
            with Progress() as progress:
                task = progress.add_task(
                    f"Rendering methods for class {class_elem['name']}...",
                    total=num_members,
                )
                method_positions = fibonacci_sphere(
                    num_members, radius=member_radius, center=class_pos
                )
                for j, member_name in enumerate(members):
                    member_pos = method_positions[j]
                    sphere = pv.Sphere(radius=method_size / 2, center=member_pos)
                    plotter.add_mesh(
                        sphere, color="blue", show_edges=False, smooth_shading=True
                    )

                    line = pv.Line(class_pos, member_pos)
                    plotter.add_mesh(
                        line, color="gray", line_width=2, smooth_shading=True
                    )

                    progress.update(task, advance=1)

    plotter.add_light(pv.Light(position=(10, 10, 10), color="white", intensity=1.0))
    plotter.add_light(pv.Light(position=(-10, -10, 10), color="white", intensity=0.8))

    num_classes = len([e for e in elements if e["type"] == "class"])
    num_methods = sum(
        len(e.get("methods", [])) for e in elements if e["type"] == "class"
    )
    num_functions = len([e for e in elements if e["type"] == "function"])
    title_text = f"3D Visualization: {package_name} | Classes: {num_classes} | Methods: {num_methods} | Functions: {num_functions}"
    plotter.add_text(
        title_text,
        position="upper_edge",
        font_size=14,
        color="black",
    )
    plotter.set_background("white")
    plotter.add_axes()

    # Enhanced camera setup
    bounds = plotter.bounds
    max_dim = max(
        abs(bounds[1] - bounds[0]),
        abs(bounds[3] - bounds[2]),
        abs(bounds[5] - bounds[4]),
    )
    # Ensure max_dim accounts for scene size, with a minimum to avoid zero bounds
    max_dim = max(max_dim, 2 * class_radius, 1.0)
    distance_factor = 4.0  # Increased to ensure full scene visibility
    plotter.camera_position = [
        (
            distance_factor * max_dim,
            distance_factor * max_dim,
            distance_factor * max_dim,
        ),
        ORIGIN,  # Focal point at scene center
        (0, 0, 1),  # Up vector
    ]
    # plotter.camera.zoom(0.8)  # Slightly increased zoom for better framing
    plotter.show_bounds(grid="front", location="outer", all_edges=True)
    plotter.show_grid()
    plotter.reset_camera()
    plotter.render()

    # Save the visualization
    save_path = Path(save_path).with_suffix(f".{save_format}")
    try:
        if save_format == "html":
            plotter.export_html(save_path)
            print("Saved HTML visualization to", save_path)
        elif save_format in ["png", "jpeg"]:
            # Create a new off-screen plotter for screenshots
            screenshot_plotter = pv.Plotter(off_screen=True)

            plotter.add_floor()
            print("Created off-screen plotter for screenshot")

            # Copy meshes from global plotter
            for actor in plotter.actors.values():
                if isinstance(actor, pv.Actor):
                    if hasattr(actor, "mapper") and actor.mapper.GetInput():
                        mesh = actor.mapper.GetInput()
                        screenshot_plotter.add_mesh(
                            mesh,
                            color=actor.prop.GetColor(),
                            show_edges=actor.prop.GetEdgeVisibility(),
                            line_width=actor.prop.GetLineWidth(),
                            smooth_shading=actor.prop.GetInterpolation()
                            != pv.InterpolationType.FLAT,
                        )

            # Copy text actors
            for actor_key, actor in plotter.actors.items():
                if "text" in actor_key.lower() and hasattr(actor, "GetText"):
                    text = actor.GetText()
                    position = (
                        actor.GetPosition()
                        if hasattr(actor, "GetPosition")
                        else "upper_edge"
                    )
                    font_size = (
                        actor.GetTextProperty().GetFontSize()
                        if hasattr(actor, "GetTextProperty")
                        else 14
                    )
                    color = (
                        actor.GetTextProperty().GetColor()
                        if hasattr(actor, "GetTextProperty")
                        else "black"
                    )
                    """
                    screenshot_plotter.add_text(
                        text,
                        position=position,
                        font_size=font_size,
                        color=color,
                    )
                    """

            # Copy lights
            for light in plotter.renderer.GetLights():
                screenshot_plotter.add_light(
                    pv.Light(
                        position=light.GetPosition(),
                        color=light.GetDiffuseColor(),
                        intensity=light.GetIntensity(),
                    )
                )

            # Copy axes
            screenshot_plotter.add_axes()

            # Copy camera settings
            screenshot_plotter.camera_position = plotter.camera_position
            screenshot_plotter.set_background("white")

            # Render and save screenshot
            print("Saving screenshot with off-screen plotter")
            screenshot_plotter.screenshot(save_path, window_size=[1200, 800])
            screenshot_plotter.close()  # Clean up
            print("Saved screenshot to", save_path)
    except Exception as e:
        logger.error("Failed to save visualization: %s", e)
        if save_format in ["png", "jpeg"]:
            fallback_path = Path(save_path).with_suffix(".html")
            try:
                plotter.export_html(fallback_path)
                logger.info(f"Saved fallback HTML visualization to {fallback_path}")
            except Exception as fallback_error:
                logger.error(f"Fallback save also failed: {fallback_error}")

    return plotter, title_text


class RepositoryVisualizer(param.Parameterized):
    repo_path = param.String(
        default=DEFAULT_REP, doc="Path to the Python repository to analyze"
    )
    save_path = param.String(
        default=DEFAULT_PACKAGE_NAME,
        doc="Path to save the visualization file (without extension)",
    )
    old_title = param.String(
        default=f"{DEFAULT_PACKAGE_NAME} 3d visualization",
        doc="Title for the visualization",
    )
    save_format = param.Selector(
        objects=["html", "png", "jpeg"],
        default="html",
        doc="Format for the output file",
    )
    class_radius = param.Number(
        default=4.0,
        bounds=(1.0, 10.0),
        step=0.5,
        doc="Radius of the sphere for class placement",
    )
    member_radius_scale = param.Number(
        default=1.25,
        bounds=(0.5, 3.0),
        step=0.25,
        doc="Scaling factor for the radius of member placement around classes",
    )
    show_interactive = param.Boolean(
        default=False, doc="Show the visualization interactively (separate window)"
    )
    available_classes = param.List(
        default=[], doc="Available classes in the repository"
    )
    selected_classes = param.ListSelector(
        default=[], objects=[], doc="Classes to highlight"
    )
    available_functions = param.List(
        default=[], doc="Available functions in the repository"
    )
    selected_functions = param.ListSelector(
        default=[], objects=[], doc="Functions to highlight"
    )
    include_functions = param.Boolean(
        default=True, doc="Include functions in the visualization"
    )
    status = param.String(default="Ready", doc="Status of the visualization process")

    def __init__(self, **params):
        super().__init__(**params)
        self.elements = []
        self.update_classes()  # Automatically parse default repository

    @param.depends("repo_path", watch=True)
    def update_classes(self):
        print("Updating classes and functions for repo_path:", self.repo_path)
        if os.path.exists(self.repo_path):
            self.status = "Analyzing repository..."
            self.elements = collect_elements(self.repo_path)

            # Process classes
            class_names = sorted(
                [e["name"] for e in self.elements if e["type"] == "class"]
            )  # Sort classes
            print("Found classes:", class_names)
            self.selected_classes = []
            self.available_classes = class_names
            self.param.selected_classes.objects = class_names

            # Process functions
            function_names = sorted(
                [e["name"] for e in self.elements if e["type"] == "function"]
            )  # Sort functions
            print("Found functions:", function_names)
            self.selected_functions = []
            self.available_functions = function_names
            self.param.selected_functions.objects = function_names

            # Update save path to match the repository name when repo path changes
            self.save_path = os.path.basename(self.repo_path)
            self.status = f"Found {len(class_names)} classes and {len(function_names)} functions in the repository."
        else:
            self.status = "Repository path does not exist."
            self.available_classes = []
            self.selected_classes = []
            self.param.selected_classes.objects = []
            self.available_functions = []
            self.selected_functions = []
            self.param.selected_functions.objects = []
            print("Invalid repo path")

    def visualize(self, event=None):
        print("Visualize called")
        if not os.path.exists(self.repo_path):
            self.status = "Repository path does not exist."
            logger.error("Invalid repo path: %s", self.repo_path)
            return
        if not self.elements:
            self.status = "No elements found in the repository."
            logger.error("No elements found")
            return
        self.status = "Creating visualization..."

        # Filter elements based on selection
        filtered_elements = []

        # Add selected classes (or all classes if none selected)
        if self.selected_classes:
            class_names = self.selected_classes
            for e in self.elements:
                if e["type"] == "class" and e["name"] in class_names:
                    filtered_elements.append(e)
        else:
            # If no classes selected, include all classes
            for e in self.elements:
                if e["type"] == "class":
                    filtered_elements.append(e)

        # Add selected functions (or all functions if none selected and include_functions is True)
        if self.include_functions:
            if self.selected_functions:
                function_names = self.selected_functions
                for e in self.elements:
                    if e["type"] == "function" and e["name"] in function_names:
                        filtered_elements.append(e)
            else:
                # If no functions selected but include_functions is True, include all functions
                for e in self.elements:
                    if e["type"] == "function":
                        filtered_elements.append(e)

        # Update status message
        status_parts = []
        if self.selected_classes:
            status_parts.append(f"Selected classes: {', '.join(self.selected_classes)}")
        if self.include_functions and self.selected_functions:
            status_parts.append(
                f"Selected functions: {', '.join(self.selected_functions)}"
            )

        if status_parts:
            self.status = " | ".join(status_parts)

        print("Filtered elements:", [e["name"] for e in filtered_elements])
        save_path = self.save_path
        if not save_path.endswith(f".{self.save_format}"):
            save_path = f"{save_path}.{self.save_format}"
        print("Save path:", save_path)
        try:
            _pl, old_title = create_3d_visualization_for_panel(
                filtered_elements,
                save_path,
                self.save_format,
                self.class_radius,
                self.member_radius_scale,
                self.old_title,
            )
            self.old_title = old_title
            print("Updating VTK pane")
            vtkpan.object = plotter.ren_win
            self.status = f"Visualization saved to {save_path}"
            self.old_title = old_title
        except Exception as e:
            self.status = f"Error creating visualization: {str(e)}"
            logger.error("Visualization error: %s", e)

    def browse_repo(self, event=None):
        self.status = "Please enter the repository path manually in the text field."
        print("Browse repo called")

    def browse_save(self, event=None):
        self.status = "Please enter the save path manually in the text field."
        print("Browse save called")


# Create the visualizer instance
visualizer = RepositoryVisualizer()

# Create widgets
repo_path_input = pn.widgets.TextInput(
    name="Repository Path",
    value=visualizer.repo_path,
    placeholder="Enter the path to the Python repository",
)
repo_path_input.link(visualizer, value="repo_path")

save_path_input = pn.widgets.TextInput(
    name="Save Path",
    value=visualizer.save_path,
    placeholder="Enter the path to save the visualization",
)
save_path_input.link(visualizer, value="save_path")

save_format_select = pn.widgets.Select(
    name="Save Format", options=["html", "png", "jpeg"], value=visualizer.save_format
)
save_format_select.link(visualizer, value="save_format")

class_radius_slider = pn.widgets.FloatSlider(
    name="Class Radius", start=1.0, end=10.0, step=0.5, value=visualizer.class_radius
)
class_radius_slider.link(visualizer, value="class_radius")

member_radius_scale_slider = pn.widgets.FloatSlider(
    name="Member Radius Scale",
    start=0.5,
    end=3.0,
    step=0.25,
    value=visualizer.member_radius_scale,
)
member_radius_scale_slider.link(visualizer, value="member_radius_scale")

"""
show_interactive_checkbox = pn.widgets.Checkbox(
    name="Show Interactive Visualization", value=visualizer.show_interactive
)
show_interactive_checkbox.link(visualizer, value="show_interactive")
"""

class_selector = pn.widgets.MultiSelect(
    name="Select Classes to Visualize",
    options=visualizer.available_classes,
    value=visualizer.selected_classes,
    size=5,
)
class_selector.link(visualizer, value="selected_classes")


# Update class selector options
def update_class_selector(event):
    class_selector.options = event.new
    print("Updated class selector options:", event.new)


visualizer.param.watch(update_class_selector, "available_classes")

# Create function selector
function_selector = pn.widgets.MultiSelect(
    name="Select Functions to Visualize",
    options=visualizer.available_functions,
    value=visualizer.selected_functions,
    size=5,
)
function_selector.link(visualizer, value="selected_functions")


# Update function selector options
def update_function_selector(event):
    function_selector.options = event.new
    print("Updated function selector options:", event.new)


visualizer.param.watch(update_function_selector, "available_functions")

# Create include functions checkbox
include_functions_checkbox = pn.widgets.Checkbox(
    name="Include Functions in Visualization", value=visualizer.include_functions
)
include_functions_checkbox.link(visualizer, value="include_functions")

visualize_button = pn.widgets.Button(name="Visualize Repository", button_type="success")
visualize_button.on_click(visualizer.visualize)

status_text = pn.widgets.StaticText(name="Status", value=visualizer.status)
visualizer.param.watch(lambda event: setattr(status_text, "value", event.new), "status")

result_display = pn.pane.HTML("")


# Update result display
def update_result(event):
    if visualizer.status.startswith("Visualization saved to"):
        save_path = visualizer.status.split("Visualization saved to ")[-1].strip()
        file_url = f"file://{os.path.abspath(save_path)}"
        result_display.object = (
            f'<div style="margin-top: 20px; padding: 15px; background-color: #f0f0f0; border-radius: 5px;">'
            f"<h3>Visualization Created</h3>"
            f"<p>The visualization has been saved to: <strong>{save_path}</strong></p>"
            f'<a href="{file_url}" target="_blank" style="font-size: 16px; padding: 10px; '
            f"background-color: #4CAF50; color: white; text-decoration: none; border-radius: 4px; "
            f"</div>"
        )
        print("Updated result display for", save_path)


visualize_button.on_click(update_result)

# Create input panel
input_panel = pn.Column(
    pn.pane.Markdown("# Repository Visualizer"),
    pn.pane.Markdown("## Input Parameters"),
    repo_path_input,
    save_path_input,
    save_format_select,
    class_radius_slider,
    member_radius_scale_slider,
    pn.pane.Markdown("## Class Selection"),
    pn.pane.Markdown("Select specific classes to visualize (leave empty to show all):"),
    class_selector,
    pn.pane.Markdown("## Function Selection"),
    include_functions_checkbox,
    pn.pane.Markdown(
        "Select specific functions to visualize (leave empty to show all):"
    ),
    function_selector,
    visualize_button,
    status_text,
    result_display,
    width=300,
)

# Initialize VTK pane
vtkpan = pn.pane.VTK(
    plotter.ren_win,
    sizing_mode="stretch_both",
    orientation_widget=True,
    enable_keybindings=True,
    min_height=400,
    min_width=400,
)
print("Initialized VTK pane")

# Create main layout
main_layout = pn.Row(input_panel, vtkpan)

# Serve the app
if __name__ == "__main__":
    main_layout.servable("Repository Visualizer")
else:
    main_layout.servable("Repository Visualizer")
