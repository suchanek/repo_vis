# pylint: disable=C0301
# pylint: disable=C0116
# pylint: disable=C0115
# pylint: disable=W0105
# pylint: disable=W0613
# pylint: disable=W0612
# pylint: disable=W0603

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
Last modified: 2025-05-03

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
from rich.logging import RichHandler
from rich.progress import Progress

# Set PyVista to use off-screen rendering for WebGL compatibility
pv.OFF_SCREEN = True
if os.getenv("PYVISTA_OFF_SCREEN", "false").lower() == "true":
    pv.OFF_SCREEN = True

ORIGIN = (0, 0, 0)
DEFAULT_REP = "/Users/egs/repos/proteusPy"
DEFAULT_PACKAGE_NAME = os.path.basename(DEFAULT_REP)
DEFAULT_SAVE_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
DEFAULT_SAVE_NAME = f"{DEFAULT_PACKAGE_NAME}_3d_visualization"

# Configure logging with Rich
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
rich_handler = RichHandler(level=logging.INFO)
logging.basicConfig(
    level=logging.DEBUG,
    format=FORMAT,
    handlers=[rich_handler],
)
logger = logging.getLogger()

# Log PyVista and VTK versions
print("PyVista version:", pv.__version__)
print("VTK version:", pv.vtk_version_info)

pn.extension("vtk", sizing_mode="stretch_both", template="fast")


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
                            logger.debug(
                                f"Skipping duplicate class '{elem['name']}' in {file_path}"
                            )
                    elif elem["type"] == "function":
                        if elem["name"] not in seen_functions:
                            seen_functions.add(elem["name"])
                            elements.append(elem)
                        else:
                            logger.debug(
                                f"Skipping duplicate function '{elem['name']}' in {file_path}"
                            )
    logger.debug(
        f"Collected elements: {[e['name'] for e in elements if e['type'] == 'class']} (classes), {[e['name'] for e in elements if e['type'] == 'function']} (functions)"
    )
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

    if samples <= 0:
        return []

    if samples == 1:
        return [center + radius * np.array([0, 0, 1])]

    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius_at_y = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # Golden angle increment

        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y

        points.append(center + radius * np.array([x, y, z]))

    return points


def create_3d_visualization_for_panel(
    visualizer,
    elements,
    save_path,
    save_format="html",
    class_radius=4.0,
    member_radius_scale=1.0,
    old_title="",
):
    """Update the visualizer's plotter with a 3D visualization of the repository structure and handle screenshots with a separate off-screen plotter."""
    logger.debug("Creating visualization for %s", save_path)

    # Reinitialize the visualizer's plotter
    visualizer.plotter = pv.Plotter(off_screen=True)
    visualizer.plotter.disable_parallel_projection()
    visualizer.plotter.enable_anti_aliasing("msaa")

    visualizer.status = "Setting up visualization environment..."
    print("Reinitialized plotter")
    package_center = np.array([0, 0, 0])
    package_name = Path(save_path).stem
    package_size = 0.8
    package_mesh = pv.Icosahedron(
        center=package_center,
        radius=package_size,
    )
    visualizer.plotter.add_mesh(
        package_mesh, color="purple", show_edges=True, smooth_shading=True
    )

    num_classes = len([e for e in elements if e["type"] == "class"])
    visualizer.status = f"Rendering {num_classes} classes..."
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
        visualizer.plotter.add_mesh(
            mesh,
            color="red",
            show_edges=True,
            smooth_shading=False,
        )
        direction = pos - package_center
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)

        line = pv.Line(package_center, pos)
        visualizer.plotter.add_mesh(line, color="black", line_width=2)

    method_size = 0.3 * 0.75
    function_size = 0.3 * 0.5

    # Add standalone functions to the visualization
    num_functions = len([e for e in elements if e["type"] == "function"])
    if num_functions > 0:
        visualizer.status = f"Rendering {num_functions} functions..."
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
                visualizer.plotter.add_mesh(
                    mesh, color="green", show_edges=True, smooth_shading=True
                )

                line = pv.Line(package_center, pos)
                visualizer.plotter.add_mesh(line, color="lightgray", line_width=1)

                progress.update(task, advance=1)

    visualizer.status = "Rendering class methods..."
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
                    visualizer.plotter.add_mesh(
                        sphere, color="blue", show_edges=False, smooth_shading=True
                    )

                    line = pv.Line(class_pos, member_pos)
                    visualizer.plotter.add_mesh(line, color="gray", line_width=1)

                    progress.update(task, advance=1)

    visualizer.status = "Finalizing visualization scene..."
    visualizer.plotter.add_light(
        pv.Light(position=(50, 100, 100), color="white", intensity=1.0)
    )
    visualizer.plotter.add_light(
        pv.Light(position=(0, 0, 10), color="white", intensity=0.8)
    )

    num_classes = len([e for e in elements if e["type"] == "class"])
    num_methods = sum(
        len(e.get("methods", [])) for e in elements if e["type"] == "class"
    )
    num_functions = len([e for e in elements if e["type"] == "function"])
    title_text = f"3D Visualization: {package_name} | Classes: {num_classes} | Methods: {num_methods} | Functions: {num_functions}"
    visualizer.plotter.add_text(
        title_text,
        position="upper_edge",
        font_size=14,
        color="black",
    )
    visualizer.plotter.set_background("lightgray")
    visualizer.plotter.add_axes()

    # Simplified camera setup
    bounds = visualizer.plotter.bounds
    max_dim = max(
        abs(bounds[1] - bounds[0]),
        abs(bounds[3] - bounds[2]),
        abs(bounds[5] - bounds[4]),
        1.0,
    )
    distance_factor = 4.0
    visualizer.plotter.camera_position = [
        (
            distance_factor * max_dim,
            distance_factor * max_dim,
            distance_factor * max_dim,
        ),
        ORIGIN,
        (0, 0, 1),
    ]

    visualizer.plotter.enable_anti_aliasing("msaa")
    visualizer.plotter.render()

    # Save the visualization
    visualizer.status = "Saving visualization..."
    save_path = Path(save_path).with_suffix(f".{save_format}")
    try:
        if save_format == "html":
            visualizer.plotter.export_html(save_path)
            print("Saved HTML visualization to", save_path)
        elif save_format in ["png", "jpeg"]:
            screenshot_plotter = pv.Plotter(off_screen=True)
            screenshot_plotter.enable_anti_aliasing("msaa")
            print("Created off-screen plotter for screenshot")

            # Copy meshes from visualizer's plotter
            for actor in visualizer.plotter.actors.values():
                if isinstance(actor, pv.Actor):
                    if hasattr(actor, "mapper") and actor.mapper.GetInput():
                        mesh = actor.mapper.GetInput()
                        interpolation = actor.prop.GetInterpolation()
                        smooth_shading = interpolation > 0
                        screenshot_plotter.add_mesh(
                            mesh,
                            color=actor.prop.GetColor(),
                            show_edges=actor.prop.GetEdgeVisibility(),
                            line_width=actor.prop.GetLineWidth(),
                            smooth_shading=smooth_shading,
                        )

            # Copy lights
            for light in visualizer.plotter.renderer.GetLights():
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
            screenshot_plotter.camera_position = visualizer.plotter.camera_position
            screenshot_plotter.set_background("lightgray")

            # Render and save screenshot
            print("Saving screenshot with off-screen plotter")
            screenshot_plotter.screenshot(save_path, window_size=[1200, 800])
            screenshot_plotter.close()
            print("Saved screenshot to", save_path)
    except Exception as e:
        logger.error("Failed to save visualization: %s", e)
        visualizer.status = f"Error saving visualization: {str(e)}"
        if save_format in ["png", "jpeg"]:
            fallback_path = Path(save_path).with_suffix(".html")
            try:
                visualizer.plotter.export_html(fallback_path)
                logger.info(f"Saved fallback HTML visualization to {fallback_path}")
                visualizer.status = (
                    f"Saved fallback HTML visualization to {fallback_path}"
                )
            except Exception as fallback_error:
                logger.error(f"Fallback save also failed: {fallback_error}")

    return visualizer.plotter, title_text


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
        self.plotter = pv.Plotter(off_screen=True)
        self.plotter.add_mesh(
            pv.Icosahedron(center=ORIGIN, radius=1.0),
            color="purple",
        )
        self.plotter.set_background("lightgray")
        self.plotter.enable_anti_aliasing("msaa")
        self.plotter.disable_parallel_projection()
        self.elements = []
        self.update_classes()

    @param.depends("repo_path", watch=True)
    def update_classes(self):
        print("Updating classes and functions for repo_path:", self.repo_path)
        if os.path.exists(self.repo_path):
            self.status = "Analyzing repository..."
            self.elements = collect_elements(self.repo_path)

            class_names = sorted(
                [e["name"] for e in self.elements if e["type"] == "class"]
            )
            self.selected_classes = []
            self.available_classes = class_names
            self.param.selected_classes.objects = class_names

            function_names = sorted(
                [e["name"] for e in self.elements if e["type"] == "function"]
            )
            self.selected_functions = []
            self.available_functions = function_names
            self.param.selected_functions.objects = function_names

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

        filtered_elements = []
        if self.selected_classes:
            class_names = self.selected_classes
            for e in self.elements:
                if e["type"] == "class" and e["name"] in class_names:
                    filtered_elements.append(e)
        else:
            for e in self.elements:
                if e["type"] == "class":
                    filtered_elements.append(e)

        if self.include_functions:
            if self.selected_functions:
                function_names = self.selected_functions
                for e in self.elements:
                    if e["type"] == "function" and e["name"] in function_names:
                        filtered_elements.append(e)
            else:
                for e in self.elements:
                    if e["type"] == "function":
                        filtered_elements.append(e)

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
            self.plotter.clear()
            _pl, old_title = create_3d_visualization_for_panel(
                self,
                filtered_elements,
                save_path,
                self.save_format,
                self.class_radius,
                self.member_radius_scale,
                self.old_title,
            )
            self.old_title = old_title
            print("Updating VTK pane")
            vtkpan.object = self.plotter.ren_win
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

    def set_window_title(self):
        """
        Sets the window title using values from the global loader.
        """

        win_title = self.old_title
        pn.state.template.param.update(title=win_title)

        mess = f"Set Window Title: {win_title}"
        print(mess)


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


# Create a reset camera button
def reset_camera(event):
    print("Camera position before:", visualizer.plotter.camera_position)
    visualizer.plotter.disable_parallel_projection()
    bounds = visualizer.plotter.bounds
    max_dim = max(
        abs(bounds[1] - bounds[0]),
        abs(bounds[3] - bounds[2]),
        abs(bounds[5] - bounds[4]),
        1.0,
    )
    distance_factor = 4.0
    new_position = [
        (
            distance_factor * max_dim,
            distance_factor * max_dim,
            distance_factor * max_dim,
        ),
        ORIGIN,
        (0, 0, 1),
    ]
    # visualizer.plotter.camera_position = new_position
    visualizer.plotter.render()
    print("Camera position after:", visualizer.plotter.camera_position)
    vtkpan.object = visualizer.plotter.ren_win
    print("Camera reset with new position")


reset_camera_button = pn.widgets.Button(name="Reset Camera", button_type="primary")
reset_camera_button.on_click(reset_camera)

# Create a single-line status display
status_display = pn.pane.HTML(
    "Ready",
    height=40,
    margin=(5, 10),
)
status_display.styles = dict(
    border="1px solid #ddd",
    overflow_y="hidden",
    overflow_x="hidden",
    white_space="nowrap",
    text_overflow="ellipsis",
    padding="8px 12px",
    background_color="#f5f5f5",
    font_size="14px",
    box_sizing="border-box",
    line_height="24px",
)


# Update status display with formatted text
def update_status(event):
    status = event.new
    if status.startswith("Visualization saved to"):
        save_path = status.split("Visualization saved to ")[-1].strip()
        file_url = f"file://{os.path.abspath(save_path)}"
        status_display.object = (
            f"<b>Success!</b> Saved to: <span style='font-size:12px'>{save_path}</span> "
            f"<a href='{file_url}' target='_blank' style='font-size:12px; "
            f"background-color:#4CAF50; color:white; text-decoration:none; "
            f"border-radius:3px; padding:3px 6px; display:inline-block;'>View</a>"
        )
    elif status.startswith("Error"):
        status_display.object = (
            f"<span style='color:#cc0000'><b>Error:</b> {status[6:]}</span>"
        )
    elif "Analyzing" in status or "Creating" in status:
        status_display.object = f"<span style='color:#0066cc'><b>⏳ {status}</b></span>"
    elif "Found" in status:
        parts = status.split("Found ")
        status_display.object = (
            f"<span style='color:#008800'><b>✓</b> Found {parts[1]}</span>"
        )
    else:
        status_display.object = status
    visualizer.set_window_title()


visualizer.param.watch(update_status, "status")

# Create input panel
input_panel = pn.Column(
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
    width=300,
)

# Initialize VTK pane
vtkpan = pn.pane.VTK(
    visualizer.plotter.ren_win,
    sizing_mode="stretch_both",
    orientation_widget=True,
    enable_keybindings=True,
)
print("Initialized VTK pane")

# Set button width
visualize_button.width = 200
visualize_button.sizing_mode = "fixed"

# Status display takes remaining width
status_display.sizing_mode = "stretch_width"

# Set reset camera button width
reset_camera_button.width = 150
reset_camera_button.sizing_mode = "fixed"

# Create a row with buttons and status display
button_status_row = pn.Row(
    visualize_button,
    reset_camera_button,
    status_display,
    sizing_mode="stretch_width",
)

# Create visualization column
visualization_column = pn.Column(
    button_status_row,
    vtkpan,
    sizing_mode="stretch_both",
)

# Create main layout
main_layout = pn.Row(input_panel, visualization_column)

# Serve the app
if __name__ == "__main__":
    main_layout.servable("Repository Visualizer")
else:
    main_layout.servable("Repository Visualizer")
