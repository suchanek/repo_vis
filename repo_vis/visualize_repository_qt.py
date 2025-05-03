# pylint: disable=C0301
# pylint: disable=C0116
# pylint: disable=C0115
# pylint: disable=W0105
# pylint: disable=W0613
# pylint: disable=W0612
# pylint: disable=W0603
# pylint: disable=E0611
# pylint: disable=C0413

"""
Module: visualize_repository_qt

This program provides a PyQt5-based application for visualizing the structure of a Python repository in 3D.
It uses PyVista for 3D rendering and PyQt5 for creating an interactive user interface. The application
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
 - python visualize_repository_qt.py

Author: Eric G. Suchanek, PhD
Last modified: 2025-05-03

"""

# -*- coding: utf-8 -*-

import ast
import logging
import os
import sys
from pathlib import Path

import numpy as np
import param
import pyvista as pv


def can_import(module_name):
    """Check we can import the requested module."""
    try:
        return __import__(module_name)
    except ImportError:
        return None


# Check if PyQt5 is installed
if can_import("PyQt5") is None:
    sys.exit(
        "This program requires PyQt5 to be installed. "
        "Please install it using: pip install proteusPy[pyqt5]"
    )

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera

# Set PyVista to use off-screen rendering only when required
pv.OFF_SCREEN = False
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
    level=logging.INFO,
    format=FORMAT,
    handlers=[rich_handler],
)
logger = logging.getLogger()

# Log PyVista and VTK versions
print("PyVista version:", pv.__version__)
print("VTK version:", pv.vtk_version_info)


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
    visualizer.plotter = pv.Plotter(off_screen=pv.OFF_SCREEN)
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
    window_title = param.String(
        default=f"{DEFAULT_PACKAGE_NAME} 3d visualization",
        doc="Window title for the application",
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
        self.plotter = pv.Plotter(off_screen=pv.OFF_SCREEN)
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

    def visualize(self, event=None, vtkpan=None):
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
            self.window_title = old_title
            print("Updating VTK pane")
            vtkpan.SetRenderWindow(self.plotter.ren_win)
            vtkpan.update()
            self.status = f"Visualization saved to {save_path}"
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
        """Set the window title reactively."""
        self.window_title = self.old_title
        print(f"Set window title: {self.window_title}")


class MainWindow(QMainWindow):
    status_changed = pyqtSignal(str)

    def __init__(self, visualizer):
        super().__init__()
        self.visualizer = visualizer
        self.setWindowTitle(self.visualizer.window_title)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Styling to match Panel's look and feel
        self.setStyleSheet(
            """
            QWidget {
                font-family: Arial, sans-serif;
                font-size: 12px;
            }
            QLineEdit, QComboBox, QListWidget, QCheckBox, QPushButton, QLabel {
                margin: 5px;
                padding: 5px;
            }
            QLineEdit, QComboBox, QListWidget {
                border: 1px solid #ddd;
                border-radius: 3px;
            }
            QPushButton {
                background-color: '#4CAF50';
                color: white;
                border: none;
                border-radius: 3px;
                padding: 8px;
            }
            QPushButton#reset {
                background-color: '#2196F3';
            }
            QLabel {
                border: 1px solid #ddd;
                background-color: '#f5f5f5';
                padding: 8px;
            }
        """
        )

        # Left control panel
        control_panel = QVBoxLayout()
        control_panel.setSpacing(10)
        control_panel.setContentsMargins(10, 10, 10, 10)

        # Input Parameters section
        control_panel.addWidget(
            QLabel("<h2>Input Parameters</h2>", font=QFont("Arial", 14, QFont.Bold))
        )

        # Repository Path
        repo_label = QLabel("Repository Path")
        control_panel.addWidget(repo_label)
        self.repo_path_input = QLineEdit(self.visualizer.repo_path)
        self.repo_path_input.setPlaceholderText(
            "Enter the path to the Python repository"
        )
        control_panel.addWidget(self.repo_path_input)

        # Save Path
        save_path_label = QLabel("Save Path")
        control_panel.addWidget(save_path_label)
        self.save_path_input = QLineEdit(self.visualizer.save_path)
        self.save_path_input.setPlaceholderText(
            "Enter the path to save the visualization"
        )
        control_panel.addWidget(self.save_path_input)

        # Save Format
        save_format_label = QLabel("Save Format")
        control_panel.addWidget(save_format_label)
        self.save_format_select = QComboBox()
        self.save_format_select.addItems(["html", "png", "jpeg"])
        self.save_format_select.setCurrentText(self.visualizer.save_format)
        control_panel.addWidget(self.save_format_select)

        # Class Radius
        class_radius_label = QLabel("Class Radius")
        control_panel.addWidget(class_radius_label)
        self.class_radius_slider = QSlider(Qt.Horizontal)
        self.class_radius_slider.setMinimum(10)  # 1.0
        self.class_radius_slider.setMaximum(100)  # 10.0
        self.class_radius_slider.setValue(int(self.visualizer.class_radius * 10))
        self.class_radius_slider.setTickInterval(5)
        self.class_radius_slider.setTickPosition(QSlider.TicksBelow)
        control_panel.addWidget(self.class_radius_slider)
        self.class_radius_value = QLabel(f"{self.visualizer.class_radius:.1f}")
        control_panel.addWidget(self.class_radius_value)

        # Member Radius Scale
        member_radius_label = QLabel("Member Radius Scale")
        control_panel.addWidget(member_radius_label)
        self.member_radius_scale_slider = QSlider(Qt.Horizontal)
        self.member_radius_scale_slider.setMinimum(5)  # 0.5
        self.member_radius_scale_slider.setMaximum(30)  # 3.0
        self.member_radius_scale_slider.setValue(
            int(self.visualizer.member_radius_scale * 10)
        )
        self.member_radius_scale_slider.setTickInterval(2)
        self.member_radius_scale_slider.setTickPosition(QSlider.TicksBelow)
        control_panel.addWidget(self.member_radius_scale_slider)
        self.member_radius_value = QLabel(f"{self.visualizer.member_radius_scale:.2f}")
        control_panel.addWidget(self.member_radius_value)

        # Class Selection
        control_panel.addWidget(
            QLabel("<h2>Class Selection</h2>", font=QFont("Arial", 14, QFont.Bold))
        )
        class_select_label = QLabel(
            "Select specific classes to visualize, empty for all:"
        )
        control_panel.addWidget(class_select_label)
        self.class_selector = QListWidget()
        self.class_selector.setSelectionMode(QListWidget.MultiSelection)
        for item in self.visualizer.available_classes:
            self.class_selector.addItem(item)
        control_panel.addWidget(self.class_selector)

        # Function Selection
        control_panel.addWidget(
            QLabel("<h2>Function Selection</h2>", font=QFont("Arial", 14, QFont.Bold))
        )
        self.include_functions_checkbox = QCheckBox(
            "Include Functions in Visualization"
        )
        self.include_functions_checkbox.setChecked(self.visualizer.include_functions)
        control_panel.addWidget(self.include_functions_checkbox)
        function_select_label = QLabel(
            "Select specific functions to visualize, empty for all:"
        )
        control_panel.addWidget(function_select_label)
        self.function_selector = QListWidget()
        self.function_selector.setSelectionMode(QListWidget.MultiSelection)
        for item in self.visualizer.available_functions:
            self.function_selector.addItem(item)
        control_panel.addWidget(self.function_selector)

        # Add stretch to push content up
        control_panel.addStretch()

        # Right visualization panel
        vis_panel = QVBoxLayout()
        vis_panel.setSpacing(10)
        vis_panel.setContentsMargins(10, 10, 10, 10)

        # Button row
        button_row = QHBoxLayout()
        self.visualize_button = QPushButton("Visualize Repository")
        self.visualize_button.setFixedWidth(200)
        button_row.addWidget(self.visualize_button)
        self.reset_camera_button = QPushButton("Reset Camera")
        self.reset_camera_button.setFixedWidth(200)
        self.reset_camera_button.setObjectName("reset")
        button_row.addWidget(self.reset_camera_button)
        self.status_display = QLabel("Ready")
        self.status_display.setTextInteractionFlags(Qt.TextBrowserInteraction)
        button_row.addWidget(self.status_display, stretch=1)
        vis_panel.addLayout(button_row)

        # VTK widget
        self.vtk_widget = QtInteractor(self)
        vis_panel.addWidget(self.vtk_widget, stretch=1)

        # Add layouts to main layout
        control_widget = QWidget()
        control_widget.setLayout(control_panel)
        control_widget.setFixedWidth(300)
        main_layout.addWidget(control_widget)
        vis_widget = QWidget()
        vis_widget.setLayout(vis_panel)
        main_layout.addWidget(vis_widget, stretch=1)

        # Connect signals
        self.repo_path_input.textChanged.connect(self.update_repo_path)
        self.save_path_input.textChanged.connect(self.update_save_path)
        self.save_format_select.currentTextChanged.connect(self.update_save_format)
        self.class_radius_slider.valueChanged.connect(self.update_class_radius)
        self.member_radius_scale_slider.valueChanged.connect(
            self.update_member_radius_scale
        )
        self.class_selector.itemSelectionChanged.connect(self.update_selected_classes)
        self.function_selector.itemSelectionChanged.connect(
            self.update_selected_functions
        )
        self.include_functions_checkbox.stateChanged.connect(
            self.update_include_functions
        )
        self.visualize_button.clicked.connect(
            lambda: self.visualizer.visualize(vtkpan=self.vtk_widget)
        )
        self.reset_camera_button.clicked.connect(self.reset_camera)
        self.status_changed.connect(self.update_status_display)
        self.visualizer.param.watch(self.on_status_change, "status")
        self.visualizer.param.watch(self.update_class_selector, "available_classes")
        self.visualizer.param.watch(
            self.update_function_selector, "available_functions"
        )
        self.visualizer.param.watch(self.update_window_title, "window_title")

    def update_repo_path(self, text):
        self.visualizer.repo_path = text

    def update_save_path(self, text):
        self.visualizer.save_path = text

    def update_save_format(self, text):
        self.visualizer.save_format = text

    def update_class_radius(self, value):
        self.visualizer.class_radius = value / 10.0
        self.class_radius_value.setText(f"{self.visualizer.class_radius:.1f}")

    def update_member_radius_scale(self, value):
        self.visualizer.member_radius_scale = value / 10.0
        self.member_radius_value.setText(f"{self.visualizer.member_radius_scale:.2f}")

    def update_selected_classes(self):
        selected = [item.text() for item in self.class_selector.selectedItems()]
        self.visualizer.selected_classes = selected

    def update_selected_functions(self):
        selected = [item.text() for item in self.function_selector.selectedItems()]
        self.visualizer.selected_functions = selected

    def update_include_functions(self, state):
        self.visualizer.include_functions = state == Qt.Checked

    def on_status_change(self, event):
        self.status_changed.emit(event.new)
        if event.new.startswith("Visualization saved to"):
            self.visualizer.set_window_title()

    def update_status_display(self, status):
        if status.startswith("Visualization saved to"):
            save_path = status.split("Visualization saved to ")[-1].strip()
            file_url = f"file://{os.path.abspath(save_path)}"
            self.status_display.setText(
                f"<b>Success!</b> Saved to: <span style='font-size:10px'>{save_path}</span> "
                f"<a href='{file_url}' style='font-size:10px; background-color:#4CAF50; color:white; text-decoration:none; "
                f"border-radius:3px; padding:3px 6px; display:inline-block;'>View</a>"
            )
        elif status.startswith("Error"):
            self.status_display.setText(
                f"<span style='color:#cc0000'><b>Error:</b> {status[6:]}</span>"
            )
        elif "Analyzing" in status or "Creating" in status:
            self.status_display.setText(
                f"<span style='color:#0066cc'><b>⏳ {status}</b></span>"
            )
        elif "Found" in status:
            parts = status.split("Found ")
            self.status_display.setText(
                f"<span style='color:#008800'><b>✓</b> Found {parts[1]}</span>"
            )
        else:
            self.status_display.setText(status)

    def update_class_selector(self, event):
        self.class_selector.clear()
        for item in event.new:
            self.class_selector.addItem(item)
        print("Updated class selector options:", event.new)

    def update_function_selector(self, event):
        self.function_selector.clear()
        for item in event.new:
            self.function_selector.addItem(item)
        print("Updated function selector options:", event.new)

    def update_window_title(self, event):
        self.setWindowTitle(event.new)
        print(f"Set window title: {event.new}")

    def reset_camera(self):
        print("Resetting camera...")
        self.vtk_widget.reset_camera()
        self.vtk_widget.render()


if __name__ == "__main__":
    app = QApplication([])
    visualizer = RepositoryVisualizer()
    window = MainWindow(visualizer)
    window.resize(1200, 800)
    window.show()
    app.exec_()
