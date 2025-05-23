# pylint: disable=C0301,C0116,C0115,W0613,E0611,C0413,E0401,W0601,W0621,C0302,E1101,C0103,W0718

"""
Module: pkg_visualizer

A PyQt5 application for 3D visualization of a Python package's structure.
Parses classes, methods, and functions, rendering them as 3D objects using PyVista.

Key Features:
- Extracts package structure using AST.
- Visualizes classes (green icosahedrons or cubes), methods (blue icosahedrons),
  and functions (red cylinders or cubes) with connections to the package center.
- Interactive UI for customizing and filtering visualizations by class, method, and function.
- Supports saving visualizations in multiple formats (HTML, PNG, JPEG).
- Enables picking of objects to display their docstrings in a popup window.
- Camera controls including reset view and animated spinning.
- Adaptive rendering based on package size (different shapes and connection types).
- Highlighting and zooming for selected elements.

Usage:
Run: python pkg_visualizer.py --package_path <path_to_repo> --save_path <path_to_save>

Author: Eric G. Suchanek, PhD
Last modified: 2025-05-21 23:03:42
"""

import argparse
import atexit
import gc
import logging
import os
import sys
import time
import types
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import param
import pyvista as pv
from find_red_bounding_box import remove_all_red_actors
from markdown import markdown
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

# from rich import print as rprint
from rich.logging import RichHandler
from utility import (
    can_import,
    collect_elements,
    fibonacci_sphere,
    format_docstring_to_markdown,
    set_pyvista_theme,
)

# Constants
__version__: str = "0.1.0"
__author__: str = "Eric G. Suchanek, PhD"
__revised__: str = "2025-05-21"

DEFAULT_TITLE: str = f"Python Package 3D Visualization v{__version__} {__author__}"

ORIGIN: Tuple[float, float, float] = (0, 0, 0)
DEFAULT_REP: str = "/Users/egs/repos/proteusPy"
DEFAULT_PACKAGE_NAME: str = os.path.basename(DEFAULT_REP)
DEFAULT_SAVE_PATH: str = os.path.join(os.path.expanduser("~"), "Desktop")
DEFAULT_SAVE_NAME: str = f"{DEFAULT_PACKAGE_NAME}_3d_visualization"

PACKAGE_RADIUS: float = 2.0
PACKAGE_COLOR: str = "purple"
PACKAGE_MESH: pv.PolyData = pv.Icosahedron(center=ORIGIN, radius=PACKAGE_RADIUS)

CLASS_OBJECT_RADIUS: float = 0.25 * PACKAGE_RADIUS
CLASS_COLOR: str = "green"
CLASS_CYLINDER_THRESHOLD: int = 500
CLASS_MAX_LINES: int = 1000
CLASS_MAX_ICOSAHEDRON = 2000

METHOD_OBJECT_RADIUS: float = 0.40 * CLASS_OBJECT_RADIUS
METHOD_COLOR: str = "blue"
METHOD_MAX_LINES = 2000

FUNCTION_OBJECT_RADIUS: float = 0.1 * PACKAGE_RADIUS
FUNCTION_RADIUS: float = 1.25 * PACKAGE_RADIUS
FUNCTION_COLOR: str = "red"
FUNCTION_MAX_CYLINDER: int = 1000

CYLINDER_RADIUS: float = 0.05
CYLINDER_RESOLUTION: int = 16
CYLINDER_COLOR: str = "gray"


BUTTON_WIDTH: int = 110

ZOOM_FACTOR: float = 5.0

# Configure logging
logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Patch PyVista's PolyData.__del__ method to prevent errors during exit
if hasattr(pv.PolyData, "__del__"):
    original_del = pv.PolyData.__del__

    def safe_del(self):
        """
        Safe version of PolyData.__del__ that catches and ignores errors.
        """
        try:
            original_del(self)
        except (TypeError, AttributeError, RuntimeError):
            # Ignore errors during deletion
            pass

    # Replace the original __del__ method with our safe version
    pv.PolyData.__del__ = safe_del
    logger.debug("Patched PyVista PolyData.__del__ method")

# Also patch MultiBlock.__del__ if it exists
if hasattr(pv.MultiBlock, "__del__"):
    original_multiblock_del = pv.MultiBlock.__del__

    def safe_multiblock_del(self):
        """
        Safe version of MultiBlock.__del__ that catches and ignores errors.
        """
        try:
            original_multiblock_del(self)
        except (TypeError, AttributeError, RuntimeError):
            # Ignore errors during deletion
            pass

    # Replace the original __del__ method with our safe version
    pv.MultiBlock.__del__ = safe_multiblock_del
    logger.debug("Patched PyVista MultiBlock.__del__ method")

if can_import("PyQt5") is None:
    sys.exit("This program requires PyQt5. Install: pip install proteusPy[pyqt5]")


class DocstringPopup(QDialog):
    def __init__(self, title: str, docstring: str, parent=None, on_close_callback=None):
        """
        Initialize the popup window to display a docstring.

        :param title: Title of the popup window.
        :type title: str
        :param docstring: The docstring to display.
        :type docstring: str
        :param parent: Parent widget.
        :type parent: QWidget
        :param on_close_callback: Callback function to execute when the window is closed.
        :type on_close_callback: callable or None
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(600, 400)
        self.on_close_callback = on_close_callback

        # Set the dialog to be modeless
        self.setWindowModality(Qt.NonModal)

        # Position the popup in the upper left of the screen
        if parent:
            screen_geometry = parent.screen().geometry()
            self.move(screen_geometry.x() + 50, screen_geometry.y() + 50)

        layout = QVBoxLayout(self)

        # Render the docstring as HTML using Markdown
        html_content = markdown(docstring or "No docstring available.")

        # Add a QTextBrowser to display the docstring
        text_browser = QTextBrowser(self)
        text_browser.setHtml(html_content)
        layout.addWidget(text_browser)

        # Add a close button
        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

    def closeEvent(self, event):
        """
        Handle the close event and trigger the callback if provided.

        :param event: The close event.
        :type event: QCloseEvent
        """
        if self.on_close_callback:
            self.on_close_callback()
        super().closeEvent(event)


def create_allium_visualization(
    viz_instance: "PackageVisualizer",
    elements: List[Dict[str, Union[str, int, List[str]]]],
    save_path: str,
    save_format: str = "html",
    class_radius: float = 4.0,
    member_radius_scale: float = 1.0,
    old_title: str = "",
    plotter: Optional[pv.Plotter] = None,
) -> Tuple[pv.Plotter, str, Dict[str, Dict[str, Union[str, object]]]]:
    """
    Create a 3D 'Allium' visualization of the package structure using PyVista.

    :param viz_instance: Instance of PackageVisualizer to update status.
    :type viz_instance: PackageVisualizer
    :param elements: List of elements (classes, functions) to visualize.
    :type elements: List[Dict[str, Union[str, int, List[str]]]]
    :param save_path: Path to save the visualization.
    :type save_path: str
    :param save_format: Format to save the visualization ('html', 'png', 'jpg').
    :type save_format: str
    :param class_radius: Radius for class visualization.
    :type class_radius: float
    :param member_radius_scale: Scale factor for member visualization.
    :type member_radius_scale: float
    :param old_title: Previous title of the visualization.
    :type old_title: str
    :param plotter: PyVista plotter instance. If None, a new one is created.
    :type plotter: Optional[pv.Plotter]
    :return: Tuple containing the plotter, title text, and mesh-to-element mapping.
    :rtype: Tuple[pv.Plotter, str, Dict[pv.PolyData, Dict[str, str]]]
    """
    viz_instance.status = "Setting up visualization..."
    QApplication.processEvents()

    actor_to_element: Dict[str, Dict[str, Union[str, object]]] = {}
    mesh_id_counter = 0  # Counter to generate unique IDs for meshes
    plotter.clear_actors()
    plotter.remove_all_lights()
    plotter.renderer_layer = 0
    plotter.disable_parallel_projection()
    plotter.enable_anti_aliasing("msaa")
    plotter.add_axes()

    # Add lights (unchanged)
    plotter.add_light(
        pv.Light(
            position=(80, 80, 80), focal_point=(0, 0, 0), color="white", intensity=0.9
        )
    )
    headlight = pv.Light(light_type="headlight", color="white", intensity=0.9)
    plotter.add_light(headlight)

    package_center: np.ndarray = np.array(ORIGIN)
    package_name: str = Path(save_path).stem
    package_radius: float = PACKAGE_RADIUS
    package_mesh: pv.PolyData = pv.Icosahedron(
        center=package_center, radius=package_radius
    )
    plotter.add_mesh(
        package_mesh,
        color="purple",
        show_edges=False,
        smooth_shading=False,
        name="package",
    )

    plotter.reset_camera()
    plotter.view_xy()
    plotter.camera.focal_point = ORIGIN

    num_classes: int = len([e for e in elements if e["type"] == "class"])
    num_functions: int = len([e for e in elements if e["type"] == "function"])
    total_methods: int = sum(
        len(class_elem.get("methods", []))
        for class_elem in [e for e in elements if e["type"] == "class"]
    )

    class_positions: List[np.ndarray] = fibonacci_sphere(
        num_classes, radius=class_radius, center=package_center
    )

    # Initialize MultiBlocks
    class_meshes = pv.MultiBlock()
    method_meshes = pv.MultiBlock()
    function_meshes = pv.MultiBlock()
    class_connection_meshes = pv.MultiBlock()  # MultiBlock for class connections
    function_connection_meshes = pv.MultiBlock()  # MultiBlock for function connections
    method_connection_meshes = pv.MultiBlock()  # MultiBlock for method connections

    # Render classes
    # rprint(f"[bold green]Starting to render {num_classes} classes...[/bold green]")
    logger.debug("Starting to render %s classes...", num_classes)
    class_index = 0

    for element in elements:
        if element["type"] != "class":
            continue
        pos: np.ndarray = class_positions[class_index]
        class_index += 1
        method_count: int = len(element.get("methods", []))
        if num_classes > CLASS_MAX_ICOSAHEDRON:
            mesh: pv.PolyData = pv.Cube(
                center=pos,
                x_length=viz_instance.class_object_radius * 2,
                y_length=viz_instance.class_object_radius * 2,
                z_length=viz_instance.class_object_radius * 2,
            )
        else:
            mesh: pv.PolyData = pv.Icosahedron(
                radius=viz_instance.class_object_radius,
                center=pos,
            )
        class_meshes.append(mesh)
        mesh_id = f"class_{mesh_id_counter}"
        mesh_id_counter += 1
        actor_to_element[mesh_id] = {
            "type": "class",
            "name": element["name"],
            "docstring": element.get("docstring", ""),
            "mesh": mesh,  # Store the mesh object in the value
        }
        # Calculate distance between package center and class position
        distance = np.linalg.norm(pos - package_center)

        if num_classes < CLASS_CYLINDER_THRESHOLD:
            # Draw cylinder for classes < CLASS_CYLINDER_THRESHOLD
            cylinder = pv.Cylinder(
                center=(package_center + pos) / 2,  # Midpoint between center and class
                direction=pos - package_center,  # Direction vector
                radius=CYLINDER_RADIUS,  # Cylinder radius
                height=distance,  # Cylinder height = distance
                resolution=CYLINDER_RESOLUTION,  # Number of sides
            )
            class_connection_meshes.append(cylinder)
        elif num_classes <= CLASS_MAX_LINES:
            # Draw line for distance between 501 and 2000
            line: pv.PolyData = pv.Line(package_center, pos)
            class_connection_meshes.append(line)
        # No connection if distance > 2000

        update_interval: int = max(1, int(num_classes * 0.10))
        if class_index % update_interval == 0 or class_index == num_classes:
            percent_complete = int((class_index / num_classes) * 100)
            progress_bar = "█" * (percent_complete // 10) + "░" * (
                (100 - percent_complete) // 10
            )
            viz_instance.status = f"Rendering Classes | {progress_bar} {percent_complete}% ({class_index}/{num_classes})"
            QApplication.processEvents()

    if class_meshes.n_blocks > 0:
        plotter.add_mesh(
            class_meshes,
            color="green",
            show_edges=False,
            smooth_shading=False,
            name="classes",
        )

    # Add the class connection meshes if any exist
    if class_connection_meshes.n_blocks > 0:
        plotter.add_mesh(
            class_connection_meshes,
            color=CYLINDER_COLOR,
            smooth_shading=True,
            name="class_connections",
        )

    plotter.view_xy()
    plotter.camera.focal_point = np.array(ORIGIN)
    # rprint("[bold green]Finished rendering classes![/bold green]")
    logger.debug("Finished rendering classes!")

    # Render functions !!!
    if num_functions > 0:
        viz_instance.status = f"Rendering {num_functions} functions..."
        QApplication.processEvents()
        function_radius_scale: float = np.sqrt(max(num_functions / 500.0, 1.0))
        function_positions: List[np.ndarray] = fibonacci_sphere(
            num_functions,
            radius=FUNCTION_RADIUS * function_radius_scale,
            center=package_center,
        )

        logger.debug("Starting to render %s functions...", num_functions)

        for i, element in enumerate([e for e in elements if e["type"] == "function"]):
            pos: np.ndarray = function_positions[i]
            if num_functions > FUNCTION_MAX_CYLINDER:
                mesh: pv.PolyData = pv.Cube(
                    center=pos, x_length=0.15, y_length=0.15, z_length=0.15
                )
            else:
                mesh: pv.PolyData = pv.Cylinder(
                    radius=FUNCTION_OBJECT_RADIUS,
                    height=FUNCTION_OBJECT_RADIUS,
                    center=pos,
                    direction=(0, 1, 1),
                    resolution=CYLINDER_RESOLUTION,
                )

            function_meshes.append(mesh)
            mesh_id = f"function_{mesh_id_counter}"
            mesh_id_counter += 1
            actor_to_element[mesh_id] = {
                "type": "function",
                "name": element["name"],
                "docstring": element.get("docstring", ""),
                "mesh": mesh,  # Store the mesh object in the value
            }
            if num_functions <= 1000:
                line: pv.PolyData = pv.Line(package_center, pos)
                function_connection_meshes.append(line)

            update_interval: int = max(1, int(num_functions * 0.10))
            if (i + 1) % update_interval == 0 or (i + 1) == num_functions:
                percent_complete = int(((i + 1) / num_functions) * 100)
                progress_bar = "█" * (percent_complete // 10) + "░" * (
                    (100 - percent_complete) // 10
                )
                viz_instance.status = f"Rendering Functions | {progress_bar} {percent_complete}% ({i+1}/{num_functions})"
                QApplication.processEvents()

        if function_meshes.n_blocks > 0:
            plotter.add_mesh(
                function_meshes,
                color=FUNCTION_COLOR,
                show_edges=False,
                smooth_shading=True,
                name="functions",
            )

        # Add the function connection meshes if any exist
        if function_connection_meshes.n_blocks > 0:
            plotter.add_mesh(
                function_connection_meshes,
                color=CYLINDER_COLOR,
                line_width=2,
                name="function_connections",
            )

        # rprint("[bold green]Finished rendering functions![/bold green]")
        logger.debug("Finished rendering functions!")

    # Render methods
    if viz_instance.render_methods and total_methods > 0:
        logger.debug("Starting to render %s methods...", total_methods)

        method_count: int = 0
        percent_complete = 0
        progress_bar = "░" * 10
        viz_instance.status = f"Rendering Methods | {progress_bar} {percent_complete}% (0/{total_methods})"
        QApplication.processEvents()

        for class_pos, class_elem in zip(
            class_positions, [e for e in elements if e["type"] == "class"]
        ):
            members: List[Dict[str, Union[str, int]]] = class_elem.get("methods", [])
            if members:
                method_positions: List[np.ndarray] = fibonacci_sphere(
                    len(members),
                    radius=member_radius_scale,
                    center=class_pos,
                )

                for j, member in enumerate(members):
                    method_mesh: pv.PolyData = pv.Icosahedron(
                        radius=viz_instance.method_object_radius,
                        center=method_positions[j],
                    )
                    method_meshes.append(method_mesh)
                    mesh_id = f"method_{mesh_id_counter}"
                    mesh_id_counter += 1
                    actor_to_element[mesh_id] = {
                        "type": "method",
                        "name": f"{class_elem['name']}.{member['name']}",
                        "docstring": member.get("docstring", ""),
                        "mesh": method_mesh,  # Store the mesh object in the value
                    }
                    if total_methods <= METHOD_MAX_LINES:
                        line: pv.PolyData = pv.Line(class_pos, method_positions[j])
                        method_connection_meshes.append(line)
                    method_count += 1

                update_interval: int = max(5, int(total_methods * 0.05))
                if (
                    method_count % update_interval == 0
                    or method_count == total_methods
                    or method_count <= 10
                ):
                    percent_complete = int((method_count / total_methods) * 100)
                    progress_bar = "█" * (percent_complete // 10) + "░" * (
                        (100 - percent_complete) // 10
                    )
                    viz_instance.status = f"Rendering Methods | {progress_bar} {percent_complete}% ({method_count}/{total_methods})"

                    QApplication.processEvents()

        if method_meshes.n_blocks > 0:
            plotter.add_mesh(
                method_meshes,
                color=METHOD_COLOR,
                show_edges=False,
                smooth_shading=False,
                name="methods",
            )

        # Add the method connection meshes if any exist
        if method_connection_meshes.n_blocks > 0:
            plotter.add_mesh(
                method_connection_meshes,
                color=METHOD_COLOR,
                line_width=1,
                name="method_connections",
            )

        # rprint("[bold green]Finished rendering methods![/bold green]")
        logger.debug("Finished rendering methods!")

    plotter.view_xy()
    plotter.reset_camera()
    plotter.camera.focal_point = [0, 0, 0]

    num_methods: int = sum(
        len(e.get("methods", [])) for e in elements if e["type"] == "class"
    )
    # Track the total number of triangles in the visualization
    total_triangles = 0

    # Update triangle count for class meshes
    for i in range(class_meshes.n_blocks):
        total_triangles += class_meshes[i].n_faces_strict

    # Update triangle count for class connection meshes
    for i in range(class_connection_meshes.n_blocks):
        total_triangles += class_connection_meshes[i].n_faces_strict

    # Update triangle count for function meshes
    for i in range(function_meshes.n_blocks):
        total_triangles += function_meshes[i].n_faces_strict

    # Update triangle count for function connection meshes
    for i in range(function_connection_meshes.n_blocks):
        total_triangles += function_connection_meshes[i].n_faces_strict

    # Update triangle count for method connection meshes
    for i in range(method_connection_meshes.n_blocks):
        total_triangles += method_connection_meshes[i].n_faces_strict

    # Update triangle count for method meshes
    for i in range(method_meshes.n_blocks):
        total_triangles += method_meshes[i].n_faces_strict

    logger.debug("Total number of triangles in the visualization: %d", total_triangles)

    # Store the total triangles count in the visualizer instance
    viz_instance.num_faces = total_triangles

    title_text: str = (
        f"3D Visualization: {package_name} | Classes: {num_classes} | Methods: {num_methods} | Functions: {num_functions} | Faces: {total_triangles}"
    )

    plotter.render()
    viz_instance.status = "Scene generation complete."
    QApplication.processEvents()

    return plotter, title_text, actor_to_element


class PackageVisualizer(param.Parameterized):
    package_path: str = param.String(default=DEFAULT_REP, doc="Package path")
    save_path: str = param.String(default=DEFAULT_PACKAGE_NAME, doc="Save path")
    old_title: str = param.String(
        default=DEFAULT_TITLE, doc="Previous title for the plotter"
    )
    window_title: str = param.String(
        default=f"{DEFAULT_PACKAGE_NAME} 3d visualization", doc="Window title"
    )
    save_format: str = param.Selector(
        objects=["html", "png", "jpg"], default="html", doc="Output format"
    )
    class_radius: float = param.Number(
        default=5.5, bounds=(2.0, 75.0), step=0.5, doc="Class radius"
    )
    member_radius_scale: float = param.Number(
        default=1.0, doc="Member radius scale (static)"
    )
    class_object_radius: float = param.Number(
        default=0.75,
        doc="Radius of the class object, relative to the package object radius",
    )
    method_object_radius: float = param.Number(
        default=METHOD_OBJECT_RADIUS,
        doc="Radius of the method object, relative to the class object radius",
    )
    available_classes: List[str] = param.List(default=[], doc="Available classes")
    selected_classes: List[str] = param.ListSelector(
        default=[], objects=[], doc="Selected classes"
    )
    available_methods: List[str] = param.List(default=[], doc="Available methods")
    selected_methods: List[str] = param.ListSelector(
        default=[], objects=[], doc="Selected methods"
    )
    available_functions: List[str] = param.List(default=[], doc="Available functions")
    selected_functions: List[str] = param.ListSelector(
        default=[], objects=[], doc="Selected functions"
    )
    include_functions: bool = param.Boolean(default=False, doc="Include functions")
    render_methods: bool = param.Boolean(default=True, doc="Render methods")
    status: str = param.String(default="Ready", doc="Status")
    num_classes: int = param.Integer(default=0, doc="Number of classes in the package")
    num_functions: int = param.Integer(
        default=0, doc="Number of functions in the package"
    )
    num_methods: int = param.Integer(default=0, doc="Number of methods in the package")
    num_faces: int = param.Integer(
        default=0, doc="Number of faces (triangles) in the visualization"
    )

    def __init__(self, plotter: Optional[pv.Plotter] = None, **params: dict) -> None:
        """
        Initialize the PackageVisualizer.
        """
        super().__init__(**params)
        self.plotter: Optional[pv.Plotter] = plotter or pv.Plotter()
        self.elements: List[Dict[str, Union[str, int, List[str]]]] = []
        self.actor_to_element: Dict[str, Dict[str, Union[str, object]]] = {}
        self.update_classes()

    def set_plotter(self, plotter: pv.Plotter) -> None:
        self.plotter = plotter

    @param.depends("package_path", watch=True)
    def update_classes(self) -> bool:
        """
        Update the list of available classes, functions, and methods based on the package path.
        """
        self.window_title = DEFAULT_TITLE
        if os.path.exists(self.package_path):
            self.status = "Analyzing package..."
            self.elements = collect_elements(self.package_path)

            # Compute the number of classes, functions, and methods
            self.num_classes = len([e for e in self.elements if e["type"] == "class"])
            self.num_functions = len(
                [e for e in self.elements if e["type"] == "function"]
            )
            self.num_methods = sum(
                len(e.get("methods", [])) for e in self.elements if e["type"] == "class"
            )

            # Update available classes, methods, and functions
            class_names: List[str] = sorted(
                [e["name"] for e in self.elements if e["type"] == "class"]
            )
            self.available_classes = class_names
            self.param.selected_classes.objects = class_names

            # Collect method names in format "ClassName.method_name"
            method_names: List[str] = []
            for e in self.elements:
                if e["type"] == "class" and "methods" in e:
                    for method in e["methods"]:
                        method_names.append(f"{e['name']}.{method['name']}")
            method_names.sort()
            self.available_methods = method_names
            self.param.selected_methods.objects = method_names

            function_names: List[str] = sorted(
                [e["name"] for e in self.elements if e["type"] == "function"]
            )
            self.available_functions = function_names
            self.param.selected_functions.objects = function_names

            self.save_path = os.path.basename(self.package_path)
            self.status = f"Package loaded: {self.package_path}"

            # Update the window title
            self.window_title = f"Pkg: {self.package_path} | Classes: {self.num_classes} | Functions: {self.num_functions} | Methods: {self.num_methods} | Faces: {self.num_faces}"

            if self.plotter:
                # Check if it's a PyVista plotter with clear_actors method
                if hasattr(self.plotter, "clear_actors"):
                    self.plotter.clear_actors()
                return True
        else:
            self.reset_elements()
            self.status = "Error: Package path does not exist!"
            self.window_title = DEFAULT_TITLE
            # Clear the plotter to remove old meshes when an invalid path is entered
            if self.plotter and hasattr(self.plotter, "clear_actors"):
                self.plotter.clear_actors()
            return False

    def reset_elements(self) -> None:
        """
        Reset the elements and available classes, methods, and functions.
        """
        self.available_classes = []
        self.selected_classes = []
        self.param.selected_classes.objects = []
        self.available_functions = []
        self.selected_functions = []
        self.param.selected_functions.objects = []
        self.available_methods = []
        self.selected_methods = []
        self.param.selected_methods.objects = []
        self.elements = []

    def log_actor_to_element(self) -> None:
        """
        Log the actor_to_element dictionary for debugging purposes.
        """
        logger.debug("Logging actor_to_element dictionary")
        logger.debug("Number of elements: %d", len(self.actor_to_element))

        # Count elements by type
        type_counts = {"class": 0, "method": 0, "function": 0}
        for mesh_id, elem_data in self.actor_to_element.items():
            elem_type = elem_data.get("type")
            if elem_type in type_counts:
                type_counts[elem_type] += 1

        logger.debug("Element counts by type: %s", type_counts)

        # Log some sample elements of each type
        for elem_type in ["class", "method", "function"]:
            logger.debug("Sample elements of type %s:", elem_type)
            count = 0
            for mesh_id, elem_data in self.actor_to_element.items():
                if elem_data.get("type") == elem_type:
                    mesh = elem_data.get("mesh")
                    center = mesh.center if mesh else "No mesh"
                    logger.debug(
                        "  %s: %s (center: %s)", mesh_id, elem_data.get("name"), center
                    )
                    count += 1
                    if count >= 5:  # Log at most 5 elements of each type
                        break

    def visualize(self) -> None:
        """
        Create and display the 3D visualization based on selected parameters.
        """
        if not os.path.exists(self.package_path):
            self.status = "Error: Package path does not exist."
            return
        if not self.elements:
            self.status = "No elements found."
            return

        filtered_elements: List[Dict[str, Union[str, int, List[str]]]] = []

        # Filter classes
        if self.selected_classes:
            for e in self.elements:
                if e["type"] == "class" and e["name"] in self.selected_classes:
                    # If we have selected methods, filter the methods in this class
                    if self.selected_methods and "methods" in e:
                        # Create a copy of the class element to modify
                        class_copy = e.copy()
                        # Filter methods based on selection
                        filtered_methods = []
                        for method in e["methods"]:
                            method_full_name = f"{e['name']}.{method['name']}"
                            if method_full_name in self.selected_methods:
                                filtered_methods.append(method)
                        # Replace methods with filtered list
                        class_copy["methods"] = filtered_methods
                        filtered_elements.append(class_copy)
                    else:
                        # No method filtering, add the class as is
                        filtered_elements.append(e)
        else:
            # No class filtering, but we might still need to filter methods
            for e in self.elements:
                if e["type"] == "class":
                    if self.selected_methods and "methods" in e:
                        # Create a copy of the class element to modify
                        class_copy = e.copy()
                        # Filter methods based on selection
                        filtered_methods = []
                        for method in e["methods"]:
                            method_full_name = f"{e['name']}.{method['name']}"
                            if method_full_name in self.selected_methods:
                                filtered_methods.append(method)
                        # Replace methods with filtered list
                        class_copy["methods"] = filtered_methods
                        filtered_elements.append(class_copy)
                    else:
                        # No method filtering, add the class as is
                        filtered_elements.append(e)

        # Filter functions
        if self.include_functions:
            if self.selected_functions:
                for e in self.elements:
                    if e["type"] == "function" and e["name"] in self.selected_functions:
                        filtered_elements.append(e)
            else:
                filtered_elements.extend(
                    [e for e in self.elements if e["type"] == "function"]
                )

        save_path: str = self.save_path
        if not save_path.endswith(f".{self.save_format}"):
            save_path = f"{save_path}.{self.save_format}"

        try:
            _, title_text, actor_to_element = create_allium_visualization(
                self,
                filtered_elements,
                save_path,
                self.save_format,
                self.class_radius,
                self.member_radius_scale,
                self.old_title,
                self.plotter,
            )
            self.old_title = title_text
            self.window_title = title_text
            self.actor_to_element = actor_to_element

            # Log the actor_to_element dictionary for debugging
            self.log_actor_to_element()

        except (ValueError, RuntimeError) as e:
            self.status = f"Error creating visualization: {str(e)}"


class MainWindow(QMainWindow):
    status_changed: pyqtSignal = pyqtSignal(str)

    def __init__(
        self,
        package_path: str = DEFAULT_REP,
        save_path: str = DEFAULT_PACKAGE_NAME,
        width: int = 1200,
        height: int = 800,
    ) -> None:
        """
        Initialize the main window for the visualization application.

        :param package_path: Path to the package to visualize
        :type package_path: str
        :param save_path: Path to save the visualization
        :type save_path: str
        :param width: Width of the window
        :type width: int
        :param height: Height of the window
        :type height: int
        """
        super().__init__()
        self.timer = None
        self.current_frame = 0
        self.spin_count = 0
        self.status = "Ready"

        # Create the PackageVisualizer internally
        set_pyvista_theme("auto", verbose=True)
        self.vtk_plotter: QtInteractor = QtInteractor(self, theme=pv._GlobalTheme())
        self.visualizer: PackageVisualizer = PackageVisualizer(
            plotter=self.vtk_plotter, package_path=package_path, save_path=save_path
        )

        self.setWindowTitle(self.visualizer.window_title)
        self._current_picked_actor: Optional[pv.Actor] = None
        self._current_popup: Optional[DocstringPopup] = None
        self._original_camera_state = None  # Store original camera state before zooming

        central_widget: QWidget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout: QHBoxLayout = QHBoxLayout(central_widget)

        self.setStyleSheet(
            """
            QWidget { font-family: Arial; font-size: 12px; }
            QLineEdit, QComboBox, QListWidget, QCheckBox, QPushButton, QLabel { margin: 2px; padding: 3px; }
            QLineEdit, QComboBox, QListWidget { border: 1px solid #ddd; border-radius: 3px; }
            QPushButton { background-color: '#4CAF50'; color: white; border: none; border-radius: 3px; padding: 6px; }
            QPushButton#reset { background-color: '#FFEB3B'; color: black; }
            QPushButton#reset-view { background-color: '#FFEB3B'; color: black; }
            QLabel { border: 1px solid #ddd; background-color: '#f5f5f5'; padding: 4px; }
            """
        )

        control_panel: QVBoxLayout = QVBoxLayout()
        control_panel.setSpacing(5)
        control_panel.setContentsMargins(6, 6, 6, 6)

        control_panel.addWidget(
            QLabel(
                "<h2>Input Parameters</h2>",
                font=QFont("Arial", 14, QFont.Bold),
                styleSheet="background: transparent; border: none;",
            )
        )

        package_path_label: QLabel = QLabel(
            "<b style='font-size:13px;'>Package Path</b>"
        )
        control_panel.addWidget(package_path_label)
        self.package_path_input: QLineEdit = QLineEdit(self.visualizer.package_path)
        self.package_path_input.setPlaceholderText("Enter package path")
        control_panel.addWidget(self.package_path_input)

        save_path_label: QLabel = QLabel("<b style='font-size:13px;'>Save Path</b>")
        control_panel.addWidget(save_path_label)
        self.save_path_input: QLineEdit = QLineEdit(self.visualizer.save_path)
        self.save_path_input.setPlaceholderText("Enter save path")
        control_panel.addWidget(self.save_path_input)

        save_format_label: QLabel = QLabel("<b style='font-size:13px;'>Save Format</b>")
        control_panel.addWidget(save_format_label)
        self.save_format_select: QComboBox = QComboBox()
        self.save_format_select.addItems(["html", "png", "jpg"])
        self.save_format_select.setCurrentText(self.visualizer.save_format)
        control_panel.addWidget(self.save_format_select)

        vis_params_label: QLabel = QLabel(
            "<b style='font-size:13px;'>Visualization Parameters</b>"
        )
        control_panel.addWidget(vis_params_label)

        control_panel.addWidget(QLabel("Class Radius"))
        self.class_radius_slider: QSlider = QSlider(Qt.Horizontal)
        self.class_radius_slider.setMinimum(80)
        self.class_radius_slider.setMaximum(1000)
        self.class_radius_slider.setValue(int(self.visualizer.class_radius * 20))
        self.class_radius_slider.setTickInterval(71)
        self.class_radius_slider.setTickPosition(QSlider.TicksBelow)
        control_panel.addWidget(self.class_radius_slider)

        control_panel.addWidget(
            QLabel(
                "<h2>Class Selection</h2>",
                font=QFont("Arial", 14, QFont.Bold),
                styleSheet="background: transparent; border: none;",
            )
        )
        control_panel.addWidget(QLabel("Select classes (empty for all):"))
        self.class_selector: QListWidget = QListWidget()
        self.class_selector.setSelectionMode(QListWidget.MultiSelection)
        self.class_selector.setMaximumHeight(80)
        for item in self.visualizer.available_classes:
            self.class_selector.addItem(item)
        control_panel.addWidget(self.class_selector)

        control_panel.addWidget(
            QLabel(
                "<h2>Method Selection</h2>",
                font=QFont("Arial", 14, QFont.Bold),
                styleSheet="background: transparent; border: none;",
            )
        )
        control_panel.addWidget(QLabel("Select methods (empty for all):"))
        self.method_selector: QListWidget = QListWidget()
        self.method_selector.setSelectionMode(QListWidget.MultiSelection)
        self.method_selector.setMaximumHeight(80)
        for item in self.visualizer.available_methods:
            self.method_selector.addItem(item)
        control_panel.addWidget(self.method_selector)

        control_panel.addWidget(
            QLabel(
                "<h2>Function Selection</h2>",
                font=QFont("Arial", 14, QFont.Bold),
                styleSheet="background: transparent; border: none;",
            )
        )
        for widget in control_panel.findChildren(QLabel):
            widget.setStyleSheet("background: transparent; border: none;")

        control_panel.addWidget(QLabel("Select functions (empty for all):"))

        self.function_selector: QListWidget = QListWidget()
        self.function_selector.setSelectionMode(QListWidget.MultiSelection)
        self.function_selector.setMaximumHeight(80)
        for item in self.visualizer.available_functions:
            self.function_selector.addItem(item)
        control_panel.addWidget(self.function_selector)

        # Add a label for render options
        render_options_label = QLabel(
            "<h2>Render Options</h2>",
            font=QFont("Arial", 14, QFont.Bold),
            styleSheet="background: transparent; border: none;",
        )
        control_panel.addWidget(render_options_label)

        # Create a horizontal layout for the checkboxes
        checkbox_layout = QHBoxLayout()

        self.include_functions_checkbox: QCheckBox = QCheckBox("Render Functions")
        self.include_functions_checkbox.setChecked(self.visualizer.include_functions)
        checkbox_layout.addWidget(self.include_functions_checkbox)

        self.render_methods_checkbox: QCheckBox = QCheckBox("Render Methods")
        self.render_methods_checkbox.setChecked(self.visualizer.render_methods)
        checkbox_layout.addWidget(self.render_methods_checkbox)

        # Add the checkbox layout to the control panel
        control_panel.addLayout(checkbox_layout)

        # Set initial enabled state for selectors
        self.function_selector.setEnabled(self.visualizer.include_functions)
        self.method_selector.setEnabled(self.visualizer.render_methods)

        # Add stretch to push the visualize button to the bottom
        control_panel.addStretch()

        self.visualize_button: QPushButton = QPushButton("Visualize Package")
        control_panel.addWidget(self.visualize_button)

        vis_panel: QVBoxLayout = QVBoxLayout()
        vis_panel.setSpacing(10)
        vis_panel.setContentsMargins(10, 10, 10, 10)

        button_row: QHBoxLayout = QHBoxLayout()

        self.button_spin = QPushButton("Spin Package")
        self.button_spin.clicked.connect(self.spin_camera)
        self.button_spin.setFixedWidth(BUTTON_WIDTH)
        self.button_spin.setStyleSheet("background-color: '#2196F3'; color: white;")
        button_row.addWidget(self.button_spin)

        self.save_button: QPushButton = QPushButton("Save View")
        self.save_button.setFixedWidth(BUTTON_WIDTH)
        button_row.addWidget(self.save_button)

        self.reset_view_button: QPushButton = QPushButton("Reset View")
        self.reset_view_button.setFixedWidth(BUTTON_WIDTH)
        self.reset_view_button.setStyleSheet(
            "background-color: '#FFEB3B'; color: black;"
        )
        button_row.addWidget(self.reset_view_button)

        self.reset_settings_button: QPushButton = QPushButton("Reset Settings")
        self.reset_settings_button.setFixedWidth(BUTTON_WIDTH)
        self.reset_settings_button.setObjectName("reset-view")
        self.reset_settings_button.setStyleSheet(
            "background-color: '#FF0000'; color: white;"
        )
        button_row.addWidget(self.reset_settings_button)

        self.save_button.clicked.connect(self.save_current_view)
        self.reset_view_button.clicked.connect(self.reset_camera)

        self.status_display: QLabel = QLabel("Ready")
        self.status_display.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.status_display.setStyleSheet("font-weight: bold; font-size: 14px;")
        button_row.addWidget(self.status_display, stretch=1)

        # Store a reference to the plotter for easier access
        self.plotter = self.vtk_plotter

        vis_panel.addWidget(self.vtk_plotter, stretch=1)
        vis_panel.addLayout(button_row)

        # Create widgets to hold the layouts
        control_widget: QWidget = QWidget()
        control_widget.setLayout(control_panel)
        control_widget.setFixedWidth(375)

        vis_widget: QWidget = QWidget()
        vis_widget.setLayout(vis_panel)

        # Add widgets to main layout
        main_layout.addWidget(control_widget)
        main_layout.addWidget(vis_widget, stretch=1)

        # Configure layout properties
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 3)

        # Set size policies
        central_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        control_widget.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.MinimumExpanding
        )
        control_panel.setSizeConstraint(QVBoxLayout.SetMinimumSize)

        self.vtk_plotter.enable_mesh_picking(
            callback=self.on_pick,
            show=False,
            show_actors=False,
            show_message=False,
            font_size=14,
            left_clicking=False,
            use_actor=True,  # !!!
            through=True,
        )

        if hasattr(self.vtk_plotter, "picker"):
            logger.debug("Configuring VTK picker")
            picker = self.vtk_plotter.picker
            picker.SetTolerance(0.005)  # Adjust tolerance for better precision
            # Avoid restricting to pick list to ensure MultiBlock picking works
            picker.SetPickFromList(0)  # Disable pick list restriction

            # Ensure the picker is set up for mesh picking
            if hasattr(picker, "SetPickable"):
                for actor_name in ["classes", "methods", "functions"]:
                    actor = self.vtk_plotter.actors.get(actor_name)
                    if actor:
                        picker.SetPickable(actor, 1)  # Make sure actors are pickable
                        logger.debug("Set actor %s as pickable", actor_name)

        # Log picker configuration
        logger.debug("Picker type: %s", type(picker).__name__)
        logger.debug("Picker tolerance: %f", picker.GetTolerance())
        logger.debug("Pick from list: %d", picker.GetPickFromList())

        self.package_path_input.editingFinished.connect(self.update_package_path)
        self.save_path_input.textChanged.connect(self.update_save_path)
        self.save_format_select.currentTextChanged.connect(self.update_save_format)
        self.class_radius_slider.sliderReleased.connect(
            lambda: self.update_class_radius(self.class_radius_slider.value())
        )
        self.class_selector.itemSelectionChanged.connect(self.update_selected_classes)
        self.method_selector.itemSelectionChanged.connect(self.update_selected_methods)
        self.function_selector.itemSelectionChanged.connect(
            self.update_selected_functions
        )
        self.include_functions_checkbox.stateChanged.connect(
            self.update_include_functions
        )
        self.render_methods_checkbox.stateChanged.connect(self.update_render_methods)
        self.visualize_button.clicked.connect(self.on_visualize_clicked)
        self.reset_settings_button.clicked.connect(self.reset_settings)
        self.status_changed.connect(self.update_status_display)
        self.visualizer.param.watch(self.on_status_change, "status")
        self.visualizer.param.watch(self.update_class_selector, "available_classes")
        self.visualizer.param.watch(self.update_method_selector, "available_methods")
        self.visualizer.param.watch(
            self.update_function_selector, "available_functions"
        )
        self.visualizer.param.watch(self.update_window_title, "window_title")

        for widget in self.findChildren(QLabel):
            widget.setStyleSheet("background: transparent; border: none;")

        self.class_selector.itemClicked.connect(self.show_class_docstring)
        self.method_selector.itemClicked.connect(self.show_method_docstring)
        self.function_selector.itemClicked.connect(self.show_function_docstring)

        self.class_radius_slider.setValue(int(self.visualizer.class_radius * 20))
        font = QFont("Arial", 12)
        self.setFont(font)
        self.resize(width, height)

        # Disable bounding box generation in the plotter
        # disable_bounding_box_generation(self.plotter)

        return

    def show_class_docstring(self, item) -> None:
        """
        Show the docstring for the selected class in a popup and highlight the mesh.
        """
        # Close any existing popup
        if self._current_popup is not None and self._current_popup.isVisible():
            self._current_popup.close()
            self._current_popup = None

        class_name = item.text()
        class_element = next(
            (
                e
                for e in self.visualizer.elements
                if e["type"] == "class" and e["name"] == class_name
            ),
            None,
        )
        if class_element:
            mesh_entry = next(
                (
                    (mesh_id, elem)
                    for mesh_id, elem in self.visualizer.actor_to_element.items()
                    if elem["type"] == "class" and elem["name"] == class_name
                ),
                None,
            )
            if mesh_entry:
                mesh_id, elem = mesh_entry
                self.highlight_actor(elem["mesh"])

            self._current_popup = DocstringPopup(
                f"Class: {class_name}",
                format_docstring_to_markdown(class_element.get("docstring", "")),
                self,
                on_close_callback=self.reset_picking_state,
            )
            self._current_popup.show()

    def show_function_docstring(self, item) -> None:
        """
        Show the docstring for the selected function in a popup and highlight the mesh.
        """
        # Close any existing popup
        if self._current_popup is not None and self._current_popup.isVisible():
            self._current_popup.close()
            self._current_popup = None

        function_name = item.text()
        function_element = next(
            (
                e
                for e in self.visualizer.elements
                if e["type"] == "function" and e["name"] == function_name
            ),
            None,
        )
        if function_element:
            mesh_entry = next(
                (
                    (mesh_id, elem)
                    for mesh_id, elem in self.visualizer.actor_to_element.items()
                    if elem["type"] == "function" and elem["name"] == function_name
                ),
                None,
            )
            if mesh_entry:
                mesh_id, elem = mesh_entry
                self.highlight_actor(elem["mesh"])

            self._current_popup = DocstringPopup(
                f"Function: {function_name}",
                format_docstring_to_markdown(function_element.get("docstring", "")),
                self,
                on_close_callback=self.reset_picking_state,
            )
            self._current_popup.show()

    def highlight_actor(self, mesh):
        """
        Highlight the given mesh by creating a temporary actor, adjust camera to focus on it,
        and zoom in for a better view using PyVista's built-in camera zoom.

        :param mesh: The mesh to highlight.
        :type mesh: pv.PolyData
        """
        if not self.visualizer.plotter:
            logger.error("Plotter is not initialized.")
            return
        self.reset_actor_appearances()

        # Reset camera to default position before zooming
        self.reset_camera()

        # Add the mesh with specific settings to prevent bounding box
        highlight_actor = self.plotter.add_mesh(
            mesh,
            color="pink",
            show_edges=True,
            edge_color="white",
            line_width=3,
            pickable=False,  # Make it not pickable to avoid bounding box issues
            render_points_as_spheres=False,  # Disable point rendering
            render_lines_as_tubes=False,  # Disable line rendering as tubes
            show_scalar_bar=False,  # Hide scalar bar
            reset_camera=False,  # Don't reset camera
            culling=False,  # Disable culling
            lighting=True,  # Enable lighting
        )
        self._current_picked_actor = highlight_actor

        # Store original camera state after reset
        self._original_camera_state = {
            "position": self.plotter.camera.position,
            "focal_point": self.plotter.camera.focal_point,
            "view_up": self.plotter.camera.up,
            "distance": self.plotter.camera.distance,
        }
        logger.debug("Original camera state saved: %s", self._original_camera_state)

        # Get the mesh center
        mesh_center = np.array(mesh.center)

        # Set camera focal point to mesh center
        self.plotter.camera.focal_point = mesh_center

        # Apply zoom using PyVista's built-in camera zoom
        # Zoom factor > 1 zooms in, < 1 zooms out
        self.plotter.camera.Zoom(ZOOM_FACTOR)

        # Render the scene with the new camera settings
        self.plotter.render()

        # Log camera adjustment
        logger.debug(
            "Camera adjusted to focus on object at %s with zoom factor %s",
            mesh_center,
            ZOOM_FACTOR,
        )

    def log_plotter_actors(self):
        """
        Log detailed information about plotter actors for debugging.
        """
        logger.debug("Logging plotter actors")
        logger.debug("Number of actors: %d", len(self.plotter.actors))

        for name, actor in self.plotter.actors.items():
            logger.debug("Actor: %s", name)
            if hasattr(actor, "GetProperty"):
                color = actor.GetProperty().GetColor()
                logger.debug("  Color: %s", color)
                if color == (1.0, 0.0, 0.0):  # Red actor
                    logger.debug("  ** RED ACTOR DETECTED **")
            if hasattr(actor, "GetMapper") and actor.GetMapper():
                mapper = actor.GetMapper()
                if hasattr(mapper, "GetInput") and mapper.GetInput():
                    input_obj = mapper.GetInput()
                    input_type = type(input_obj).__name__
                    logger.debug("  Input type: %s", input_type)
                    if hasattr(input_obj, "bounds"):
                        logger.debug("  Bounds: %s", input_obj.bounds)
                    if isinstance(input_obj, pv.MultiBlock):
                        logger.debug("  MultiBlock with %d blocks", input_obj.n_blocks)

    def on_pick(self, actor):
        """
        Callback function triggered when a cell is picked.

        :param actor: The picked actor.
        :type actor: pv.Actor
        """
        if not self.visualizer.plotter:
            logger.error("Plotter is not initialized.")
            self.update_status_display("Plotter is not initialized.")
            return
        logger.debug("on_pick called with actor: %s", actor)
        self.reset_actor_appearances()

        # Close any existing popup
        if self._current_popup is not None and self._current_popup.isVisible():
            self._current_popup.close()
            self._current_popup = None

        # If no actor was picked or no picked point, exit early
        if actor is None:
            logger.debug("No actor was picked")
            self.update_status_display("No object picked.")
            # self.reset_picking_state()
            return

        if (
            not hasattr(self.vtk_plotter, "picked_point")
            or self.vtk_plotter.picked_point is None
        ):
            logger.debug("No picked point available")
            self.update_status_display("No object picked.")
            # self.reset_picking_state()
            return

        picked_point = self.vtk_plotter.picked_point
        logger.debug("Picked point: %s", picked_point)

        # Log actor information
        if hasattr(actor, "GetProperty"):
            color = actor.GetProperty().GetColor()
            logger.debug("Actor color: %s", color)

        # Get the picked actor's name from the plotter's actors dictionary
        picked_actor_name = None
        logger.debug("Searching for actor in plotter.actors dictionary")
        for name, act in self.plotter.actors.items():
            logger.debug("Checking actor: %s", name)
            if act == actor:
                picked_actor_name = name
                logger.debug("Found actor name: %s", picked_actor_name)
                break

        # If we couldn't identify the actor by name, try to identify by type
        if picked_actor_name is None:
            logger.debug("Could not identify actor by name, trying by type")
            # Check if it's one of our MultiBlock actors
            if actor.GetMapper() and actor.GetMapper().GetInput():
                logger.debug("Actor has mapper and input")
                if isinstance(actor.GetMapper().GetInput(), pv.MultiBlock):
                    logger.debug("Actor input is a MultiBlock")
                    # Determine which type of MultiBlock it is based on color
                    color = actor.GetProperty().GetColor()
                    logger.debug("Actor color: %s", color)
                    if np.allclose(color, [0, 1, 0], atol=0.1):  # Green for classes
                        picked_actor_name = "classes"
                    elif np.allclose(color, [0, 0, 1], atol=0.1):  # Blue for methods
                        picked_actor_name = "methods"
                    elif np.allclose(color, [1, 0, 0], atol=0.1):  # Red for functions
                        picked_actor_name = "functions"
                    logger.debug(
                        "Identified MultiBlock actor as: %s", picked_actor_name
                    )

        # If we still don't have an actor name, we can't proceed
        if picked_actor_name is None:
            logger.debug("Could not identify picked object")
            self.update_status_display("Could not identify picked object.")
            # self.reset_picking_state()
            return

        # Find the closest mesh to the picked point
        closest_mesh_id = None
        min_distance = float("inf")

        # Only process meshes that match the picked actor type
        actor_type_map = {
            "classes": "class",
            "methods": "method",
            "functions": "function",
        }

        element_type = actor_type_map.get(picked_actor_name)
        logger.debug("Element type from actor: %s", element_type)

        if element_type:
            # Log the number of elements in actor_to_element
            logger.debug(
                "Number of elements in actor_to_element: %d",
                len(self.visualizer.actor_to_element),
            )

            # Filter elements by type
            matching_elements = 0
            for mesh_id, elem_data in self.visualizer.actor_to_element.items():
                if elem_data.get("type") == element_type:
                    matching_elements += 1
                    if elem_data.get("mesh"):
                        mesh = elem_data.get("mesh")
                        mesh_center = np.array(mesh.center)
                        distance = np.linalg.norm(mesh_center - np.array(picked_point))
                        logger.debug(
                            "Element %s of type %s at distance %f",
                            elem_data.get("name"),
                            element_type,
                            distance,
                        )
                        if distance < min_distance:
                            min_distance = distance
                            closest_mesh_id = mesh_id

            logger.debug(
                "Found %d elements of type %s", matching_elements, element_type
            )
            logger.debug(
                "Closest element: %s at distance %f",
                closest_mesh_id,
                min_distance if closest_mesh_id else float("inf"),
            )

        if closest_mesh_id:
            element = self.visualizer.actor_to_element[closest_mesh_id]
            mesh = element["mesh"]
            self.highlight_actor(mesh)
            element_type = element["type"]
            element_name = element["name"]
            docstring = element["docstring"]
            title = f"{element_type.capitalize()}: {element_name}"
            # self.update_status_display(f"Picked {element_type}: {element_name}")
            self._current_popup = DocstringPopup(
                title,
                format_docstring_to_markdown(docstring),
                self,
                on_close_callback=self.reset_picking_state,
            )
            self._current_popup.show()
        else:
            self.update_status_display(
                f"No {element_type} found near the picked point."
            )
            self.reset_picking_state()

    def reset_actor_appearances(self):
        """
        Reset all actors to their original appearance and clear picking highlights.
        This includes removing any bounding boxes or outlines that might be visible.
        """
        if not hasattr(self.visualizer, "actor_to_element"):
            logger.debug("No actor_to_element attribute found in visualizer")
            return

        # Log the current actors in the plotter
        logger.debug("Current actors in plotter: %s", list(self.plotter.actors.keys()))

        # Remove highlight actor
        if self._current_picked_actor:
            logger.debug("Removing highlight actor")
            self.plotter.remove_actor(self._current_picked_actor, reset_camera=False)
            self._current_picked_actor = None

        # Reset MultiBlock actor colors
        for actor_name in ["classes", "methods", "functions"]:
            actor = self.plotter.actors.get(actor_name)
            if actor:
                color = {
                    "classes": CLASS_COLOR,
                    "methods": METHOD_COLOR,
                    "functions": FUNCTION_COLOR,
                }[actor_name]
                logger.debug("Resetting color for %s actor to %s", actor_name, color)
                actor.prop.color = color
                actor.prop.show_edges = False
                actor.prop.line_width = 1

        # Remove bounds actors and hide bounds
        for actor_name in list(self.plotter.actors.keys()):
            if "bounds" in actor_name.lower() or "outline" in actor_name.lower():
                logger.debug("Removing bounds/outline actor: %s", actor_name)
                self.plotter.remove_actor(actor_name, reset_camera=False)

        logger.debug("Reset actor appearances and cleared picking highlights")

    def reset_picking_state(self):
        """
        Reset the picking state after interaction is complete.
        """
        remove_all_red_actors(self.plotter)
        self.reset_actor_appearances()
        self.class_selector.clearSelection()
        self.method_selector.clearSelection()
        self.function_selector.clearSelection()
        # self.plotter.picker = None

        self.update_status_display("Ready")
        self.plotter.render()

    def update_package_path(self) -> None:
        """
        Update the package path based on user input.
        """
        text: str = self.package_path_input.text()
        self.visualizer.status = "Loading Package..."
        self.visualizer.package_path = text

        repo_name = os.path.basename(text)
        self.visualizer.save_path = repo_name
        self.save_path_input.setText(repo_name)

        # Only update window title and status if the package path exists
        # The update_classes method (called when package_path changes) will have already
        # set the appropriate error status if the path doesn't exist
        if os.path.exists(text):
            num_classes = self.visualizer.num_classes
            num_functions = self.visualizer.num_functions
            num_methods = self.visualizer.num_methods
            num_faces = self.visualizer.num_faces

            self.setWindowTitle(
                f"Pkg: {text} | Classes: {num_classes} | Functions: {num_functions} | Methods: {num_methods} | Faces: {num_faces}"
            )
            self.visualizer.status = "Package loaded. Ready to visualize."
        # No need for an else clause as update_classes already sets the error status

    def update_save_path(self, text: str) -> None:
        """
        Update the save path based on user input.
        """
        self.visualizer.save_path = text

    def update_save_format(self, text: str) -> None:
        """
        Update the save format based on user selection.
        """
        self.visualizer.save_format = text

    def update_class_radius(self, value: int) -> None:
        """
        Update the class radius based on slider value.
        """
        min_bound, max_bound = self.visualizer.param.class_radius.bounds
        clamped_value = max(min_bound * 20.0, min(max_bound * 20.0, value))
        self.visualizer.class_radius = clamped_value / 20.0

    def update_selected_classes(self) -> None:
        """
        Update the selected classes based on user selection.
        """
        self.visualizer.selected_classes = [
            item.text() for item in self.class_selector.selectedItems()
        ]

    def update_selected_methods(self) -> None:
        """
        Update the selected methods based on user selection.
        """
        self.visualizer.selected_methods = [
            item.text() for item in self.method_selector.selectedItems()
        ]

    def update_selected_functions(self) -> None:
        """
        Update the selected functions based on user selection.
        """
        self.visualizer.selected_functions = [
            item.text() for item in self.function_selector.selectedItems()
        ]

    def show_method_docstring(self, item) -> None:
        """
        Show the docstring for the selected method in a popup and highlight the mesh.
        """
        # Close any existing popup
        if self._current_popup is not None and self._current_popup.isVisible():
            self._current_popup.close()
            self._current_popup = None

        method_name = item.text()
        # Method name is in format "ClassName.method_name"
        class_name, method_name_only = method_name.split(".")

        # Find the class element
        class_element = next(
            (
                e
                for e in self.visualizer.elements
                if e["type"] == "class" and e["name"] == class_name
            ),
            None,
        )

        if class_element and "methods" in class_element:
            # Find the method in the class's methods
            method = next(
                (m for m in class_element["methods"] if m["name"] == method_name_only),
                None,
            )

            if method:
                # Find the mesh for this method
                mesh_entry = next(
                    (
                        (mesh_id, elem)
                        for mesh_id, elem in self.visualizer.actor_to_element.items()
                        if elem["type"] == "method" and elem["name"] == method_name
                    ),
                    None,
                )

                if mesh_entry:
                    mesh_id, elem = mesh_entry
                    self.highlight_actor(elem["mesh"])

                self._current_popup = DocstringPopup(
                    f"Method: {method_name}",
                    format_docstring_to_markdown(method.get("docstring", "")),
                    self,
                    on_close_callback=self.reset_picking_state,
                )
                self._current_popup.show()

    def update_include_functions(self, state: int) -> None:
        """
        Update the include functions flag based on checkbox state and enable/disable the function selector.
        """
        self.visualizer.include_functions = state == Qt.Checked
        self.function_selector.setEnabled(self.visualizer.include_functions)

    def update_render_methods(self, state: int) -> None:
        """
        Update the render methods flag based on checkbox state and enable/disable the method selector.
        """
        self.visualizer.render_methods = state == Qt.Checked
        self.method_selector.setEnabled(self.visualizer.render_methods)

    def on_visualize_clicked(self) -> None:
        """
        Handle the visualize button click event.
        """
        logger.debug("Visualize button clicked")
        self.visualizer.visualize()

        # After visualization is complete, log the plotter actors for debugging
        logger.debug("Visualization complete, logging plotter actors")
        self.log_plotter_actors()

    def on_status_change(self, event: param.Event) -> None:
        """
        Handle status change events from the visualizer.
        """
        self.status_changed.emit(event.new)
        QApplication.processEvents()

    def update_status_display(self, status: str) -> None:
        """
        Update the status display label with the current status.
        """
        match status:
            case status if status.startswith("Visualization "):
                save_path: str = status.split("Visualization saved to ")[-1].strip()
                self.status_display.setText(
                    f"<span style='color:#006600; font-size:14px;'>{save_path}</span> "
                )

            case status if status.startswith("Error"):
                self.status_display.setText(
                    f"<span style='color:#cc0000; font-size:14px;'><b>💥 Error:</b> {status[6:]}</span>"
                )

            case status if "Analyzing" in status or "Creating" in status:
                self.status_display.setText(
                    f"<span style='color:#0066cc; font-size:14px;'><b>⏳ {status}</b></span>"
                )

            case status if "Ready" in status:
                self.status_display.setText(
                    f"<span style='color:#006600; font-size:14px;'><b>⚡ {status}</b></span>"
                )

            case status if "Scene generation" in status or "Spin complete" in status:
                self.status_display.setText(
                    f"<span style='color:#006600; font-size:14px;'><b>⏳ {status}</b></span>"
                )
                time.sleep(0.5)
                status = "Ready"
                self.status_display.setText(
                    f"<span style='color:#006600; font-size:14px;'><b>⚡ {status}</b></span>"
                )

            case status if "Found" in status:
                parts: List[str] = status.split("Found ")
                self.status_display.setText(
                    f"<span style='color:#008800; font-size:14px;'><b>✓</b> Found {parts[1]}</span>"
                )

            case status if "Rendering" in status:
                progress_parts = status.split("|")
                if len(progress_parts) >= 2:
                    task_name = progress_parts[0].strip()
                    progress_info = progress_parts[1].strip()
                    self.status_display.setText(
                        f"<span style='color:#0066cc; font-size:14px;'><b>{task_name}</b></span> | <span style='color:#008800; font-size:14px;'>{progress_info}</span>"
                    )
                else:
                    self.status_display.setText(status)
            case _:
                self.status_display.setText(status)
        self.status = status

    def update_class_selector(self, event: param.Event) -> None:
        """
        Update the class selector with new available classes.
        """
        self.class_selector.clear()
        for item in event.new:
            self.class_selector.addItem(item)

    def update_method_selector(self, event: param.Event) -> None:
        """
        Update the method selector with new available methods.
        """
        self.method_selector.clear()
        for item in event.new:
            self.method_selector.addItem(item)

    def update_function_selector(self, event: param.Event) -> None:
        """
        Update the function selector with new available functions.
        """
        self.function_selector.clear()
        for item in event.new:
            self.function_selector.addItem(item)

    def update_window_title(self, event: param.Event) -> None:
        """
        Update the window title with the new title.
        """
        self.setWindowTitle(event.new)

    def reset_camera(self) -> None:
        """
        Reset the camera to its default position and update the status.
        """
        plotter = self.visualizer.plotter

        if plotter:

            self.vtk_plotter.view_xy(render=False)
            plotter.camera.focal_point = [0, 0, 0]
            self.visualizer.status = "Ready: View reset to default."
        else:
            self.visualizer.status = "Error: Plotter is not initialized."

    def save_current_view(self) -> None:
        """
        Save the current visualization view to a file.
        """
        save_path: str = self.visualizer.save_path
        save_format: str = self.visualizer.save_format

        if not save_path.endswith(f".{save_format}"):
            save_path = f"{save_path}.{save_format}"

        self.visualizer.status = f"Saving visualization to {save_path}..."
        self.status_changed.emit(self.visualizer.status)
        QApplication.processEvents()

        plotter: pv.Plotter = self.visualizer.plotter
        save_path = Path(save_path).with_suffix(f".{save_format}")

        # Ensure the parent directory exists
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured parent directory exists: %s", save_path.parent)
        except PermissionError as e:
            error_msg = f"Permission denied when creating directory: {str(e)}"
            logger.error(error_msg)
            self.visualizer.status = f"Error saving: {error_msg}"
            self.status_changed.emit(self.visualizer.status)
            QApplication.processEvents()
            return
        except OSError as e:
            error_msg = f"Failed to create directory: {str(e)}"
            logger.error(error_msg)
            self.visualizer.status = f"Error saving: {error_msg}"
            self.status_changed.emit(self.visualizer.status)
            QApplication.processEvents()
            return

        # rprint("[bold green]Starting save operation...[/bold green]")
        logger.debug("Starting save operation to %s", save_path)

        try:
            if save_format == "html":
                plotter.export_html(save_path)
            elif save_format in ["png", "jpg"]:
                # rprint("[bold green]Taking screenshot of current view...[/bold green]")
                logger.debug("Taking screenshot of current view...")
                original_text_actors: List[Tuple[str, pv.Actor]] = []
                for actor_key in list(plotter.actors.keys()):
                    if "text" in actor_key.lower():
                        original_text_actors.append(
                            (actor_key, plotter.actors[actor_key])
                        )
                title_actor: pv.Actor = plotter.add_text(
                    self.visualizer.old_title,
                    position="upper_edge",
                    font_size=12,
                    color="black",
                )
                plotter.screenshot(save_path, window_size=[1200, 1200])
                plotter.remove_actor(title_actor)
                plotter.render()
            else:
                raise ValueError(f"Unsupported save format: {save_format}")

            self.visualizer.status = f"Visualization saved to: {save_path}"
            logger.debug("Visualization saved to %s", save_path)
            self.status_changed.emit(self.visualizer.status)
            QApplication.processEvents()
        except (ValueError, RuntimeError, FileNotFoundError) as e:
            logger.error("Failed to save: %s", e)
            self.visualizer.status = f"Error saving visualization: {str(e)}"
            self.status_changed.emit(self.visualizer.status)
            QApplication.processEvents()

    def spin_camera(self):
        """
        Spins the camera around the scene center.
        """
        spins = 2
        sps = 1  # spin/second
        fps = 30  # frames per second
        dpf = 2  # degrees per frame
        duration = spins / sps
        center = np.array(ORIGIN)
        n_points = int(360 / dpf)

        plotter = self.visualizer.plotter
        self.visualizer.status = "Spinning camera..."
        self.update_status_display("Spinning camera...")

        center_pos = center
        up = np.array((0, 1, 0))
        current_pos = np.array(plotter.camera_position[0])
        radius = np.linalg.norm(current_pos - center_pos)
        theta = np.linspace(0, 2 * np.pi, n_points)
        path = np.c_[
            radius * np.cos(theta), np.zeros_like(theta), radius * np.sin(theta)
        ]

        self.current_frame = 0
        self.spin_count = 0

        def update_camera():
            if self.current_frame < len(path):
                pos = path[self.current_frame]
                plotter.camera_position = [pos, center, up]
                if self.current_frame < len(path) - 1:
                    self.current_frame += 1
                plotter.render()
            else:
                self.current_frame = 0
                self.spin_count += 1
                if self.spin_count >= spins:
                    stop_timer()
                    return

        def stop_timer():
            self.timer.stop()
            self.update_status_display("Spin complete.")

        self.timer = QTimer(self)
        self.timer.timeout.connect(update_camera)
        interval = int(1000 / fps)
        self.timer.start(int(interval / 2))
        QTimer.singleShot(int(duration * 1000 * 2), stop_timer)

    def reset_settings(self) -> None:
        """
        Reset all settings to their default values, reset the view, and set status to 'Ready'.
        """
        logger.debug("Resetting settings")
        default_class_radius = 5.5
        self.class_radius_slider.setValue(int(default_class_radius * 20))
        self.visualizer.class_radius = default_class_radius
        self.class_radius_slider.update()
        QApplication.processEvents()
        self.reset_camera()
        self.visualizer.status = "Ready"
        # self.update_status_display("Ready")
        # self.visualizer.visualize()

        # After visualization is complete, log the plotter actors for debugging
        # logger.debug("Reset complete, logging plotter actors")
        # self.log_plotter_actors()

        return

    def cleanup_pyvista_objects(self):
        """
        Perform thorough cleanup of PyVista objects to prevent errors during exit.
        This breaks potential circular references and ensures proper cleanup.
        """
        logger.debug("Performing thorough PyVista cleanup")

        # Close any open popups
        if hasattr(self, "_current_popup") and self._current_popup is not None:
            if (
                hasattr(self._current_popup, "isVisible")
                and self._current_popup.isVisible()
            ):
                try:
                    self._current_popup.close()
                except Exception as e:
                    logger.debug("Error closing popup: %s", e)
            self._current_popup = None

        # Clear the actor_to_element dictionary and break references
        if hasattr(self, "visualizer") and hasattr(self.visualizer, "actor_to_element"):
            try:
                # Break mesh references in actor_to_element
                for mesh_id in list(self.visualizer.actor_to_element.keys()):
                    elem_data = self.visualizer.actor_to_element[mesh_id]
                    if "mesh" in elem_data:
                        elem_data["mesh"] = None
                self.visualizer.actor_to_element.clear()
            except Exception as e:
                logger.debug("Error clearing actor_to_element: %s", e)

        # Clear and close the plotter
        if hasattr(self, "plotter") and self.plotter is not None:
            try:
                # Remove all actors first
                if hasattr(self.plotter, "actors"):
                    actor_keys = list(self.plotter.actors.keys())
                    for actor_key in actor_keys:
                        try:
                            self.plotter.remove_actor(actor_key, render=False)
                        except Exception as e:
                            logger.debug("Error removing actor %s: %s", actor_key, e)

                # Clear all MultiBlock objects
                if hasattr(self.visualizer, "plotter"):
                    self.visualizer.plotter = None

                # Clear the plotter
                if hasattr(self.plotter, "clear_actors"):
                    self.plotter.clear_actors()
                if hasattr(self.plotter, "clear"):
                    self.plotter.clear()
                if hasattr(self.plotter, "close"):
                    self.plotter.close()

                # Set plotter to None to break references
                self.plotter = None
                self.vtk_plotter = None
            except Exception as e:
                logger.debug("Error clearing plotter: %s", e)

        # Force garbage collection
        gc.collect()

    def closeEvent(self, event):
        """
        Handle the window close event with thorough cleanup to prevent PyVista errors.
        """
        logger.debug("Window close event received")

        # Perform thorough cleanup
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.cleanup_pyvista_objects()
            except Exception as e:
                logger.debug("Error during cleanup: %s", e)

        event.accept()

    def run(self) -> None:
        """Start the Qt application loop and display the plot."""
        try:
            # Get or create QApplication instance
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            self.show()
            app.exec()
        except Exception as e:
            logger.error("Error running the application: %s", e)
            self.visualizer.status = f"Error running the application: {str(e)}"
            self.status_changed.emit(self.visualizer.status)
            QApplication.processEvents()
        finally:
            sys.exit()


# class MainWindow(QMainWindow) ends here


# More aggressive global cleanup function for atexit
def global_cleanup():
    """
    Global cleanup function registered with atexit to ensure proper cleanup
    of PyVista objects when the program exits.
    """
    logger.debug("Running global cleanup on exit")

    # Suppress warnings during cleanup
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            # Clear any remaining references to PyVista objects
            for obj in gc.get_objects():
                if isinstance(obj, pv.PolyData) or isinstance(obj, pv.MultiBlock):
                    try:
                        # Monkey patch the object's __del__ method to prevent errors
                        if hasattr(obj, "__del__"):
                            obj.__del__ = types.MethodType(lambda self: None, obj)

                        # Set object attributes to None to break circular references
                        for attr_name in dir(obj):
                            if not attr_name.startswith("__"):
                                try:
                                    setattr(obj, attr_name, None)
                                    logger.info(
                                        "Setting %s attribute %s to None for cleanup",
                                        obj,
                                        attr_name,
                                    )
                                except (AttributeError, TypeError):
                                    pass
                    except Exception as e:
                        logger.debug("Error cleaning up PyVista object: %s", e)

            # Force garbage collection
            gc.collect()

        except Exception as e:
            logger.debug("Error during global cleanup: %s", e)


# Register the cleanup function with atexit
atexit.register(global_cleanup)

# Monkey patch the sys.excepthook to catch and ignore specific PyVista errors
original_excepthook = sys.excepthook


def custom_excepthook(exc_type, exc_value, exc_traceback):
    # Ignore specific PyVista/VTK errors during exit
    if exc_type is TypeError and (
        "'NoneType' and 'tuple'" in str(exc_value)
        or "has no attribute" in str(exc_value)
    ):
        return
    # Pass other exceptions to the original handler
    original_excepthook(exc_type, exc_value, exc_traceback)


sys.excepthook = custom_excepthook

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D Visualization of Python Package Structure"
    )
    parser.add_argument(
        "--package_path",
        "-k",
        type=str,
        help="Path to the input package",
        required=False,
        default=DEFAULT_REP,
    )
    parser.add_argument(
        "--save_path",
        "-s",
        type=str,
        help="Path to save the output visualization. Defaults to the input package name if not provided.",
        required=False,
    )
    parser.add_argument(
        "--width",
        "-w",
        type=int,
        help="Width of the window",
        default=1200,
    )
    parser.add_argument(
        "--height",
        "-t",
        type=int,
        help="Height of the window",
        default=800,
    )
    args = parser.parse_args()

    package_path = args.package_path if args.package_path else DEFAULT_REP
    save_path = (
        args.save_path
        if args.save_path
        else os.path.basename(os.path.normpath(package_path))
    )

    app: QApplication = QApplication([])
    window: MainWindow = MainWindow(
        package_path=package_path,
        save_path=save_path,
        width=args.width,
        height=args.height,
    )

    window.run()

    # end of file
