# pylint: disable=C0301,C0116,C0115,W0613,E0611,C0413

"""
Module: visualize_repository_qt

A PyQt5 application for 3D visualization of a Python repository's structure.
Parses classes, methods, and functions, rendering them as 3D objects using PyVista.

Key Features:
- Extracts repository structure using AST.
- Visualizes classes (red dodecahedrons, scaled by method count), methods (blue spheres), and functions (green cylinders around package center).
- Interactive UI for customizing and saving visualizations (HTML, PNG, JPEG).
- Supports picking of classes, methods, and functions to display their docstrings via cell picking.

Usage:
Run: python visualize_repository_qt.py

Author: Eric G. Suchanek, PhD
Last modified: 2025-05-05 23:00:00
"""

import ast
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import param
import pyvista as pv
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
from rich import print as rprint
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)


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


# Constants
ORIGIN: Tuple[float, float, float] = (0, 0, 0)
DEFAULT_REP: str = "/Users/egs/repos/proteusPy"
DEFAULT_PACKAGE_NAME: str = os.path.basename(DEFAULT_REP)
DEFAULT_SAVE_PATH: str = os.path.join(os.path.expanduser("~"), "Desktop")
DEFAULT_SAVE_NAME: str = f"{DEFAULT_PACKAGE_NAME}_3d_visualization"
FONTSIZE: int = 12

# Configure logging
logging.basicConfig(level=logging.WARNING, handlers=[RichHandler()])
logger: logging.Logger = logging.getLogger()


# Check PyQt5
def can_import(module_name: str) -> Optional[object]:
    """
    Check if a module can be imported.

    :param module_name: Name of the module to check.
    :type module_name: str
    :return: The imported module if successful, None otherwise.
    :rtype: Optional[object]
    """
    try:
        return __import__(module_name)
    except ImportError:
        return None


if can_import("PyQt5") is None:
    sys.exit("This program requires PyQt5. Install: pip install proteusPy[pyqt5]")


def parse_file(file_path: str) -> List[Dict[str, Union[str, int, List[str]]]]:
    """
    Parse a Python file and extract class and function definitions using AST.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            tree: ast.AST = ast.parse(file.read(), filename=file_path)
    except (SyntaxError, UnicodeDecodeError):
        return []

    elements: List[Dict[str, Union[str, int, List[str]]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            elements.append(
                {
                    "type": "class",
                    "name": node.name,
                    "methods": [
                        {
                            "name": n.name,
                            "docstring": ast.get_docstring(n),
                            "lineno": n.lineno,
                        }
                        for n in node.body
                        if isinstance(n, ast.FunctionDef)
                    ],
                    "lineno": node.lineno,
                    "docstring": ast.get_docstring(node),
                }
            )
        elif isinstance(node, ast.FunctionDef) and not any(
            isinstance(parent, ast.ClassDef) for parent in ast.iter_child_nodes(tree)
        ):
            elements.append(
                {
                    "type": "function",
                    "name": node.name,
                    "lineno": node.lineno,
                    "docstring": ast.get_docstring(node),
                }
            )
    return elements


def collect_elements(repo_path: str) -> List[Dict[str, Union[str, int, List[str]]]]:
    """
    Collect class and function elements from all Python files in a repository.
    """
    elements: List[Dict[str, Union[str, int, List[str]]]] = []
    seen_classes: set = set()
    seen_functions: set = set()
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                file_elements: List[Dict[str, Union[str, int, List[str]]]] = parse_file(
                    os.path.join(root, file)
                )
                for elem in file_elements:
                    if elem["type"] == "class" and elem["name"] not in seen_classes:
                        seen_classes.add(elem["name"])
                        elements.append(elem)
                    elif (
                        elem["type"] == "function"
                        and elem["name"] not in seen_functions
                    ):
                        seen_functions.add(elem["name"])
                        elements.append(elem)
    return elements


def fibonacci_sphere(
    samples: int, radius: float = 1.0, center: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """
    Generate points on a sphere using the Fibonacci spiral algorithm.
    """
    if center is None:
        center = np.array([0, 0, 0])
    if samples <= 0:
        return []
    if samples == 1:
        return [center + radius * np.array([0, 0, 1])]

    points: List[np.ndarray] = []
    phi: float = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(samples):
        y: float = 1 - (i / float(samples - 1)) * 2
        radius_at_y: float = np.sqrt(1 - y * y)
        theta: float = phi * i
        x: float = np.cos(theta) * radius_at_y
        z: float = np.sin(theta) * radius_at_y
        points.append(center + radius * np.array([x, y, z]))
    return points


def create_3d_visualization(
    viz_instance: "RepositoryVisualizer",
    elements: List[Dict[str, Union[str, int, List[str]]]],
    save_path: str,
    save_format: str = "html",
    class_radius: float = 4.0,
    member_radius_scale: float = 1.0,
    old_title: str = "",
    plotter: Optional[pv.Plotter] = None,
) -> Tuple[pv.Plotter, str, Dict[pv.Actor, Dict[str, str]]]:
    """
    Create a 3D visualization of the repository structure using PyVista.

    :param viz_instance: Instance of RepositoryVisualizer to update status.
    :type viz_instance: RepositoryVisualizer
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
    :return: Tuple containing the plotter, title text, and actor-to-element mapping.
    :rtype: Tuple[pv.Plotter, str, Dict[pv.Actor, Dict[str, str]]]
    """
    viz_instance.status = "Setting up visualization..."
    QApplication.processEvents()

    # Initialize actor-to-element mapping
    actor_to_element: Dict[pv.Actor, Dict[str, str]] = {}

    # Reset plotter
    plotter.clear_actors()
    plotter.remove_all_lights()
    plotter.renderer.ResetCamera()
    plotter.renderer_layer = 0

    plotter.disable_parallel_projection()
    plotter.enable_anti_aliasing("msaa")
    plotter.add_axes()
    plotter.add_light(pv.Light(position=(50, 100, 100), color="white", intensity=1.0))
    plotter.add_light(
        pv.Light(position=(-50, -100, -100), color="white", intensity=0.1)
    )
    plotter.add_light(pv.Light(position=(0, 0, 100), color="white", intensity=0.5))

    package_center: np.ndarray = np.array([0, 0, 0])
    package_name: str = Path(save_path).stem
    package_radius: float = 1.0
    package_mesh: pv.PolyData = pv.Icosahedron(
        center=package_center, radius=package_radius
    )
    plotter.add_mesh(
        package_mesh, color="purple", show_edges=False, smooth_shading=False
    )

    num_classes: int = len([e for e in elements if e["type"] == "class"])
    viz_instance.status = f"Rendering {num_classes} classes..."
    QApplication.processEvents()
    class_positions: List[np.ndarray] = fibonacci_sphere(
        num_classes, radius=class_radius, center=package_center
    )
    class_index: int = 0

    rprint(f"[bold green]Starting to render {num_classes} classes...[/bold green]")
    logger.info("Starting to render %s classes...", num_classes)
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, complete_style="green", finished_style="bold green"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        expand=True,
    ) as progress:
        task = progress.add_task("[bold red]Rendering classes...", total=num_classes)

        for element in elements:
            if element["type"] != "class":
                continue
            pos: np.ndarray = class_positions[class_index]
            class_index += 1
            method_count: int = len(element.get("methods", []))
            scaled_radius: float = (0.75 / 2) * (1 + 0.1 * np.log(method_count + 1))
            mesh: pv.PolyData = pv.Dodecahedron(radius=scaled_radius, center=pos)
            class_actor = plotter.add_mesh(
                mesh, color="red", show_edges=False, smooth_shading=False
            )
            actor_to_element[class_actor] = {
                "type": "class",
                "name": element["name"],
                "docstring": element.get("docstring", ""),
            }
            line: pv.PolyData = pv.Cylinder(
                radius=0.025,
                height=np.linalg.norm(pos - package_center),
                center=(pos + package_center) / 2,
                direction=pos - package_center,
            )
            plotter.add_mesh(line, color="red", show_edges=False, smooth_shading=True)
            rprint(f"[bold red]Actor: {class_actor}...[/bold red]")
            update_interval: int = max(1, int(num_classes * 0.20))
            if class_index % update_interval == 0 or class_index == num_classes:
                progress.update(task, completed=class_index)
                QApplication.processEvents()

    rprint("[bold green]Finished rendering classes![/bold green]")
    logger.info("Finished rendering classes!")

    num_functions: int = len([e for e in elements if e["type"] == "function"])
    if num_functions > 0:
        viz_instance.status = f"Rendering {num_functions} functions..."
        QApplication.processEvents()
        function_positions: List[np.ndarray] = fibonacci_sphere(
            num_functions, radius=package_radius * 1.25, center=package_center
        )

        rprint(
            f"[bold green]Starting to render {num_functions} functions...[/bold green]"
        )
        logger.info("Starting to render %s functions...", num_functions)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(
                bar_width=40, complete_style="green", finished_style="bold green"
            ),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            expand=True,
        ) as progress:
            func_task = progress.add_task(
                "[bold green]Rendering functions...", total=num_functions
            )

            for i, element in enumerate(
                [e for e in elements if e["type"] == "function"]
            ):
                pos: np.ndarray = function_positions[i]
                mesh: pv.PolyData = pv.Cylinder(
                    radius=0.15 / 2, height=0.15 / 2, center=pos, direction=(0, 0, 1)
                )
                function_actor = plotter.add_mesh(
                    mesh, color="green", show_edges=False, smooth_shading=True
                )
                actor_to_element[function_actor] = {
                    "type": "function",
                    "name": element["name"],
                    "docstring": element.get("docstring", ""),
                }
                line: pv.PolyData = pv.Line(package_center, pos)
                plotter.add_mesh(line, color="green", line_width=2)
                plotter.reset_camera()

                update_interval: int = max(1, int(num_functions * 0.20))
                if (i + 1) % update_interval == 0 or (i + 1) == num_functions:
                    progress.update(func_task, completed=i + 1)
                    QApplication.processEvents()

        rprint("[bold green]Finished rendering functions![/bold green]")
        logger.info("Finished rendering functions!")

    viz_instance.status = "Rendering methods..."
    QApplication.processEvents()

    total_methods: int = sum(
        len(class_elem.get("methods", []))
        for class_elem in [e for e in elements if e["type"] == "class"]
    )

    if total_methods > 0:
        rprint(
            f"[bold green]Starting to render {total_methods} methods...[/bold green]"
        )
        logger.info("Starting to render %s methods...", total_methods)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(
                bar_width=40, complete_style="green", finished_style="bold green"
            ),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            expand=True,
        ) as progress:
            method_task = progress.add_task(
                "[bold blue]Rendering methods...", total=total_methods
            )

            method_count: int = 0
            for class_pos, class_elem in zip(
                class_positions, [e for e in elements if e["type"] == "class"]
            ):
                members: List[Dict[str, Union[str, int]]] = class_elem.get(
                    "methods", []
                )
                if members:
                    method_positions: List[np.ndarray] = fibonacci_sphere(
                        len(members),
                        radius=member_radius_scale * 0.75,
                        center=class_pos,
                    )
                    for j, member in enumerate(members):
                        sphere: pv.PolyData = pv.Sphere(
                            radius=0.225 / 2, center=method_positions[j]
                        )
                        method_actor = plotter.add_mesh(
                            sphere, color="blue", show_edges=False, smooth_shading=True
                        )
                        actor_to_element[method_actor] = {
                            "type": "method",
                            "name": f"{class_elem['name']}.{member['name']}",
                            "docstring": member.get("docstring", ""),
                        }
                        line: pv.PolyData = pv.Line(class_pos, method_positions[j])
                        plotter.add_mesh(line, color="blue", line_width=1)
                        method_count += 1

                        update_interval: int = max(1, int(total_methods * 0.20))
                        if (
                            method_count % update_interval == 0
                            or method_count == total_methods
                        ):
                            progress.update(method_task, completed=method_count)
                            QApplication.processEvents()

        rprint("[bold green]Finished rendering methods![/bold green]")
        logger.info("Finished rendering methods!")

    plotter.reset_camera()
    QApplication.processEvents()

    num_methods: int = sum(
        len(e.get("methods", [])) for e in elements if e["type"] == "class"
    )
    num_functions: int = len([e for e in elements if e["type"] == "function"])

    title_text: str = (
        f"3D Visualization: {package_name} | Classes: {num_classes} | Methods: {num_methods} | Functions: {num_functions}"
    )

    plotter.render()
    viz_instance.status = "Scene generation complete."

    return plotter, title_text, actor_to_element


class RepositoryVisualizer(param.Parameterized):
    repo_path: str = param.String(default=DEFAULT_REP, doc="Repository path")
    save_path: str = param.String(default=DEFAULT_PACKAGE_NAME, doc="Save path")
    old_title: str = param.String(
        default=f"{DEFAULT_PACKAGE_NAME} 3d visualization", doc="Visualization title"
    )
    window_title: str = param.String(
        default=f"{DEFAULT_PACKAGE_NAME} 3d visualization", doc="Window title"
    )
    save_format: str = param.Selector(
        objects=["html", "png", "jpg"], default="html", doc="Output format"
    )
    class_radius: float = param.Number(
        default=4.0, bounds=(1.0, 10.0), step=0.5, doc="Class radius"
    )
    member_radius_scale: float = param.Number(
        default=1.25, bounds=(0.5, 3.0), step=0.25, doc="Member radius scale"
    )
    available_classes: List[str] = param.List(default=[], doc="Available classes")
    selected_classes: List[str] = param.ListSelector(
        default=[], objects=[], doc="Selected classes"
    )
    available_functions: List[str] = param.List(default=[], doc="Available functions")
    selected_functions: List[str] = param.ListSelector(
        default=[], objects=[], doc="Selected functions"
    )
    include_functions: bool = param.Boolean(default=True, doc="Include functions")
    status: str = param.String(default="Ready", doc="Status")

    def __init__(self, plotter: Optional[pv.Plotter], **params: dict) -> None:
        """
        Initialize the RepositoryVisualizer.
        """
        super().__init__(**params)
        self.plotter: Optional[pv.Plotter] = plotter
        self.elements: List[Dict[str, Union[str, int, List[str]]]] = []
        self.actor_to_element: Dict[pv.Actor, Dict[str, str]] = {}
        self.update_classes()

    @param.depends("repo_path", watch=True)
    def update_classes(self) -> None:
        """
        Update the list of available classes and functions based on the repository path.
        """
        if os.path.exists(self.repo_path):
            self.status = "Analyzing repository..."
            self.elements = collect_elements(self.repo_path)
            class_names: List[str] = sorted(
                [e["name"] for e in self.elements if e["type"] == "class"]
            )
            self.available_classes = class_names
            self.param.selected_classes.objects = class_names
            function_names: List[str] = sorted(
                [e["name"] for e in self.elements if e["type"] == "function"]
            )
            self.available_functions = function_names
            self.param.selected_functions.objects = function_names
            self.save_path = os.path.basename(self.repo_path)
            self.status = (
                f"Found {len(class_names)} classes and {len(function_names)} functions."
            )
        else:
            self.status = "Repository path does not exist."
            self.available_classes = []
            self.selected_classes = []
            self.param.selected_classes.objects = []
            self.available_functions = []
            self.selected_functions = []
            self.param.selected_functions.objects = []

    def visualize(self) -> None:
        """
        Create and display the 3D visualization based on selected parameters.
        """
        if not os.path.exists(self.repo_path):
            self.status = "Repository path does not exist."
            return
        if not self.elements:
            self.status = "No elements found."
            return

        filtered_elements: List[Dict[str, Union[str, int, List[str]]]] = []
        if self.selected_classes:
            for e in self.elements:
                if e["type"] == "class" and e["name"] in self.selected_classes:
                    filtered_elements.append(e)
        else:
            filtered_elements.extend([e for e in self.elements if e["type"] == "class"])

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
            plotter, title_text, actor_to_element = create_3d_visualization(
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
        except (ValueError, RuntimeError) as e:
            self.status = f"Error creating visualization: {str(e)}"


class MainWindow(QMainWindow):
    status_changed: pyqtSignal = pyqtSignal(str)

    def show_class_docstring(self, item) -> None:
        """
        Show the docstring for the selected class in a popup.
        """
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
            popup = DocstringPopup(
                f"Class: {class_name}",
                class_element.get("docstring", ""),
                self,
                on_close_callback=lambda: self.class_selector.clearSelection(),
            )
            popup.exec_()

    def show_function_docstring(self, item) -> None:
        """
        Show the docstring for the selected function in a popup.
        """
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
            popup = DocstringPopup(
                f"Function: {function_name}",
                function_element.get("docstring", ""),
                self,
                on_close_callback=lambda: self.function_selector.clearSelection(),
            )
            popup.exec_()

    def __init__(self, visualizer: RepositoryVisualizer) -> None:
        """
        Initialize the main window for the visualization application.
        """
        super().__init__()
        self.timer = None
        self.current_frame = 0
        self.visualizer: RepositoryVisualizer = visualizer
        self.setWindowTitle(self.visualizer.window_title)

        central_widget: QWidget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout: QHBoxLayout = QHBoxLayout(central_widget)

        self.setStyleSheet(
            """
            QWidget { font-family: Arial, sans-serif; font-size: 12px; }
            QLineEdit, QComboBox, QListWidget, QCheckBox, QPushButton, QLabel { margin: 2px; padding: 3px; }
            QLineEdit, QComboBox, QListWidget { border: 1px solid #ddd; border-radius: 3px; }
            QPushButton { background-color: '#4CAF50'; color: white; border: none; border-radius: 3px; padding: 6px; }
            QPushButton#reset { background-color: '#FFEB3B'; color: black; }
            QPushButton#reset-view { background-color: '#FFEB3B'; color: black; }
            QLabel { border: 1px solid #ddd; background-color: '#f5f5f5'; padding: 4px; }
            """
        )

        control_panel: QVBoxLayout = QVBoxLayout()
        control_panel.setSpacing(3)
        control_panel.setContentsMargins(8, 8, 8, 8)

        control_panel.addWidget(
            QLabel(
                "<h2>Input Parameters</h2>",
                font=QFont("Arial", 14, QFont.Bold),
                styleSheet="background: transparent; border: none;",
            )
        )

        repo_path_label: QLabel = QLabel(
            "<b style='font-size:13px;'>Repository Path</b>"
        )
        control_panel.addWidget(repo_path_label)
        self.repo_path_input: QLineEdit = QLineEdit(self.visualizer.repo_path)
        self.repo_path_input.setPlaceholderText("Enter repository path")
        control_panel.addWidget(self.repo_path_input)

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
        self.class_radius_slider.setMinimum(15)
        self.class_radius_slider.setMaximum(100)
        self.class_radius_slider.setValue(int(self.visualizer.class_radius * 10))
        self.class_radius_slider.setTickInterval(5)
        self.class_radius_slider.setTickPosition(QSlider.TicksBelow)
        control_panel.addWidget(self.class_radius_slider)

        self.member_radius_scale_slider: QSlider = QSlider(Qt.Horizontal)
        self.member_radius_scale_slider.setMinimum(5)
        self.member_radius_scale_slider.setMaximum(30)
        self.member_radius_scale_slider.setValue(
            int(self.visualizer.member_radius_scale * 10)
        )
        self.member_radius_scale_slider.setTickInterval(1)
        self.member_radius_scale_slider.setTickPosition(QSlider.TicksBelow)
        control_panel.addWidget(self.member_radius_scale_slider)
        control_panel.addWidget(QLabel("Member Radius Scale"))

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
                "<h2>Function Selection</h2>",
                font=QFont("Arial", 14, QFont.Bold),
                styleSheet="background: transparent; border: none;",
            )
        )
        for widget in control_panel.findChildren(QLabel):
            widget.setStyleSheet("background: transparent; border: none;")

        self.include_functions_checkbox: QCheckBox = QCheckBox("Include Functions")
        self.include_functions_checkbox.setChecked(self.visualizer.include_functions)
        control_panel.addWidget(self.include_functions_checkbox)
        control_panel.addWidget(QLabel("Select functions (empty for all):"))

        self.function_selector: QListWidget = QListWidget()
        self.function_selector.setSelectionMode(QListWidget.MultiSelection)
        self.function_selector.setMaximumHeight(80)
        for item in self.visualizer.available_functions:
            self.function_selector.addItem(item)
        control_panel.addWidget(self.function_selector)

        self.button_spin = QPushButton("Spin Repository")
        self.button_spin.clicked.connect(self.spin_camera)
        control_panel.addWidget(self.button_spin)

        control_panel.addStretch()

        vis_panel: QVBoxLayout = QVBoxLayout()
        vis_panel.setSpacing(10)
        vis_panel.setContentsMargins(10, 10, 10, 10)

        button_row: QHBoxLayout = QHBoxLayout()

        self.visualize_button: QPushButton = QPushButton("Visualize Repository")
        self.visualize_button.setFixedWidth(150)
        button_row.addWidget(self.visualize_button)

        self.save_button: QPushButton = QPushButton("Save View")
        self.save_button.setFixedWidth(150)
        button_row.addWidget(self.save_button)

        self.reset_camera_button: QPushButton = QPushButton("Reset Zoom")
        self.reset_camera_button.setFixedWidth(100)
        self.reset_camera_button.setObjectName("reset")
        button_row.addWidget(self.reset_camera_button)

        self.reset_view_button: QPushButton = QPushButton("Reset Orientation")
        self.reset_view_button.setFixedWidth(110)
        self.reset_view_button.setObjectName("reset-view")
        button_row.addWidget(self.reset_view_button)

        self.reset_view_button.clicked.connect(self.reset_view)
        self.save_button.clicked.connect(self.save_current_view)

        self.status_display: QLabel = QLabel("Ready")
        self.status_display.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.status_display.setStyleSheet("font-weight: bold; font-size: 14px;")
        button_row.addWidget(self.status_display, stretch=1)

        set_pyvista_theme("auto", verbose=True)
        self.vtk_widget: QtInteractor = QtInteractor(self, theme=pv._GlobalTheme())

        vis_panel.addWidget(self.vtk_widget, stretch=1)
        vis_panel.addLayout(button_row)

        control_widget: QWidget = QWidget()
        control_widget.setLayout(control_panel)
        control_widget.setFixedWidth(375)
        main_layout.addWidget(control_widget)

        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        vis_widget: QWidget = QWidget()
        vis_widget.setLayout(vis_panel)
        main_layout.addWidget(vis_widget, stretch=1)

        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 3)

        central_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        control_widget.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.MinimumExpanding
        )
        control_panel.setSizeConstraint(QVBoxLayout.SetMinimumSize)

        self.vtk_widget.enable_mesh_picking(
            callback=self.on_pick,  # Callback function for picking
            show=True,  # Show the picked cell
            show_message=True,  # Display a message when picking
            style="wireframe",  # Highlight picked cell in wireframe
            line_width=5,  # Width of the wireframe
            color="pink",  # Highlight color
            font_size=14,  # Font size for messages
            left_clicking=True,  # Enable left-click picking
            use_actor=True,  # Return the picked actor
        )

        self.repo_path_input.editingFinished.connect(self.update_repo_path)
        self.save_path_input.textChanged.connect(self.update_save_path)
        self.save_format_select.currentTextChanged.connect(self.update_save_format)
        self.class_radius_slider.sliderReleased.connect(
            lambda: self.update_class_radius(self.class_radius_slider.value())
        )
        self.member_radius_scale_slider.sliderReleased.connect(
            lambda: self.update_member_radius_scale(
                self.member_radius_scale_slider.value()
            )
        )
        self.class_selector.itemSelectionChanged.connect(self.update_selected_classes)
        self.function_selector.itemSelectionChanged.connect(
            self.update_selected_functions
        )
        self.include_functions_checkbox.stateChanged.connect(
            self.update_include_functions
        )
        self.visualize_button.clicked.connect(self.visualizer.visualize)
        self.reset_camera_button.clicked.connect(self.reset_camera)
        self.status_changed.connect(self.update_status_display)
        self.visualizer.param.watch(self.on_status_change, "status")
        self.visualizer.param.watch(self.update_class_selector, "available_classes")
        self.visualizer.param.watch(
            self.update_function_selector, "available_functions"
        )
        self.visualizer.param.watch(self.update_window_title, "window_title")

        for widget in self.findChildren(QLabel):
            widget.setStyleSheet("background: transparent; border: none;")

        self.class_selector.itemClicked.connect(self.show_class_docstring)
        self.function_selector.itemClicked.connect(self.show_function_docstring)

    def on_pick(self, mesh):
        """
        Callback function triggered when a cell is picked.

        :param mesh: The picked mesh (actor).
        :type mesh: pv.PolyData
        """
        # The mesh parameter is the actor in enable_cell_picking
        actor = mesh
        # rprint(f"Picked actor: {actor}")
        logger.debug("Picked actor: %s", actor)
        if actor in self.visualizer.actor_to_element:
            element = self.visualizer.actor_to_element[actor]
            element_type = element["type"]
            element_name = element["name"]
            docstring = element["docstring"]
            title = f"{element_type.capitalize()}: {element_name}"
            self.update_status_display(f"Picked {element_type}: {element_name}")
            popup = DocstringPopup(
                title,
                docstring,
                self,
                on_close_callback=lambda: self.update_status_display("Ready"),
            )
            popup.exec_()
        else:
            self.update_status_display("No object picked.")

    def update_repo_path(self) -> None:
        """
        Update the repository path based on user input.
        """
        text: str = self.repo_path_input.text()
        self.visualizer.status = "Loading Repository..."
        self.visualizer.repo_path = text

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
        self.visualizer.class_radius = value / 10.0
        self.visualizer.visualize()

    def update_member_radius_scale(self, value: int) -> None:
        """
        Update the member radius scale based on slider value.
        """
        self.visualizer.member_radius_scale = value / 10.0
        self.visualizer.visualize()

    def update_selected_classes(self) -> None:
        """
        Update the selected classes based on user selection.
        """
        self.visualizer.selected_classes = [
            item.text() for item in self.class_selector.selectedItems()
        ]

    def update_selected_functions(self) -> None:
        """
        Update the selected functions based on user selection.
        """
        self.visualizer.selected_functions = [
            item.text() for item in self.function_selector.selectedItems()
        ]

    def update_include_functions(self, state: int) -> None:
        """
        Update the include functions flag based on checkbox state.
        """
        self.visualizer.include_functions = state == Qt.Checked

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
        if status.startswith("Visualization saved to"):
            save_path: str = status.split("Visualization saved to ")[-1].strip()
            self.status_display.setText(
                f"<span style='font-size:12px'>{save_path}</span> "
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
            parts: List[str] = status.split("Found ")
            self.status_display.setText(
                f"<span style='color:#008800'><b>✓</b> Found {parts[1]}</span>"
            )
        else:
            self.status_display.setText(status)

    def update_class_selector(self, event: param.Event) -> None:
        """
        Update the class selector with new available classes.
        """
        self.class_selector.clear()
        for item in event.new:
            self.class_selector.addItem(item)

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
        Reset the camera to its default position.
        """
        self.vtk_widget.reset_camera()
        self.vtk_widget.render()

    def save_current_view(self) -> None:
        """
        Save the current visualization view to a file.
        """
        save_path: str = self.visualizer.save_path
        save_format: str = self.visualizer.save_format

        if not save_path.endswith(f".{save_format}"):
            save_path = f"{save_path}.{save_format}"

        self.visualizer.status = f"Saving visualization to {save_path}..."
        QApplication.processEvents()

        self.status_changed.emit("Saving visualization...")
        QApplication.processEvents()

        plotter: pv.Plotter = self.visualizer.plotter
        save_path = Path(save_path).with_suffix(f".{save_format}")

        rprint("[bold green]Starting save operation...[/bold green]")
        logger.info("Starting save operation to %s", save_path)

        try:
            if save_format == "html":
                plotter.export_html(save_path)
            elif save_format in ["png", "jpg"]:
                rprint("[bold green]Taking screenshot of current view...[/bold green]")
                logger.info("Taking screenshot of current view...")
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
            rprint(f"[bold green]{self.visualizer.status}[/bold green]")
            logger.info("Visualization saved to %s", save_path)
            self.status_changed.emit(self.visualizer.status)
            QApplication.processEvents()
        except (ValueError, RuntimeError) as e:
            logger.error("Failed to save: %s", e)
            self.visualizer.status = f"Error saving visualization: {str(e)}"

    def reset_view(self) -> None:
        """
        Reset the view to its default orientation.
        """
        self.vtk_widget.reset_camera()
        self.vtk_widget.view_isometric()
        self.visualizer.status = "View reset to default orientation."

    def spin_camera(self, duration=5, n_frames=150):
        """
        Spins the camera around the z-axis in an orbital path.
        """
        theta = np.linspace(0, 2 * np.pi, n_frames)
        path = np.c_[np.cos(theta), np.sin(theta), np.zeros_like(theta)]
        duration = 5

        def update_camera():
            self.visualizer.plotter.camera_position = path[self.current_frame]
            self.current_frame = (self.current_frame + 1) % n_frames

        self.timer = QTimer(self)
        self.timer.timeout.connect(update_camera)
        self.current_frame = 0

        def stop_timer():
            self.timer.stop()
            self.update_status_display("Spin complete.")

        self.update_status_display("Spinning scene...")
        interval = int(duration * 1000 / n_frames)
        self.timer.start(interval)
        QTimer.singleShot(duration * 1000, stop_timer)


def get_theme() -> str:
    """
    Determine the display theme for the current operating system.
    """
    system: str = platform.system()

    def _get_macos_theme() -> str:
        script: str = """
        tell application "System Events"
            tell appearance preferences
                if dark mode is true then
                    return "dark"
                else
                    return "light"
                end if
            end tell
        end tell
        """
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            if result.returncode == 0:
                theme: str = result.stdout.strip().lower()
                if theme in ["dark", "light"]:
                    return theme
        except subprocess.CalledProcessError as e:
            logger.error("Failed to get macOS theme: %s", e.stderr)
        except (OSError, ValueError) as e:
            logger.error("Error getting macOS theme: %s", e)
        return "light"

    def _get_windows_theme() -> str:
        try:
            from winreg import (
                HKEY_CURRENT_USER,
                CloseKey,
                ConnectRegistry,
                OpenKey,
                QueryValueEx,
            )

            registry = ConnectRegistry(None, HKEY_CURRENT_USER)
            key_path: str = (
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
            )
            key = OpenKey(registry, key_path)
            try:
                value, _ = QueryValueEx(key, "AppsUseLightTheme")
                return "dark" if value == 0 else "light"
            finally:
                CloseKey(key)
        except ImportError:
            logger.warning("winreg module not available")
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            OSError,
            ValueError,
        ) as e:
            logger.error("Failed to get Windows theme: %s", e)
        return "light"

    def _get_linux_theme() -> str:
        try:
            result = subprocess.run(
                ["gsettings", "get", "org.gnome.desktop.interface", "gtk-theme"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            if result.returncode == 0 and "dark" in result.stdout.strip().lower():
                return "dark"
        except subprocess.CalledProcessError as e:
            logger.error("Subprocess error while getting Linux theme: %s", e)
        except FileNotFoundError as e:
            logger.error("Command not found while getting Linux theme: %s", e)
        except OSError as e:
            logger.error("OS error while getting Linux theme: %s", e)
        return "light"

    theme_getters: Dict[str, callable] = {
        "Darwin": _get_macos_theme,
        "Windows": _get_windows_theme,
        "Linux": _get_linux_theme,
    }

    return theme_getters.get(system, lambda: "light")()


def set_pyvista_theme(theme: str, verbose: bool = False) -> str:
    """
    Set the PyVista theme based on the provided theme parameter.
    """
    _theme: str = get_theme()

    match theme.lower():
        case "auto":
            if _theme == "light":
                pv.set_plot_theme("document")
            else:
                pv.set_plot_theme("dark")
                _theme = "dark"
        case "light":
            pv.set_plot_theme("document")
        case "dark":
            pv.set_plot_theme("dark")
            _theme = "dark"
        case _:
            raise ValueError("Invalid theme. Must be 'auto', 'light', or 'dark'.")

    dpi_scale: float = get_platform_dpi_scale()

    base_font_size: int = FONTSIZE
    base_title_size: int = FONTSIZE + 2

    pv.global_theme.font.size = int(base_font_size / dpi_scale)
    pv.global_theme.font.title_size = int(base_title_size / dpi_scale)

    if verbose:
        logger.info("PyVista theme set to: %s", _theme.lower())
        logger.info(
            "Font size set to: %s (base: %s, scale: %s)",
            pv.global_theme.font.size,
            base_font_size,
            dpi_scale,
        )

    return _theme


def get_platform_dpi_scale() -> float:
    """
    Get the DPI scale factor based on the current platform.
    """
    if sys.platform.startswith("win"):
        return 1.25
    elif sys.platform.startswith("darwin"):
        return 2.0
    else:
        return 1.0


if __name__ == "__main__":
    app: QApplication = QApplication([])
    visualizer: RepositoryVisualizer = RepositoryVisualizer(plotter=None)
    window: MainWindow = MainWindow(visualizer)
    visualizer.plotter = window.vtk_widget
    window.resize(1200, 800)
    window.show()
    app.exec_()
