# pylint: disable=C0301,C0116,C0115,W0613,E0611,C0413

"""
Module: visualize_repository_qt

A PyQt5 application for 3D visualization of a Python repository's structure.
Parses classes, methods, and functions, rendering them as 3D objects using PyVista.

Key Features:
- Extracts repository structure using AST.
- Visualizes classes (red dodecahedrons), methods (blue spheres), and functions (green cylinders).
- Interactive UI for customizing and saving visualizations (HTML, PNG, JPEG).

Usage:
Run: python visualize_repository_qt.py

Author: Eric G. Suchanek, PhD
Last modified: 2025-05-03
"""

import ast
import logging
import os
import sys
from pathlib import Path

import numpy as np
import param
import pyvista as pv
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
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from rich.logging import RichHandler

# Constants
ORIGIN = (0, 0, 0)
DEFAULT_REP = "/Users/egs/repos/proteusPy"
DEFAULT_PACKAGE_NAME = os.path.basename(DEFAULT_REP)
DEFAULT_SAVE_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
DEFAULT_SAVE_NAME = f"{DEFAULT_PACKAGE_NAME}_3d_visualization"

# Configure logging
logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger()


# Check PyQt5
def can_import(module_name):
    try:
        return __import__(module_name)
    except ImportError:
        return None


if can_import("PyQt5") is None:
    sys.exit("This program requires PyQt5. Install: pip install proteusPy[pyqt5]")


def parse_file(file_path):
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
    elements = []
    seen_classes = set()
    seen_functions = set()
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                file_elements = parse_file(os.path.join(root, file))
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


def fibonacci_sphere(samples, radius=1.0, center=None):
    if center is None:
        center = np.array([0, 0, 0])
    if samples <= 0:
        return []
    if samples == 1:
        return [center + radius * np.array([0, 0, 1])]

    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2
        radius_at_y = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        points.append(center + radius * np.array([x, y, z]))
    return points


def create_3d_visualization(
    visualizer,
    elements,
    save_path,
    save_format="html",
    class_radius=4.0,
    member_radius_scale=1.0,
    old_title="",
    plotter=None,
):
    visualizer.status = "Setting up visualization..."
    QApplication.processEvents()

    # Reset plotter
    plotter.clear_actors()
    plotter.remove_all_lights()
    plotter.renderer.ResetCamera()
    plotter.renderer_layer = 0

    plotter.disable_parallel_projection()
    plotter.enable_anti_aliasing("msaa")
    plotter.set_background("lightgray")
    plotter.add_axes()
    plotter.add_light(pv.Light(position=(50, 100, 100), color="white", intensity=1.0))
    plotter.add_light(
        pv.Light(position=(-50, -100, -100), color="white", intensity=0.1)
    )
    plotter.add_light(pv.Light(position=(0, 0, 100), color="white", intensity=0.5))

    package_center = np.array([0, 0, 0])
    package_name = Path(save_path).stem
    package_mesh = pv.Icosahedron(center=package_center, radius=0.8)
    plotter.add_mesh(
        package_mesh, color="purple", show_edges=False, smooth_shading=False
    )

    num_classes = len([e for e in elements if e["type"] == "class"])
    visualizer.status = f"Rendering {num_classes} classes..."
    QApplication.processEvents()
    class_positions = fibonacci_sphere(
        num_classes, radius=class_radius, center=package_center
    )
    class_index = 0
    for element in elements:
        if element["type"] != "class":
            continue
        pos = class_positions[class_index]
        class_index += 1
        mesh = pv.Dodecahedron(radius=0.75 / 2, center=pos)
        plotter.add_mesh(mesh, color="red", show_edges=False, smooth_shading=False)
        line = pv.Cylinder(
            radius=0.025,
            height=np.linalg.norm(pos - package_center),
            center=(pos + package_center) / 2,
            direction=pos - package_center,
        )
        plotter.add_mesh(line, color="red", show_edges=False, smooth_shading=True)

    num_functions = len([e for e in elements if e["type"] == "function"])
    if num_functions > 0:
        visualizer.status = f"Rendering {num_functions} functions..."
        QApplication.processEvents()
        function_positions = fibonacci_sphere(
            num_functions, radius=class_radius * 0.4, center=package_center
        )
        for i, element in enumerate([e for e in elements if e["type"] == "function"]):
            pos = function_positions[i]
            mesh = pv.Cylinder(
                radius=0.15 / 2, height=0.15 / 2, center=pos, direction=(0, 0, 1)
            )
            plotter.add_mesh(mesh, color="green", show_edges=False, smooth_shading=True)
            line = pv.Line(package_center, pos)
            plotter.add_mesh(line, color="green", line_width=1)
            plotter.reset_camera()

    visualizer.status = "Rendering methods..."
    QApplication.processEvents()
    for class_pos, class_elem in zip(
        class_positions, [e for e in elements if e["type"] == "class"]
    ):
        members = class_elem.get("methods", [])
        if members:
            method_positions = fibonacci_sphere(
                len(members), radius=member_radius_scale * 0.75, center=class_pos
            )
            for j, _ in enumerate(members):
                sphere = pv.Sphere(radius=0.225 / 2, center=method_positions[j])
                plotter.add_mesh(
                    sphere, color="blue", show_edges=False, smooth_shading=True
                )
                line = pv.Line(class_pos, method_positions[j])
                plotter.add_mesh(line, color="blue", line_width=1)

    plotter.reset_camera()
    visualizer.status = "Finalizing scene..."
    QApplication.processEvents()

    num_methods = sum(
        len(e.get("methods", [])) for e in elements if e["type"] == "class"
    )

    title_text = f"3D Visualization: {package_name} | Classes: {num_classes} | Methods: {num_methods} | Functions: {num_functions}"

    plotter.reset_camera()
    plotter.render()

    return plotter, title_text


class RepositoryVisualizer(param.Parameterized):
    repo_path = param.String(default=DEFAULT_REP, doc="Repository path")
    save_path = param.String(default=DEFAULT_PACKAGE_NAME, doc="Save path")
    old_title = param.String(
        default=f"{DEFAULT_PACKAGE_NAME} 3d visualization", doc="Visualization title"
    )
    window_title = param.String(
        default=f"{DEFAULT_PACKAGE_NAME} 3d visualization", doc="Window title"
    )
    save_format = param.Selector(
        objects=["html", "png", "jpeg"], default="html", doc="Output format"
    )
    class_radius = param.Number(
        default=4.0, bounds=(1.0, 10.0), step=0.5, doc="Class radius"
    )
    member_radius_scale = param.Number(
        default=1.25, bounds=(0.5, 3.0), step=0.25, doc="Member radius scale"
    )
    available_classes = param.List(default=[], doc="Available classes")
    selected_classes = param.ListSelector(
        default=[], objects=[], doc="Selected classes"
    )
    available_functions = param.List(default=[], doc="Available functions")
    selected_functions = param.ListSelector(
        default=[], objects=[], doc="Selected functions"
    )
    include_functions = param.Boolean(default=True, doc="Include functions")
    status = param.String(default="Ready", doc="Status")

    def __init__(self, plotter, **params):
        super().__init__(**params)
        self.plotter = plotter
        self.elements = []
        self.update_classes()

    @param.depends("repo_path", watch=True)
    def update_classes(self):
        if os.path.exists(self.repo_path):
            self.status = "Analyzing repository..."
            self.elements = collect_elements(self.repo_path)
            class_names = sorted(
                [e["name"] for e in self.elements if e["type"] == "class"]
            )
            self.available_classes = class_names
            self.param.selected_classes.objects = class_names
            function_names = sorted(
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

    def visualize(self):
        if not os.path.exists(self.repo_path):
            self.status = "Repository path does not exist."
            return
        if not self.elements:
            self.status = "No elements found."
            return

        filtered_elements = []
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

        save_path = self.save_path
        if not save_path.endswith(f".{self.save_format}"):
            save_path = f"{save_path}.{self.save_format}"

        try:
            plotter, title_text = create_3d_visualization(
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
        except Exception as e:
            self.status = f"Error creating visualization: {str(e)}"


class MainWindow(QMainWindow):
    status_changed = pyqtSignal(str)

    def __init__(self, visualizer):
        super().__init__()
        self.visualizer = visualizer
        self.setWindowTitle(self.visualizer.window_title)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.setStyleSheet(
            """
            QWidget { font-family: Arial, sans-serif; font-size: 12px; }
            QLineEdit, QComboBox, QListWidget, QCheckBox, QPushButton, QLabel { margin: 5px; padding: 5px; }
            QLineEdit, QComboBox, QListWidget { border: 1px solid #ddd; border-radius: 3px; }
            QPushButton { background-color: '#4CAF50'; color: white; border: none; border-radius: 3px; padding: 8px; }
            QPushButton#reset { background-color: '#2196F3'; }
            QLabel { border: 1px solid #ddd; background-color: '#f5f5f5'; padding: 8px; }
            """
        )

        control_panel = QVBoxLayout()
        control_panel.setSpacing(10)
        control_panel.setContentsMargins(10, 10, 10, 10)

        control_panel.addWidget(
            QLabel(
                "<h2>Input Parameters</h2>",
                font=QFont("Arial", 14, QFont.Bold),
                styleSheet="background: transparent; border: none;",
            )
        )

        control_panel.addWidget(QLabel("Repository Path"))
        self.repo_path_input = QLineEdit(self.visualizer.repo_path)
        self.repo_path_input.setPlaceholderText("Enter repository path")
        control_panel.addWidget(self.repo_path_input)

        control_panel.addWidget(QLabel("Save Path"))
        self.save_path_input = QLineEdit(self.visualizer.save_path)
        self.save_path_input.setPlaceholderText("Enter save path")
        control_panel.addWidget(self.save_path_input)

        control_panel.addWidget(QLabel("Save Format"))
        self.save_format_select = QComboBox()
        self.save_format_select.addItems(["html", "png", "jpeg"])
        self.save_format_select.setCurrentText(self.visualizer.save_format)
        control_panel.addWidget(self.save_format_select)

        control_panel.addWidget(QLabel("Class Radius"))
        self.class_radius_slider = QSlider(Qt.Horizontal)
        self.class_radius_slider.setMinimum(10)
        self.class_radius_slider.setMaximum(100)
        self.class_radius_slider.setValue(int(self.visualizer.class_radius * 10))
        self.class_radius_slider.setTickInterval(5)
        self.class_radius_slider.setTickPosition(QSlider.TicksBelow)
        control_panel.addWidget(self.class_radius_slider)

        control_panel.addWidget(QLabel("Member Radius Scale"))
        self.member_radius_scale_slider = QSlider(Qt.Horizontal)
        self.member_radius_scale_slider.setMinimum(5)
        self.member_radius_scale_slider.setMaximum(30)
        self.member_radius_scale_slider.setValue(
            int(self.visualizer.member_radius_scale * 10)
        )
        self.member_radius_scale_slider.setTickInterval(2)
        self.member_radius_scale_slider.setTickPosition(QSlider.TicksBelow)
        control_panel.addWidget(self.member_radius_scale_slider)

        control_panel.addWidget(
            QLabel(
                "<h2>Class Selection</h2>",
                font=QFont("Arial", 14, QFont.Bold),
                styleSheet="background: transparent; border: none;",
            )
        )
        control_panel.addWidget(QLabel("Select classes (empty for all):"))
        self.class_selector = QListWidget()
        self.class_selector.setSelectionMode(QListWidget.MultiSelection)
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
        # Update other QLabel widgets to have no background
        for widget in control_panel.findChildren(QLabel):
            widget.setStyleSheet("background: transparent; border: none;")

        self.include_functions_checkbox = QCheckBox("Include Functions")
        self.include_functions_checkbox.setChecked(self.visualizer.include_functions)
        control_panel.addWidget(self.include_functions_checkbox)
        control_panel.addWidget(QLabel("Select functions (empty for all):"))
        self.function_selector = QListWidget()
        self.function_selector.setSelectionMode(QListWidget.MultiSelection)
        for item in self.visualizer.available_functions:
            self.function_selector.addItem(item)
        control_panel.addWidget(self.function_selector)

        control_panel.addStretch()

        vis_panel = QVBoxLayout()
        vis_panel.setSpacing(10)
        vis_panel.setContentsMargins(10, 10, 10, 10)

        button_row = QHBoxLayout()
        self.visualize_button = QPushButton("Visualize Repository")
        self.visualize_button.setFixedWidth(200)
        button_row.addWidget(self.visualize_button)
        self.reset_camera_button = QPushButton("Reset Camera")
        self.reset_camera_button.setFixedWidth(200)
        self.reset_camera_button.setObjectName("reset")
        button_row.addWidget(self.reset_camera_button)
        self.save_button = QPushButton("Save View")
        self.save_button.setFixedWidth(200)
        button_row.addWidget(self.save_button)

        # Connect the save button to a save function
        self.save_button.clicked.connect(self.save_current_view)
        self.status_display = QLabel("Ready")
        self.status_display.setTextInteractionFlags(Qt.TextBrowserInteraction)
        button_row.addWidget(self.status_display, stretch=1)

        self.vtk_widget = QtInteractor(self)
        vis_panel.addWidget(self.vtk_widget, stretch=1)
        vis_panel.addLayout(button_row)

        control_widget = QWidget()
        control_widget.setLayout(control_panel)
        control_widget.setFixedWidth(350)
        main_layout.addWidget(control_widget)
        vis_widget = QWidget()
        vis_widget.setLayout(vis_panel)
        main_layout.addWidget(vis_widget, stretch=1)

        # Set stretch factors for the main layout
        main_layout.setStretch(0, 1)  # Control panel
        main_layout.setStretch(1, 3)  # Visualization panel

        # Enable auto-resizing for the central widget
        central_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Adjust control panel to allow vertical stretching
        control_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        control_panel.setContentsMargins(10, 10, 10, 10)
        control_panel.setSpacing(5)  # Reduce spacing to make it more compact

        # Adjust control panel to allow shrinking vertically
        control_widget.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.MinimumExpanding
        )
        control_panel.setSizeConstraint(QVBoxLayout.SetMinimumSize)

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
        self.visualize_button.clicked.connect(self.visualizer.visualize)
        self.reset_camera_button.clicked.connect(self.reset_camera)
        self.status_changed.connect(self.update_status_display)
        self.visualizer.param.watch(self.on_status_change, "status")
        self.visualizer.param.watch(self.update_class_selector, "available_classes")
        self.visualizer.param.watch(
            self.update_function_selector, "available_functions"
        )
        self.visualizer.param.watch(self.update_window_title, "window_title")

        # Remove all backgrounds from QLabel widgets
        for widget in self.findChildren(QLabel):
            widget.setStyleSheet("background: transparent; border: none;")

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
        self.visualizer.selected_classes = [
            item.text() for item in self.class_selector.selectedItems()
        ]

    def update_selected_functions(self):
        self.visualizer.selected_functions = [
            item.text() for item in self.function_selector.selectedItems()
        ]

    def update_include_functions(self, state):
        self.visualizer.include_functions = state == Qt.Checked

    def on_status_change(self, event):
        self.status_changed.emit(event.new)

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

    def update_function_selector(self, event):
        self.function_selector.clear()
        for item in event.new:
            self.function_selector.addItem(item)

    def update_window_title(self, event):
        self.setWindowTitle(event.new)

    def reset_camera(self):
        self.vtk_widget.reset_camera()
        self.vtk_widget.render()

    def save_current_view(self):
        """
        Save the current view of the visualization to the specified path and format.
        """
        save_path = self.visualizer.save_path
        save_format = self.visualizer.save_format

        if not save_path.endswith(f".{save_format}"):
            save_path = f"{save_path}.{save_format}"

        self.param.status = "Saving visualization..."
        plotter = self.plotter
        QApplication.processEvents()
        save_path = Path(save_path).with_suffix(f".{save_format}")
        try:
            if save_format == "html":
                plotter.export_html(save_path)
            elif save_format in ["png", "jpeg"]:
                screenshot_plotter = pv.Plotter(off_screen=True)
                screenshot_plotter.add_text(
                    self.param.old_title,
                    position="upper_edge",
                )
                for actor in plotter.actors.values():
                    if isinstance(actor, pv.Actor) and actor.mapper.GetInput():
                        mesh = actor.mapper.GetInput()
                        smooth_shading = actor.prop.GetInterpolation() > 0
                        screenshot_plotter.add_mesh(
                            mesh,
                            color=actor.prop.GetColor(),
                            show_edges=actor.prop.GetEdgeVisibility(),
                            line_width=actor.prop.GetLineWidth(),
                            smooth_shading=smooth_shading,
                        )
                for light in plotter.renderer.GetLights():
                    screenshot_plotter.add_light(
                        pv.Light(
                            position=light.GetPosition(),
                            color=light.GetDiffuseColor(),
                            intensity=light.GetIntensity(),
                        )
                    )
                screenshot_plotter.add_axes()
                screenshot_plotter.camera_position = plotter.camera_position
                screenshot_plotter.set_background("lightgray")
                screenshot_plotter.screenshot(save_path, window_size=[1200, 800])
                screenshot_plotter.close()
            self.param.status = f"Visualization saved to {save_path}"
        except Exception as e:
            logger.error("Failed to save: %s", e)
            self.param.status = f"Error saving visualization: {str(e)}"


if __name__ == "__main__":
    app = QApplication([])
    visualizer = RepositoryVisualizer(plotter=None)  # Plotter will be set in MainWindow
    window = MainWindow(visualizer)
    visualizer.plotter = window.vtk_widget  # Assign the MainWindow's vtk_widget
    window.resize(1200, 800)
    window.show()
    app.exec_()
