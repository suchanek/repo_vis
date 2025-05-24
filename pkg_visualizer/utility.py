"""
Utility Module for Repository Visualization

This module provides utility functions and configurations to support the 3D visualization
of Python repository structures. It includes functions for parsing Python files, collecting
classes and functions, generating points on a sphere, determining system themes, and
configuring PyVista themes. Additionally, it offers a function to format Python docstrings
into Markdown for better readability.

Key Features:
- Parse Python files to extract class and function definitions using AST.
- Collect elements (classes and functions) from a repository.
- Generate points on a sphere using the Fibonacci spiral algorithm.
- Determine the system's display theme (light or dark) based on the operating system.
- Configure PyVista themes and font sizes for visualization.
- Format Python docstrings in :param: style to Markdown.

Author: Eric G. Suchanek, PhD
Last Modified: 2025-05-24 18:53:10
"""

import ast
import gc
import logging
import os
import platform
import subprocess
import types
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pyvista as pv
from rich.logging import RichHandler

FONTSIZE: int = 12

# Configure logger
logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def parse_file(file_path: str) -> List[Dict[str, Union[str, int, List[str]]]]:
    """
    Parse a Python file and extract class and function definitions using AST.

    :param file_path: Path to the Python file to parse.
    :type file_path: str
    :return: A list of dictionaries containing class and function details.
    :rtype: List[Dict[str, Union[str, int, List[str]]]]
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

    :param repo_path: Path to the repository to analyze.
    :type repo_path: str
    :return: A list of dictionaries containing class and function details.
    :rtype: List[Dict[str, Union[str, int, List[str]]]]
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

    :param samples: Number of points to generate.
    :type samples: int
    :param radius: Radius of the sphere.
    :type radius: float
    :param center: Center of the sphere as a numpy array.
    :type center: Optional[np.ndarray]
    :return: A list of numpy arrays representing points on the sphere.
    :rtype: List[np.ndarray]
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


def fibonacci_annulus(
    samples: int,
    inner_radius: float = 1.0,
    outer_radius: float = 2.0,
    center: Optional[np.ndarray] = None,
    y_thickness: float = 0.2,
) -> List[np.ndarray]:
    """
    Generate points in a ring-shaped annulus (circular disk with hole) using the Fibonacci spiral algorithm.
    Points are distributed along the XZ plane with a small Y variation for better visualization.

    :param samples: Number of points to generate.
    :type samples: int
    :param inner_radius: Inner radius of the annulus.
    :type inner_radius: float
    :param outer_radius: Outer radius of the annulus.
    :type outer_radius: float
    :param center: Center of the annulus as a numpy array.
    :type center: Optional[np.ndarray]
    :param y_thickness: Thickness of the annulus in the Y direction.
    :type y_thickness: float
    :return: A list of numpy arrays representing points in the annulus.
    :rtype: List[np.ndarray]
    """
    if center is None:
        center = np.array([0, 0, 0])
    if samples <= 0:
        return []
    if samples == 1:
        radius = (inner_radius + outer_radius) / 2
        return [center + radius * np.array([1, 0, 0])]

    points: List[np.ndarray] = []
    phi: float = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle in radians

    # Calculate radius increment for each point
    radius_range = outer_radius - inner_radius
    radius_increment = radius_range / samples

    for i in range(samples):
        # Calculate radius: gradually increases from inner to outer radius as we spiral outward
        radius = inner_radius + (i * radius_increment)

        # Calculate angle based on golden angle increment
        theta = phi * i

        # Position in XZ plane (with small Y variation for aesthetics)
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        y = (np.random.random() * 2 - 1) * y_thickness  # Small random variation in Y

        points.append(center + np.array([x, y, z]))

    return points


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


def get_theme() -> str:
    """
    Determine the display theme for the current operating system.

    :return: The system's display theme ('light' or 'dark').
    :rtype: str
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
        except subprocess.CalledProcessError:
            pass
        except (OSError, ValueError):
            pass
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
            pass
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            OSError,
            ValueError,
        ):
            pass
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
        except subprocess.CalledProcessError:
            pass
        except FileNotFoundError:
            pass
        except OSError:
            pass
        return "light"

    theme_getters: Dict[str, callable] = {
        "Darwin": _get_macos_theme,
        "Windows": _get_windows_theme,
        "Linux": _get_linux_theme,
    }

    return theme_getters.get(system, lambda: "light")()


def get_platform_dpi_scale() -> float:
    """
    Get the DPI scale factor based on the current platform.

    :return: The DPI scale factor.
    :rtype: float
    """
    if platform.system().startswith("Windows"):
        return 1.25
    elif platform.system().startswith("Darwin"):
        return 2.0
    else:
        return 1.0


def set_pyvista_theme(theme: str, verbose: bool = False) -> str:
    """
    Set the PyVista theme based on the provided theme parameter.

    :param theme: The desired theme ('auto', 'light', or 'dark').
    :type theme: str
    :param verbose: Whether to log detailed information about the theme configuration.
    :type verbose: bool
    :return: The applied theme ('light' or 'dark').
    :rtype: str
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


def rotation_matrix_axis_angle(axis: List[float], angle_deg: float) -> np.ndarray:
    """
    Compute a rotation matrix for a given axis and angle (in degrees).

    :param axis: The axis of rotation as a list of three floats.
    :type axis: List[float]
    :param angle_deg: The angle of rotation in degrees.
    :type angle_deg: float
    :return: A 3x3 numpy array representing the rotation matrix.
    :rtype: np.ndarray
    """
    angle_rad = np.radians(angle_deg)
    axis = np.array(axis) / np.linalg.norm(axis)  # Normalize axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    t = 1 - c
    x, y, z = axis
    return np.array(
        [
            [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
            [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
            [t * x * z - y * s, t * y * z + x * s, t * z * z + c],
        ]
    )


def format_docstring_to_markdown(docstring: str) -> str:
    """
    Convert a Python docstring in :param: style to a Markdown-formatted string.

    :param docstring: The input docstring to format.
    :type docstring: str
    :return: A Markdown-formatted string.
    :rtype: str
    """
    import re

    if not docstring:
        return "No docstring available."

    # Split the docstring into lines
    lines = docstring.strip().split("\n")

    # Initialize Markdown components
    markdown_lines = []
    markdown_lines.append(f"# {lines[0]}")  # Title from the first line

    # Process the rest of the lines
    for line in lines[1:]:
        line = line.strip()
        if line.startswith(":param"):
            match = re.match(r":param (\w+): (.+)", line)
            if match:
                param_name, param_desc = match.groups()
                markdown_lines.append(f"- **{param_name}**: {param_desc}")
        elif line.startswith(":type") or line.startswith(":rtype"):
            continue  # Skip type annotations
        elif line.startswith(":return:"):
            return_desc = line.replace(":return:", "**Returns:**")
            markdown_lines.append(return_desc)
        else:
            markdown_lines.append(line)

    return "\n".join(markdown_lines)


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
                if isinstance(obj, pv.MultiBlock) and hasattr(obj, "clear_all_data"):
                    try:
                        # Use clear_all_data() for MultiBlock objects
                        logger.debug("Clearing MultiBlock data with clear_all_data()")
                        obj.clear_all_data()
                    except Exception as e:
                        logger.debug("Error clearing MultiBlock data: %s", e)
                elif isinstance(obj, pv.PolyData) or isinstance(obj, pv.MultiBlock):
                    try:
                        # Monkey patch the object's __del__ method to prevent errors
                        if hasattr(obj, "__del__"):
                            obj.__del__ = types.MethodType(lambda self: None, obj)

                        # Set object attributes to None to break circular references
                        for attr_name in dir(obj):
                            if not attr_name.startswith("__"):
                                try:
                                    setattr(obj, attr_name, None)
                                    logger.debug(
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


# End of file
