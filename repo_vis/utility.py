import ast
import logging
import os
import platform
import subprocess
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


def rotation_matrix_axis_angle(axis, angle_deg):
    """Compute a rotation matrix for a given axis and angle (in degrees)."""
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


# End of file
