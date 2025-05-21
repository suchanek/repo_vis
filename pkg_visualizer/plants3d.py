"""plants3d.py

A module for generating and rendering 3D plant models (Sunflower and Giant Allium) using PyVista and PyQt5.

Classes:
    Sunflower: 3D sunflower model with petals, seeds, center, and stem.
    GiantAllium: 3D giant allium model with a spherical head of florets and a tall stem.

Dependencies:
    numpy, pyvista, pyvistaqt, PyQt5

Author: Eric G. Suchanek, PhD.
Last updated: 2025-05-09 19:24:35
"""

import argparse
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pyvista as pv
from PyQt5 import QtWidgets
from pyvistaqt import BackgroundPlotter
from utility import fibonacci_sphere, set_pyvista_theme


class Sunflower:
    """Builds and manages a 3D sunflower model composed of petals, seeds, center, and stem.

    :param plotter: Optional PyVista BackgroundPlotter instance.
    :type plotter: BackgroundPlotter or None
    :param petal_length: Length of each petal.
    :type petal_length: float
    :param petal_width: Width of each petal.
    :type petal_width: float
    :param petal_thickness: Thickness of each petal.
    :type petal_thickness: float
    :param center_radius: Radius of the sunflower center.
    :type center_radius: float
    :param center_height: Height of the sunflower center cylinder.
    :type center_height: float
    :param num_seeds: Number of seeds to arrange in the center.
    :type num_seeds: int
    :param stem_height: Height of the stem.
    :type stem_height: float
    :param stem_radius: Radius of the stem.
    :type stem_radius: float
    :param num_petals: Number of petals to generate.
    :type num_petals: int
    :param tilt_angle: Tilt angle of the flower head in degrees.
    :type tilt_angle: float
    :param showgrid: Whether to show the grid in the plot.
    :type showgrid: bool
    :param show_axes: Whether to show the axes in the plot.
    :type show_axes: bool
    """

    def __init__(
        self,
        plotter=None,
        petal_length: float = 2.5,
        petal_width: float = 0.8,
        petal_thickness: float = 0.05,
        center_radius: float = 2.0,
        center_height: float = 0.3,
        num_seeds: int = 300,
        stem_height: float = 5,
        stem_radius: float = 0.2,
        num_petals: int = 24,
        tilt_angle: float = 15,
        showgrid: bool = False,
        show_axes: bool = True,
    ) -> None:
        """Initialize internal parameters and assemble the sunflower model.

        :param plotter: Optional BackgroundPlotter instance.
        :type plotter: BackgroundPlotter or None
        :param petal_length: Length of each petal.
        :type petal_length: float
        :param petal_width: Width of each petal.
        :type petal_width: float
        :param petal_thickness: Thickness of each petal.
        :type petal_thickness: float
        :param center_radius: Radius of the sunflower center.
        :type center_radius: float
        :param center_height: Height of the center cylinder.
        :type center_height: float
        :param num_seeds: Number of seeds to generate.
        :type num_seeds: int
        :param stem_height: Height of the stem.
        :type stem_height: float
        :param stem_radius: Radius of the stem.
        :type stem_radius: float
        :param num_petals: Total number of petals.
        :type num_petals: int
        :param tilt_angle: Tilt angle of the flower head in degrees.
        :type tilt_angle: float
        :param showgrid: Whether to show the grid in the plot.
        :type showgrid: bool
        :param show_axes: Whether to show the axes in the plot.
        :type show_axes: bool
        """
        self.petal_length = petal_length
        self.petal_width = petal_width
        self.petal_thickness = petal_thickness
        self.center_radius = center_radius
        self.center_height = center_height
        self.num_seeds = num_seeds
        self.stem_height = stem_height
        self.stem_radius = stem_radius
        self.num_petals = num_petals
        self.tilt_angle = tilt_angle
        self.showgrid = showgrid
        self.show_axes = show_axes
        self.plotter = self.assemble(plotter)

    def create_petal(self) -> pv.PolyData:
        """Create a single petal mesh and extrude it to the specified thickness.

        :return: A PolyData mesh of the petal.
        :rtype: pv.PolyData
        """
        x = np.linspace(0, self.petal_length, 20)
        y = (
            np.sin(np.pi * x / self.petal_length)
            * self.petal_width
            * (1 - 0.3 * (x / self.petal_length))
        )
        z = np.zeros_like(x)
        pts = np.vstack((x, y, z)).T
        pts = np.vstack((pts, np.vstack((x, -y, z)).T[::-1]))
        pts[:, 2] = np.sin(np.pi * pts[:, 0] / self.petal_length) * 0.4
        cloud = pv.PolyData(pts)
        surf = cloud.delaunay_2d()
        return surf.extrude([0, 0, self.petal_thickness], capping=True)

    def create_center(self) -> pv.Cylinder:
        """Create the center cylinder mesh of the sunflower.

        :return: A Cylinder mesh representing the flower center.
        :rtype: pv.Cylinder
        """
        return pv.Cylinder(
            center=(0, 0, 0),
            direction=(0, 0, 1),
            radius=self.center_radius,
            height=self.center_height,
            resolution=50,
        )

    def create_seeds(self) -> list:
        """Generate seed sphere meshes arranged in a golden-angle spiral.

        :return: A list of Sphere meshes representing seeds.
        :rtype: list
        """
        seeds = []
        phi = np.pi * (3 - np.sqrt(5))
        for i in range(self.num_seeds):
            r = np.sqrt(i / self.num_seeds) * self.center_radius * 0.8
            theta = phi * i
            x, y, z = r * np.cos(theta), r * np.sin(theta), 0.15
            seeds.append(
                pv.Sphere(
                    radius=0.04,
                    center=(x, y, z),
                    theta_resolution=10,
                    phi_resolution=10,
                )
            )
        return seeds

    def create_stem(self) -> pv.Cylinder:
        """Create the stem cylinder mesh of the sunflower.

        :return: A Cylinder mesh representing the stem.
        :rtype: pv.Cylinder
        """
        return pv.Cylinder(
            center=(0, 0, -self.stem_height / 2),
            direction=(0, 0, 1),
            radius=self.stem_radius,
            height=self.stem_height,
            resolution=20,
        )

    def assemble(self, plotter=None) -> BackgroundPlotter:
        """Assemble the full sunflower model into the given or a new plotter.

        :param plotter: An optional BackgroundPlotter instance.
        :type plotter: BackgroundPlotter or None
        :return: Configured BackgroundPlotter with all meshes added.
        :rtype: BackgroundPlotter
        """
        if plotter is None:
            plotter = BackgroundPlotter()
        plotter.enable_eye_dome_lighting()
        # petals
        petal = self.create_petal()
        radius_scale = 0.7
        for i in range(self.num_petals):
            angle = 2 * np.pi * i / self.num_petals
            rotated = petal.rotate_z(np.degrees(angle), inplace=False)
            x_t, y_t = radius_scale * self.center_radius * np.cos(
                angle
            ), radius_scale * self.center_radius * np.sin(angle)
            translated = rotated.translate((x_t, y_t, 0), inplace=False)
            tilted = translated.rotate_x(self.tilt_angle, inplace=False)
            plotter.add_mesh(tilted, color="#FFC107")
        # center and seeds
        center = self.create_center().rotate_x(self.tilt_angle, inplace=False)
        seeds = [
            s.rotate_x(self.tilt_angle, inplace=False) for s in self.create_seeds()
        ]
        plotter.add_mesh(center, color="#4E2A1E")
        for s in seeds:
            plotter.add_mesh(s, color="black")
        # stem
        stem = self.create_stem()
        plotter.add_mesh(stem, color="#2E7D32")
        # environment
        if self.show_axes:
            plotter.add_axes()
        if self.showgrid:
            plotter.show_grid()
        # Set default camera: y up, looking from +z
        return plotter

    def run(self) -> None:
        """Start the Qt application loop and display the plot."""
        try:
            # Get or create QApplication instance
            app = QtWidgets.QApplication.instance()
            if app is None:
                app = QtWidgets.QApplication(sys.argv)

            # Initialize plotter if not already set
            if self.plotter is None:
                self.plotter = BackgroundPlotter()
                self.plotter = (
                    self.assemble()
                )  # Ensure assemble() returns a valid plotter

            # Show the plotter
            self.plotter.show()

            # Run the application loop
            app.exec()

        except Exception as e:
            print(f"Error running Qt application: {e}")
            raise

        finally:
            # Ensure plotter is closed properly
            if hasattr(self, "plotter") and self.plotter is not None:
                try:
                    self.plotter.close()
                except Exception as e:
                    print(f"Error closing plotter: {e}")
                self.plotter = None  # Clear reference (optional, but explicit)
                logger.debug("Plotter closed successfully.")

    def show(self) -> None:
        """Display the sunflower plot without starting a new Qt event loop.

        :return: None
        :rtype: None
        """
        if self.plotter is None:
            self.plotter = BackgroundPlotter(self.plotter)
            self.plotter = self.assemble(self.plotter)
        self.plotter.show()


class GiantAllium:
    """Builds and manages a 3D giant allium model composed of a spherical head of florets and a tall stem.

    :param plotter: Optional PyVista BackgroundPlotter instance.
    :type plotter: BackgroundPlotter or None
    :param floret_radius: Radius of each small floret sphere.
    :type floret_radius: float
    :param head_radius: Radius of the allium flower head.
    :type head_radius: float
    :param num_florets: Number of florets to arrange on the head.
    :type num_florets: int
    :param stem_height: Height of the stem.
    :type stem_height: float
    :param stem_radius: Radius of the stem.
    :type stem_radius: float
    :param showgrid: Whether to show the grid in the plot.
    :type showgrid: bool
    :param show_axes: Whether to show the axes in the plot.
    :type show_axes: bool
    """

    def __init__(
        self,
        plotter: BackgroundPlotter = None,
        floret_radius: float = 0.1,
        head_radius: float = 2.0,
        num_florets: int = 400,
        stem_height: float = 7.0,
        stem_radius: float = 0.1,
        showgrid: bool = False,
        show_axes: bool = True,
    ) -> None:
        """Initialize internal parameters and assemble the allium model.

        :param plotter: Optional BackgroundPlotter instance.
        :type plotter: BackgroundPlotter or None
        :param floret_radius: Radius of each floret sphere.
        :type floret_radius: float
        :param head_radius: Radius of the allium head.
        :type head_radius: float
        :param num_florets: Number of florets to generate.
        :type num_florets: int
        :param stem_height: Height of the stem.
        :type stem_height: float
        :param stem_radius: Radius of the stem.
        :type stem_radius: float
        :param showgrid: Whether to show the grid in the plot.
        :type showgrid: bool
        :param show_axes: Whether to show the axes in the plot.
        :type show_axes: bool
        """
        self.floret_radius = floret_radius
        self.floret_positions = []
        self.head_radius = head_radius
        self.num_florets = num_florets
        self.stem_height = stem_height
        self.stem_radius = stem_radius
        self.showgrid = showgrid
        self.show_axes = show_axes
        self.plotter = self.assemble(plotter)

    def create_florets(self) -> list:
        """Generate floret sphere meshes arranged on a spherical head (z-up)."""
        florets = []
        self.floret_positions = []
        head_center = (
            0,
            0,
            self.stem_height,
        )  # Center directly above the stem, along z
        floret_positions: List[np.ndarray] = fibonacci_sphere(
            self.num_florets, radius=self.head_radius, center=head_center
        )
        for pos in floret_positions:
            florets.append(
                pv.Sphere(
                    radius=self.floret_radius,
                    center=pos,
                    theta_resolution=16,
                    phi_resolution=16,
                )
            )
            self.floret_positions.append(pos)
        return florets

    def create_floret_stems(self) -> list:
        """Create thin cylinders connecting the stem to each floret (z-up)."""
        stalks = []
        base = (0, 0, self.stem_height)
        for pos in getattr(self, "floret_positions", []):
            direction = np.array(pos) - np.array(base)
            length = np.linalg.norm(direction)
            if length == 0:
                continue
            direction = direction / length
            stalk = pv.Cylinder(
                center=(np.array(base) + np.array(pos)) / 2,
                direction=direction,
                radius=self.floret_radius * 0.3,
                height=length,
                resolution=12,
            )
            stalks.append(stalk)
        return stalks

    def create_stem(self) -> pv.Cylinder:
        """Create the stem cylinder mesh of the allium (z-up)."""
        # Stem base at z=0, top at z=self.stem_height
        center_z = self.stem_height / 2
        return pv.Cylinder(
            center=(0, 0, center_z),
            direction=(0, 0, 1),
            radius=self.stem_radius,
            height=self.stem_height,
            resolution=24,
        )

    def assemble(self, plotter: BackgroundPlotter = None) -> BackgroundPlotter:
        """Assemble the full allium model into the given or a new plotter.

        :param plotter: An optional BackgroundPlotter instance.
        :return: Configured BackgroundPlotter with all meshes added.
        :rtype: BackgroundPlotter
        """
        if plotter is None:
            plotter = BackgroundPlotter()
        plotter.enable_eye_dome_lighting()
        # florets (head)
        florets = self.create_florets()
        for f in florets:
            plotter.add_mesh(f, color="#B39DDB")  # light purple
        # floret stalks
        for stalk in self.create_floret_stems():
            plotter.add_mesh(stalk, color="#A1887F")  # light brown/gray
        # stem
        stem = self.create_stem()
        plotter.add_mesh(stem, color="#388E3C")
        # environment
        set_pyvista_theme("auto")
        if self.show_axes:
            plotter.add_axes()
        if self.showgrid:
            plotter.show_grid()
        # Set default camera: y up, looking from +z
        # plotter.camera_position = [(0, -10, 10), (0, 0, 0), (0, 1, 0)]
        plotter.reset_camera()
        return plotter

    def run(self) -> None:
        """Start the Qt application loop and display the plot."""
        try:
            # Get or create QApplication instance
            app = QtWidgets.QApplication.instance()
            if app is None:
                app = QtWidgets.QApplication(sys.argv)

            # Initialize plotter if not already set
            if self.plotter is None:
                self.plotter = BackgroundPlotter()
                self.plotter = (
                    self.assemble()
                )  # Ensure assemble() returns a valid plotter

            # Show the plotter
            self.plotter.show()

            # Run the application loop
            app.exec()

        except Exception as e:
            print(f"Error running Qt application: {e}")
            raise

        finally:
            # Ensure plotter is closed properly
            if hasattr(self, "plotter") and self.plotter is not None:
                try:
                    self.plotter.close()
                except Exception as e:
                    print(f"Error closing plotter: {e}")
                self.plotter = None  # Clear reference (optional, but explicit)

    def show(self) -> None:
        """Display the allium plot without starting a new Qt event loop.

        :return: None
        """
        if self.plotter is None:
            self.plotter = BackgroundPlotter(self.plotter)
            self.plotter = self.assemble(self.plotter)
        self.plotter.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Display a 3D Sunflower or Giant Allium."
    )
    parser.add_argument(
        "--plant",
        "-p",
        type=str,
        choices=["sunflower", "allium"],
        default="sunflower",
        help="Which plant to display: 'sunflower' or 'allium' (default: sunflower)",
    )
    args = parser.parse_args()
    set_pyvista_theme("auto")

    # app = QtWidgets.QApplication(sys.argv)
    plant_plotter = BackgroundPlotter(show=True)
    if args.plant == "sunflower":
        plant = Sunflower(plotter=plant_plotter)
    else:
        plant = GiantAllium(plotter=plant_plotter)

    plant.run()

    # end of file
