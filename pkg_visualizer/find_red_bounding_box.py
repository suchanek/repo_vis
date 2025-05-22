"""
Module: find_red_bounding_box

Utility functions to identify and handle red bounding box cubes in PyVista plotters.
These functions help detect and optionally remove unwanted bounding boxes that
may appear in visualizations.

Author: Eric G. Suchanek, PhD
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pyvista as pv

# Configure logging
logger = logging.getLogger(__name__)


def find_red_bounding_box(
    plotter: pv.Plotter, remove: bool = False, tolerance: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Examine a plotter for red cubes that might represent bounding boxes.

    This function inspects all actors in the provided plotter and identifies
    any that are red cubes, which often represent unwanted bounding boxes.
    It can optionally remove these actors from the plotter.

    Parameters:
    -----------
    plotter : pv.Plotter
        The PyVista plotter to examine
    remove : bool, optional
        Whether to remove identified red bounding boxes, default is False
    tolerance : float, optional
        Color matching tolerance, default is 0.1

    Returns:
    --------
    List[Dict[str, Any]]
        A list of dictionaries containing information about each red bounding box found:
        - 'name': The actor name
        - 'actor': The actor object
        - 'color': The actor color
        - 'bounds': The actor bounds
        - 'removed': Whether the actor was removed (if remove=True)
    """
    if not plotter:
        logger.error("No plotter provided")
        return []

    red_boxes = []

    # Log the current actors in the plotter
    logger.debug("Examining actors in plotter: %s", list(plotter.actors.keys()))

    # Check each actor in the plotter
    for actor_name, actor in list(plotter.actors.items()):
        is_red_box = False
        actor_info = {
            "name": actor_name,
            "actor": actor,
            "color": None,
            "bounds": None,
            "removed": False,
        }

        # Check if the actor has a color property and is red
        if hasattr(actor, "GetProperty") and actor.GetProperty():
            color = actor.GetProperty().GetColor()
            actor_info["color"] = color

            # Check if the color is red (close to [1,0,0])
            if np.allclose(color, [1, 0, 0], atol=tolerance):
                logger.debug("Found red actor: %s with color %s", actor_name, color)
                is_red_box = True

        # Check if the actor has a mapper and input
        if hasattr(actor, "GetMapper") and actor.GetMapper():
            mapper = actor.GetMapper()
            if hasattr(mapper, "GetInput") and mapper.GetInput():
                input_obj = mapper.GetInput()

                # Get the bounds of the input object
                if hasattr(input_obj, "bounds"):
                    bounds = input_obj.bounds
                    actor_info["bounds"] = bounds
                    logger.debug("Actor %s bounds: %s", actor_name, bounds)

                # Check if it's a cube-like shape (all dimensions similar)
                if hasattr(input_obj, "bounds") and is_red_box:
                    bounds = input_obj.bounds
                    x_size = abs(bounds[1] - bounds[0])
                    y_size = abs(bounds[3] - bounds[2])
                    z_size = abs(bounds[5] - bounds[4])

                    # Check if it's roughly cube-shaped (all dimensions similar)
                    avg_size = (x_size + y_size + z_size) / 3
                    if (
                        abs(x_size - avg_size) / avg_size < 0.2
                        and abs(y_size - avg_size) / avg_size < 0.2
                        and abs(z_size - avg_size) / avg_size < 0.2
                    ):
                        logger.debug("Actor %s is cube-shaped", actor_name)
                        is_red_box = True
                    else:
                        is_red_box = False

                # Check if it's a bounding box or outline by name
                if (
                    "bounds" in actor_name.lower()
                    or "outline" in actor_name.lower()
                    or "box" in actor_name.lower()
                ):
                    logger.debug(
                        "Actor %s has 'bounds', 'outline', or 'box' in its name",
                        actor_name,
                    )
                    if is_red_box:
                        logger.debug("Confirmed %s is a red bounding box", actor_name)

        # If we've identified a red box and remove is True, remove it
        if is_red_box:
            red_boxes.append(actor_info)
            if remove:
                try:
                    plotter.remove_actor(actor, reset_camera=False)
                    actor_info["removed"] = True
                    logger.info("Removed red bounding box actor: %s", actor_name)
                except Exception as e:
                    logger.error("Failed to remove actor %s: %s", actor_name, str(e))

    if not red_boxes:
        logger.info("No red bounding boxes found in the plotter")
    else:
        logger.info("Found %d red bounding box actors", len(red_boxes))

    return red_boxes


def disable_bounding_box_generation(plotter: pv.Plotter) -> bool:
    """
    Attempt to disable automatic bounding box generation in a plotter.

    This function tries various methods to prevent bounding boxes from being
    automatically generated in the plotter.

    Parameters:
    -----------
    plotter : pv.Plotter
        The PyVista plotter to modify

    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Try to disable bounding box in the picker
        if hasattr(plotter, "picker") and plotter.picker:
            picker = plotter.picker

            # Disable bounding box if the method exists
            if hasattr(picker, "ShowBoundingBox"):
                logger.info("Disabling bounding box in picker")
                picker.ShowBoundingBox(False)

            # Set pick from list to 0 to avoid bounding box issues
            if hasattr(picker, "SetPickFromList"):
                picker.SetPickFromList(0)

            # Set tolerance to a small value
            if hasattr(picker, "SetTolerance"):
                picker.SetTolerance(0.005)

        # Try to disable bounding box in the renderer
        if hasattr(plotter, "renderer"):
            renderer = plotter.renderer
            if hasattr(renderer, "SetUseDepthPeeling"):
                renderer.SetUseDepthPeeling(True)

        # Remove any existing bounding boxes
        plotter.remove_bounding_box()

        # Remove any actors with "bounds" or "outline" in their names
        for actor_name in list(plotter.actors.keys()):
            if (
                "bounds" in actor_name.lower()
                or "outline" in actor_name.lower()
                or "box" in actor_name.lower()
            ):
                plotter.remove_actor(actor_name, reset_camera=False)

        logger.info("Successfully disabled bounding box generation")
        return True

    except Exception as e:
        logger.error("Failed to disable bounding box generation: %s", str(e))
        return False


def remove_all_red_actors(plotter: pv.Plotter) -> int:
    """
    Remove all red actors from a plotter regardless of shape.

    This is a more aggressive approach that removes any actor with a red color.
    Specifically excludes actors named "functions" to preserve yellow function objects.

    Parameters:
    -----------
    plotter : pv.Plotter
        The PyVista plotter to modify

    Returns:
    --------
    int
        The number of actors removed
    """
    if not plotter:
        return 0

    count = 0
    for actor_name, actor in list(plotter.actors.items()):
        # Skip the "functions" actor which contains yellow function objects
        if actor_name == "functions":
            logger.debug(
                "Skipping 'functions' actor to preserve yellow function objects"
            )
            continue

        if hasattr(actor, "GetProperty") and actor.GetProperty():
            color = actor.GetProperty().GetColor()
            # Check if the color is red (with a stricter tolerance)
            if np.allclose(color, [1, 0, 0], atol=0.05):
                try:
                    plotter.remove_actor(actor, reset_camera=False)
                    logger.info("Removed red actor: %s", actor_name)
                    count += 1
                except Exception as e:
                    logger.error("Failed to remove actor %s: %s", actor_name, str(e))

    return count
