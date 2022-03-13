import random
from typing import Tuple, List
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from PIL import ImageDraw, Image


def random_color() -> Tuple[int, ...]:
    """Return a random color

    Args:
        None

    Returns:
        Tuple[int,...]: A random RGB colour.
    """
    red = random.randrange(0, 255)
    blue = random.randrange(0, 255)
    green = random.randrange(0, 255)
    return (red, blue, green)


def scale_points(points: np.ndarray, width: int, height: int) -> List[List[int]]:
    """Scale the points to the size of the image

    Args:
        points (np.ndarray): Points to be scaled.
        width (int): The width of the image.
        height (int): The height of the image.

    Returns:
        List[List[int]]: The list of scaled points.
    """
    scaled_points = []
    for x, y in points:
        x = x * width
        y = y * height
        scaled_points.append([x, y])
    return scaled_points


def generate_voronoi_diagram(
    num_cells, width, height
) -> Tuple[Voronoi, List[List[int]]]:
    """Generate voronoi diagram as polygons

    Args:
        num_cells (int): The number of points used for
            the Voronoi diagram.
        width (int): The width of the image.
        height (int): The height of the image.

    Return:
        Tuple[Voronoi, List[List[int]]]: The Voronoi diagram
    """
    # Make up data points
    points = np.random.rand(num_cells - 4, 2)
    default = np.array(
        [
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([1.0, 1.0]),
        ]
    )
    points = np.concatenate((points, default), axis=0)
    # Scale them
    points = scale_points(points, width, height)
    # Compute Voronoi tesselation
    vor = Voronoi(points)
    # Plot
    voronoi_plot_2d(vor)
    return vor, points


def get_color_of_point(point: Tuple, rgb_im: Image, width: int, height: int) -> Tuple:
    """Get the color of specific point.

    Args:
        point (Tuple): The point in the image from which
            we want to extract the colour.
        rgb_im (Image): The image from which we want to get
            the colour.
        width (int): The width of the image.
        height (int): The height of the image.

    Returns:
        Tuple: The value of the colour.
    """
    x = int(point[0])
    y = int(point[1])
    new_point = (x, y)
    try:
        return rgb_im.getpixel(new_point)
    except:
        new_point = list(new_point)
        if new_point[0] == width:
            new_point[0] -= 1
        if new_point[1] == height:
            new_point[1] -= 1
        new_point = tuple(new_point)
        return rgb_im.getpixel(new_point)


def makeup_polygons(
    draw: ImageDraw,
    num_cells: int,
    width: int,
    height: int,
    rgb_im: Image,
    random: bool,
):
    """Makeup and draw polygons for a Voronoi diagram

    Args:
        draw (ImageDraw): `ImageDraw` object to draw new
            Voronoi diagram.
        num_cells (int): Number of random points that will
            be used for the Voronoi diagram.
        width (int): Width of the image.
        height (int): Height of the image.
        rgb_im (Image): Original image, in RBG format.
        random (bool): Whether to use random colours or not.

    Returns:
        None
    """
    voronoi, points = generate_voronoi_diagram(num_cells, width, height)
    for point, index in zip(points, voronoi.point_region):
        # Getting the region of the given point
        region = voronoi.regions[index]
        # Getting the points in arrays
        polygon = list()
        for i in region:
            # If vector is out of plot do not add
            if i != -1:
                polygon.append(voronoi.vertices[i])
        # Make tuples of the points
        polygon_tuples = list()
        for l in polygon:
            polygon_tuples.append(tuple(l))
        rgb = (0, 0, 0)
        if random:
            # Get random color
            rgb = random_color()
        else:
            # Get colors of the middle point
            rgb = get_color_of_point(point, rgb_im, width, height)
        # Draw the calculated polygon with the color of the middle point
        if polygon and polygon_tuples:
            draw.polygon(polygon_tuples, rgb)
