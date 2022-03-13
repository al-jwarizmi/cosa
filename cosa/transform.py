import random
from pathlib import Path
import numpy as np
from numpy.random import RandomState
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from cosa.utils import makeup_polygons


def k_representative_pallette(
    image: np.ndarray, num_clusters: int = random.randint(4, 5)
) -> Image:
    """Cluster pixels into num_clusters representative colors
    and transform image into to this discrete palette
    """
    ysize = image.shape[0]
    xsize = image.shape[1]
    new_image = np.reshape(image, (xsize * ysize, 3))
    model = KMeans(n_clusters=num_clusters, random_state=42).fit(new_image)
    colors = []
    for i in range(num_clusters):
        colors.append(np.median(new_image[model.labels_ == i], axis=0))
        colors[i] = colors[i] / 255
    new_image = np.zeros((xsize * ysize, 3))
    for i in range(num_clusters):
        slicer = model.labels_ == i
        new_image[slicer] = colors[i]
    new_image = np.reshape(new_image, (ysize, xsize, 3))
    new_image = Image.fromarray((new_image * 255).astype(np.uint8))
    return new_image


def elastic_transform(
    image: np.ndarray,
    alpha: int = random.randint(900, 1000),
    sigma: int = random.randint(4, 8),
    random_state: RandomState = None,
) -> Image:
    """Elastic deformation of images as described in [Simard2003]
    Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual Document
    Analysis", in Proc. of the International Conference on
    Document Analysis and Recognition, 2003.
    """
    if random_state is None:
        random_state = RandomState(None)
    shape = image.shape
    dx = (
        gaussian_filter(
            (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
        )
        * alpha
    )
    dy = (
        gaussian_filter(
            (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
        )
        * alpha
    )
    x, y, z = np.meshgrid(
        np.arange(shape[1]),
        np.arange(shape[0]),
        np.arange(shape[2]),
    )
    indices = (
        np.reshape(y + dy, (-1, 1)),
        np.reshape(x + dx, (-1, 1)),
        np.reshape(z, (-1, 1)),
    )
    distored_image = map_coordinates(image, indices, order=1, mode="reflect")
    dist_reshaped = distored_image.reshape(image.shape)
    dist_reshaped = Image.fromarray(dist_reshaped)
    return dist_reshaped


def jpeg(image: np.ndarray, iterations: int = 100) -> Image:
    """Applies a JPEG compression `iterations` times.
    This is inspired by the `JPEG Bot`. For more info,
    see:
    `https://mikewatson.me/bots/JPEGBot`
    """
    im = Image.fromarray(image)
    temp_file_path = "/tmp/jpeg_transform_cosa.jpeg"
    temp_file_path = Path(temp_file_path)
    temp_file_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(temp_file_path, format="JPEG", quality=100)
    for i in range(100):
        im = Image.open(temp_file_path)
        im.save(temp_file_path, format="JPEG", quality=115 - i)
    return Image.open(temp_file_path)


def voronoi(image: np.ndarray, num_cells: int = 3000) -> Image:
    """Creates a Voronoi diagram from a given image.
    This code was originally obtained from:
    `https://github.com/Stunkymonkey/voronoi-image`
    and adapted for COSA.

    Args:
        - image (np.ndarray): The image on which the
            transformation will be applied.
        - num_cells (int): The number of (random) points
            the Voronoi diagram will have.

    Returns:
        - Image: The image after the transformation.
    """
    # Load image and parameters
    im = Image.fromarray(image).convert("RGB")
    rgb_im = im.convert("RGB")
    width, height = im.size
    # Assert parameter values
    if num_cells > ((width * height) / 10):
        raise ValueError(
            "Sorry your image ist too small, or you want to many polygons."
        )
    assert num_cells > 5, "There must be at least 6 cells!"
    # Build Voronoi image
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    makeup_polygons(draw, num_cells, width, height, rgb_im, False)
    return image
