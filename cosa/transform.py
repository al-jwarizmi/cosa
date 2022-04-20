import random
from pathlib import Path
import numpy as np
from numpy.random import RandomState
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from cosa.utils import makeup_polygons
from math import sqrt


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


def dither(
    image, colors=None, colorstops=None, saveOutput=True, outputType="", showFinal=True
) -> Image:
    """
    A basic python function to dither a target image.
    Uses the Floyd-Steinberg dithering algorithm
    see: https://github.com/CarpenterD/python-dithering
    """
    # Load image and parameters
    im = Image.fromarray(image).convert("RGB")

    def getClosestColor(c, colors):
        """Returns closest color in 'colors' to target 'c'. All colors are represented
        as RGB tuples.\n
        Method runs in O(N) time, where 'N' is the size of 'colors'. \n
        PARAMETERS:\n
        \tc : Target color to be approximated, formatted as an RGB tuple\n
        \tcolors : a list containing all valid color options, each formatted as an RGB tuple\n
        RETURNS:\n
        \tnearest: the closest avaliable RGB tuple to 'c' contained within 'colors'
        """
        nearest = (0, 0, 0)  # always overridden in first iteration of for loop
        minDiff = 1000  # initialised to be greater than all possible differences
        for col in colors:
            diff = sqrt(
                (col[0] - c[0]) ** 2 + (col[1] - c[1]) ** 2 + (col[2] - c[2]) ** 2
            )
            if diff < minDiff:
                minDiff = diff
                nearest = col
        return nearest

    def clamp(x):
        """Clamps a given number between 0 and 255.\n
        PARAMETERS:\n
        \tx: Input number to be clamped\n
        RETURNS:\n
        \tclamped: The value of 'x' clamped between 0 and 255
        """
        return max(0, min(255, x))

    def applyErr(tup, err, factor):
        """Adds a percentage of quantization error to specified tuple\n
        PARAMETERS:\n
        \ttup: Three (3) dimensional tuple containing data\n
        \terr: Three (3) dimensional tuple containing quantization error\n
        \tfactor: Percentage of 'err' to be applied to 'tup'\n
        RETURNS:\n
        \t(r,g,b): Three (3) dimensional tuple containing the input data with
            specified amount of error added. Values are rounded and clamped
            between 0 and 255
        """
        r = clamp(int(tup[0] + err[0] * factor))
        g = clamp(int(tup[1] + err[1] * factor))
        b = clamp(int(tup[2] + err[2] * factor))
        return r, g, b

    mode, size = im.mode, im.size
    width, height = size[0], size[1]
    pix = list(im.getdata())
    im.close()

    COLORS = [(0, 0, 0), (0, 102, 102), (250, 250, 250)]

    # lambda expression to flatten x,y location
    index = lambda x, y: x + y * width
    # Floyd-Steinberg dithering. https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering
    for y in range(int(height)):
        for x in range(int(width)):
            old = pix[index(x, y)]
            new = getClosestColor(old, COLORS)
            pix[index(x, y)] = new
            # calculates difference in r/g/b channels
            err = (old[0] - new[0], old[1] - new[1], old[2] - new[2])

            if x != width - 1:
                pix[index(x + 1, y)] = applyErr(pix[index(x + 1, y)], err, 7 / 16)
            if y != height - 1:
                pix[index(x, y + 1)] = applyErr(pix[index(x, y + 1)], err, 5 / 16)
                if x > 0:
                    pix[index(x - 1, y + 1)] = applyErr(
                        pix[index(x - 1, y + 1)], err, 3 / 16
                    )
                if x != width - 1:
                    pix[index(x + 1, y + 1)] = applyErr(
                        pix[index(x + 1, y + 1)], err, 1 / 16
                    )

    newIm = Image.new(mode, size)
    newIm.putdata(pix)

    return newIm


def intarsia(
    image: np.ndarray,
    height: int,
    width: int,
    num_colours: int,
) -> Image:
    """Creates an "intarsia diagram", from a given image.
    For more information on intarsia see:
    `https://en.wikipedia.org/wiki/Intarsia_(knitting)`
    The idea is that this function takes an image, resizes it
    down, restricts the number of colours, and then resizes it
    back to its original shape.

    Args:
        - image (np.ndarray): The image on which the
            transformation will be applied.
        - height (int): The height of the intarsia diagram.
        - width (int): The width of the intarsia diagram.
        - num_colours (int): The number of colours that the
            intarsia diagram will have.

    Returns:
        - Image: The image after the transformation.
    """
    new_image = Image.fromarray(image)
    ysize = image.shape[0]
    xsize = image.shape[1]
    new_image = new_image.resize((width, height), Image.BILINEAR)
    new_image = new_image.quantize(colors=num_colours)
    new_image = new_image.convert("RGB")
    new_image = new_image.resize((xsize, ysize), Image.NEAREST)
    return new_image
