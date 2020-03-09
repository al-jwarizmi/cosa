import random
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

def k_representative_pallette(img, num_clusters = random.randint(4,5)):
    """ 
    Cluster pixels into num_clusters representative colors
    and transform image into to this discrete palette
    """
    ysize=img.shape[0]
    xsize=img.shape[1]
    new_img = np.reshape(img, (xsize * ysize, 3))
    model = KMeans(n_clusters = num_clusters, random_state = 42).fit(new_img)
    colors = []
    for i in range(num_clusters):
        colors.append(np.median(new_img[model.labels_ == i], axis = 0))
        colors[i] = colors[i] / 255
    new_img = np.zeros((xsize * ysize, 3))
    for i in range(num_clusters):
        slicer = model.labels_ == i
        new_img[slicer] = colors[i]
    new_img = np.reshape(new_img, (ysize, xsize, 3))

    return new_img

def elastic_transform(img, alpha = random.randint(900,1000), sigma = random.randint(4,8), random_state=None):
    """
    Elastic deformation of imgs as described in [Simard2003],
    [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = img.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    print(x.shape)
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_img = map_coordinates(img, indices, order=1, mode='reflect')
    dist_reshaped = distored_img.reshape(img.shape)

    return dist_reshaped

    