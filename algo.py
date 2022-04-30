import numpy as np
from typing import Callable
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist


def generate_image_with_hole(image_filename: str = 'img.jpg', mask_filename: str = 'mask.jpg'):
    """
    As suggested, takes an image and a mask and merges then such that the mask is assigned with pixel value -1
    :param image_filename: path to the image file
    :param mask_filename: path to the mask file
    :return: a merged image
    """
    im = cv.imread(image_filename, cv.IMREAD_GRAYSCALE)
    height, width = im.shape
    hole = cv.imread(mask_filename, cv.IMREAD_GRAYSCALE)
    hole[hole > 127] = 255
    hole[hole <= 127] = 0
    hole = cv.resize(hole, (width, height))
    merged = cv.bitwise_and(im, im, mask=hole)
    hole_index = tuple(np.argwhere(hole == 0).T)
    merged = merged.astype('int64')  # so we'd be able to set value to -1 (initially was uint8)
    merged[hole_index] = -1
    return merged


def get_neighbours(im: np.ndarray, pixel: np.ndarray, connectivity: int):
    """
    an O(1) function that takes a pixel and returns its neighbours indexes (with edge cases)
    :param connectivity: either 4 or 8
    :param pixel: [x,y]
    :return: the neighbors of [x,y] based on the connectivity
    """
    x, y = pixel[0], pixel[1]
    points = [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1],
              [x + 1, y + 1], [x + 1, y - 1], [x - 1, y + 1], [x - 1, y - 1]]
    if connectivity == 4:
        points = points[:4]
    # this line takes care of edges:
    points = [[x, y] for x, y in points if x != -1 and y != -1 and x != im.shape[0] + 1 and y != im.shape[1] + 1]
    return np.array(points)


def find_hole_and_boundary(I: np.ndarray, connectivity: int):
    """
    find the hole H and its boundary B in image I.
    :param connectivity: either 4 or 8
    :param I: the input grayscale image
    :return: Border (B) ,Hole (H)
    """
    H = np.argwhere(I == -1)
    B = []
    for pixel in H:
        N = get_neighbours(I, pixel, connectivity)
        edges = [nei for nei in N if I[nei[0], nei[1]] >= 0]
        B += edges
    return np.unique(B, axis=0), H


def fill_hole1(I: np.ndarray,
               H: np.ndarray,
               B: np.ndarray,
               z: float,
               eps: float,
               w: [Callable[[np.ndarray, np.ndarray], float]]):
    """
    this function implements the suggested hole filling algorithm
    :param eps: tuning parameter to avoid zero division
    :param z: tuning parameter used as power
    :param B: the border indexes
    :param H: the hole indexes
    :param I: given image
    :param w: weight function
    :return: image with filled hole
    """
    if not w: w = lambda u, v: 1 / (eps + np.linalg.norm(u - v) ** z)
    # cdist is a vectorized, slightly faster version of np.array([w(h, b) for h in H for b in B])
    w_matrix = cdist(H, B, metric=w)
    H_index = tuple(H.T)
    B_index = tuple(B.T)
    I[H_index] = (w_matrix.dot(I[B_index])) / (w_matrix.dot(np.ones(B.shape[0])))
    return I


def fill_hole2(I: np.ndarray, H: np.ndarray, B: np.ndarray, z, eps,
               w: [Callable[[np.ndarray, np.ndarray], float]] =
               lambda u, v, z, eps: 1 / (eps + np.linalg.norm(u - v) ** z)):
    """
    this function implements the suggested hole filling algorithm
    :param I: given image
    :param H: hole in the image
    :param B: boundary of H
    :param w: weight function
    :return: image with filled hole
    """
    default_w = lambda u, v: 1 / (eps + np.linalg.norm(u - v) ** z)
    # trivial solution:
    for u in H:
        I[u[0], u[1]] = sum([default_w(u, v) * I[v[0], v[1]] for v in B]) / sum([default_w(u, v) for v in B])
    return I


# %%
im = generate_image_with_hole('monkey.jpg')
# %%
boundary, hole = find_hole_and_boundary(im, 8)
# %%
# plt.imshow(im, cmap='gray')
# B_tup = tuple(np.array(boundary).T)
# H_tup = tuple(hole.T)
# plt.scatter(x=H_tup[1], y=H_tup[0], c='blue', s=1)
# plt.scatter(x=B_tup[1], y=B_tup[0], c="red", s=1)
# plt.show()
# %%
import time

start = time.time()
I1 = fill_hole1(im, hole, boundary, 2, 0.01, None)
end = time.time()
plt.imshow(I1, cmap='gray')
plt.title(f"alg 1 - vectorized, took {end - start}")
plt.show()
