import numpy as np
from typing import Callable
import cv2 as cv
from scipy.spatial.distance import cdist


def generate_image_with_hole(image_filename: str = 'img.jpg',
                             mask_filename: str = 'mask.jpg'):
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


def get_neighbours(height: int,
                   width: int,
                   pixel: np.ndarray,
                   connectivity: int):
    """
    an O(1) function that takes a pixel and returns its neighbours indexes (with edge cases)
    :param width: input width
    :param height: input height
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
    points = [[x, y] for x, y in points if x != -1 and y != -1 and x != width + 1 and y != height + 1]
    return np.array(points)


def find_hole_and_boundary(I: np.ndarray,
                           connectivity: int):
    """
    find the hole H and its boundary B in image I.
    :param connectivity: either 4 or 8
    :param I: the input grayscale image
    :return: Border (B: np.ndarray) ,Hole (H:np.ndarray)
    """
    height, width = I.shape[0], I.shape[1]
    H = np.argwhere(I == -1)
    B = []
    for pixel in H:
        N = get_neighbours(height, width, pixel, connectivity)
        edges = [nei for nei in N if I[nei[0], nei[1]] >= 0]  # all points not in H
        B += edges  # appending to B with possible duplicates
    return np.unique(B, axis=0), H


def fill_hole(I: np.ndarray,
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
    # c-dist is a vectorized, faster, version of np.array([w(h, b) for h in H for b in B])
    w_matrix = cdist(H, B, metric=w)
    H_index = tuple(H.T)
    B_index = tuple(B.T)
    I[H_index] = (w_matrix.dot(I[B_index])) / (w_matrix.dot(np.ones(B.shape[0])))
    return I
    # a non vectorized verision which is about 3 times slower may look something like this:
    # for u in H:
    #     I[u[0], u[1]] = sum([w(u, v) * I[v[0], v[1]] for v in B]) / sum([w(u, v) for v in B])


def fill_hole2(I: np.ndarray,
               H: np.ndarray,
               B: np.ndarray,
               z: float,
               eps: float,
               l: int,
               w: [Callable[[np.ndarray, np.ndarray], float]]):
    if not w: w = lambda u, v: 1 / (eps + np.linalg.norm(u - v) ** z)
    H_index_splitted = np.array_split(H, l)
    for indexes in H_index_splitted:
        center = np.average(indexes, axis=0).astype(np.int)
        w_vector = cdist(indexes, B)
        B_index, H_index = tuple(B.T), tuple(indexes.T)
        I[H_index] = (w_vector.dot(I[B_index])) // w_vector.dot(np.ones(B.shape[0]))
        # same as :
        # I[H_index] = sum([w(center, v) * I[v[0], v[1]] for v in B]) / sum([w(center, v) for v in B])
    return I
