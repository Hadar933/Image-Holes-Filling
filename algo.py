import numpy as np
from typing import Callable, Optional, Tuple
import cv2 as cv
from matplotlib import pyplot as plt


def generate_image_with_hole(image_filename: str = 'img.jpg', mask_filename: str = 'mask.jpg'):
    """
    takes an image and a mask and merges then such that the mask is assigned with pixel value -1
    :param image_filename: path to the image file
    :param mask_filename: path to the mask file
    :return: a merged image
    """
    im = cv.imread(image_filename, cv.IMREAD_GRAYSCALE)
    height, width = im.shape
    mask = cv.imread(mask_filename, cv.IMREAD_GRAYSCALE)
    mask[mask > 127] = 255
    mask[mask <= 127] = 0

    # TODO : remove this after testing
    width, height = im.shape[0] // 5, im.shape[0] // 5
    im = cv.resize(im, (width, height))

    mask = cv.resize(mask, (width, height))
    merged = cv.bitwise_and(im, im, mask=mask)
    mask_index = tuple(np.argwhere(mask == 0).T)
    merged = merged.astype('int64')  # so we'd be able to set value to -1 (initially was uint8)
    merged[mask_index] = -1
    plt.imshow(merged, cmap='gray'), plt.show()
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
    points = [[x, y] for x, y in points if x != -1 and y != -1 and x != im.shape[0] and y != im.shape[1]]
    return np.array(points)


def algorithm(I: np.ndarray, connectivity: int,
              w: [Callable[[np.ndarray, np.ndarray], float]] = lambda u, v, z, eps: 1 / (
                      eps + np.linalg.norm(u - v) ** z)):
    """
    find the hole H and its boundary B in image I.
    :param connectivity: either 4 or 8
    :param I: the input grayscale image
    :param w: a distance function between two pixels
    :return:
    """
    H = np.argwhere(I == -1)
    B = []  # vectorize? TODO
    for pixel in H:
        N = get_neighbours(I, pixel, connectivity)
        edges = [nei for nei in N if I[nei[0], nei[1]] >= 0]
        B += edges
    return np.unique(B, axis=0), H


# %%
I = generate_image_with_hole('monkey.jpg')
# %%
B, H = algorithm(I, 8)
# %%
plt.imshow(I, cmap='gray')
B_tup = tuple(np.array(B).T)
H_tup = tuple(H.T)
plt.scatter(x=H_tup[1], y=H_tup[0], c='blue')
plt.scatter(x=B_tup[1], y=B_tup[0], c="red", s=1)

plt.show()
