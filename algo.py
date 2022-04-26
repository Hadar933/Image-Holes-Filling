import cv2
import numpy as np
from typing import Callable
import cv2 as cv
import skimage.color as ski
from matplotlib import pyplot as plt


def generate_image_with_hole(image_filename: str = 'jerusalem.jpg', mask_filename: str = 'mask.jpg'):
    im = cv.imread(image_filename, cv.IMREAD_GRAYSCALE)
    height, width = im.shape
    mask = cv.imread(mask_filename, cv.IMREAD_GRAYSCALE)
    mask[mask > 127] = 255
    mask[mask <= 127] = 0
    mask = cv.resize(mask, (width, height))
    merged = cv2.bitwise_and(im, im, mask=mask)
    plt.imshow(im, cmap='gray')
    plt.show()
    plt.imshow(mask, cmap='gray')
    plt.show()
    plt.imshow(merged, cmap='gray')
    plt.show()
    return im, mask, merged


def algorithm(I: np.ndarray, w: Callable[[np.ndarray, np.ndarray], float]):
    """
    find the hole H and its boundary B in image I.
    :param I: the input grayscale image
    :param w: a distance function between two pixels
    :return:
    """
    H = np.where(I == -1)


if __name__ == '__main__':
    im, mask, merged = generate_image_with_hole()
