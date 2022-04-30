import argparse
from os.path import exists
from algo import generate_image_with_hole, find_hole_and_boundary, fill_hole, fill_hole2
from matplotlib import pyplot as plt


def check_file_path(filepath: str):
    if not exists(filepath):
        raise argparse.ArgumentTypeError(f'File {filepath} does not exists.')
    return filepath


def check_eps(eps):
    eps = float(eps)
    if eps <= 0:
        raise argparse.ArgumentTypeError(f'Epsilon must be > 0.')
    return eps


def check_connect(c):
    c = int(c)
    if c not in [4, 8]:
        raise argparse.ArgumentTypeError('The connectivity must be 4 or 8.')
    return c


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Given an original image and a hole image, merges the two and fills the hole.'
                    ' The result is saved as merged.jpg.'
                    ' Use flag -h for more information, or simply run py main.py for default behaviour.'
    )
    parser.add_argument('image_file_path', metavar='image_path', type=check_file_path, nargs="?", const='img.jpg',
                        default='img.jpg',
                        help=f'Image file path (default=img.jpg).')
    parser.add_argument('hole_file_path', metavar='hole_path', type=check_file_path, nargs="?", const='mask.jpg',
                        default='mask.jpg',
                        help=f'Mask file path (same size as image, otherwise reshaped, default=mask.jpg).')
    parser.add_argument('z', metavar='z', type=float, nargs="?", const=2, default=2,
                        help=f'Power term for the distance metric (default=2).')
    parser.add_argument('epsilon', metavar='eps', type=check_eps, nargs="?", const=0.01, default=0.01,
                        help=f'Small float value used to avoid division by zero (default=0.01).')
    parser.add_argument('connectivity', metavar='connectivity', type=check_connect, nargs="?", const=4, default=4,
                        help=f'Either 4 or 8 (default=4).')

    args = parser.parse_args()
    im_pth, msk_pth, z, eps, conn = args.image_file_path, args.hole_file_path, args.z, args.epsilon, args.connectivity

    print("Merging image+mask...")
    im = generate_image_with_hole(im_pth, msk_pth)

    print("Finding hole+boundary...")
    boundary, hole = find_hole_and_boundary(im, conn)

    # plt.imshow(im, cmap='gray')
    # B_tup = tuple(boundary.T)
    # H_tup = tuple(hole.T)
    # plt.scatter(x=H_tup[1], y=H_tup[0], c='blue', s=1)
    # plt.scatter(x=B_tup[1], y=B_tup[0], c="red", s=1)
    # plt.show()

    print("Filling hole O(n) method...")
    for l in [1,5,10,100]:
        I2 = fill_hole2(im, hole, boundary, z, eps, l, None)
        plt.imshow(I2, cmap='gray')
        plt.title(f"O(n) method, l={l}")
        plt.show()

    print("Filling hole regular method...")
    I1 = fill_hole(im, hole, boundary, z, eps, None)
    plt.imshow(I1, cmap='gray')
    plt.title("orig")
    plt.show()

    # print("Saving result as merged.jpg...")
    # plt.imshow(I1, cmap='gray')
    # plt.savefig("merged.jpg")
