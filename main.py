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


def check_alg(alg: str):
    if alg not in ['original', 'bonus1', 'bonus2']:
        raise argparse.ArgumentTypeError('The algorithm must either the original, or one of the bonuses.')
    return alg


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
    parser.add_argument('alg', metavar='algorithm', type=check_alg, nargs="?", const='original', default='original',
                        help=f'Either original, bonus1 or bonus2 (default=original).')

    args = parser.parse_args()
    im_pth, msk_pth = args.image_file_path, args.hole_file_path
    z, eps, conn = args.z, args.epsilon, args.connectivity
    alg = args.alg
    print("Merging image+mask...")
    im = generate_image_with_hole(im_pth, msk_pth)

    print("Finding hole+boundary...")
    boundary, hole = find_hole_and_boundary(im, conn)

    if alg == 'bonus1':
        l = 10  # can tune this
        print("Filling hole O(n) method...")
        I = fill_hole2(im, hole, boundary, z, eps, l, None)

    elif alg == 'bonus2':
        print("Filling hole O(nlogn) method...")
        # I = fill_hole3(im, hole, boundary, z, eps, None)

    elif alg == 'original':
        print("Filling hole regular method...")
        I = fill_hole(im, hole, boundary, z, eps, None)

    plt.imshow(I, cmap='gray')
    plt.title("orig")
    plt.show()

    print("Saving result as merged.jpg...")
    plt.imshow(I, cmap='gray')
    plt.savefig("merged.jpg")
