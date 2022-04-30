import argparse
from os.path import exists

img_default, mask_default, z_default, eps_default, connect_default = 'img.jpg', 'mask.jpg', 2, 0.01, 8


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


parser = argparse.ArgumentParser(description='Fills a hole in an image')

parser.add_argument('image file path', metavar='image_path', type=check_file_path,
                    help=f'Image file path.')

parser.add_argument('hole file path', metavar='hole_path', type=check_file_path,
                    help=f'Mask file path.')

parser.add_argument('z', metavar='z', type=float,
                    help=f'Power term for the distance metric.')

parser.add_argument('epsilon', metavar='eps', type=check_eps,
                    help=f'Small float value used to avoid division by zero.')

parser.add_argument('connectivity', metavar='connectivity', type=check_connect,
                    help=f'Either 4 or 8.')

args = parser.parse_known_args()
print(args)
