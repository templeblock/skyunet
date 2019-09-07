from PIL import Image
import numpy as np
import sys
import os
import argparse


def reverse(img_path, suffix='_mask.jpg'):
    for f in os.listdir(img_path):
        if not f.endswith(suffix):
            continue
        p = os.path.join(img_path, f)
        img = Image.open(p)
        arr = np.array(img)
        arr = 255 - arr
        img = Image.fromarray(arr)
        img.save(p)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Reverse Image Pixel Value')
    parser.add_argument('in_path', type=str, help='Path to process')
    parser.add_argument('--suffix', '-s', type=str, default='_mask.jpg',
                        help='only image with the specified suffix will be proccessed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    reverse(args.in_path, args.suffix)


if __name__ == '__main__':
    main()
