import cv2
import numpy as np
import argparse

BLACK = [0, 0, 0]

def padding(img, output, size=256):

    img = cv2.imread(img)
    width, height = img.shape[0:2]
    width = size - width % size if width % size else 0
    height = size - height % size if width % size else 0

    constant = cv2.copyMakeBorder(img, 0, height, 0, width, cv2.BORDER_CONSTANT, value=BLACK)
    cv2.imwrite(output, constant)
    return width, height


def parse_args():
    parser = argparse.ArgumentParser(
        description='Padding image to split')
    parser.add_argument('in_path', type=str, help='Path to process')
    parser.add_argument('output', type=str, 
                        help='Output')
    parser.add_argument('--size', '-s', type=int, default=256,
                        help='padding image to fit this size')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    width, height = padding(args.in_path, args.output, args.size)
    print(width, height)

if __name__ == '__main__':
    main()
