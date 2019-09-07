import cv2
import os
import sys
import argparse
import math


def split(in_file, output, size=256, mask='', prefix=''):
    im = cv2.imread(in_file)
    rows, cols = im.shape[0:2]
    rows, cols = math.floor(rows/size), math.floor(cols/size)

    for x in range(rows):
        for y in range(cols):
            tiles = im[x*size:(x+1)*size, y*size:(y+1)*size]
            new_name = '{}{}_{}{}.jpg'.format(prefix, x, y, mask)
            new_path = os.path.join(output, new_name)
            cv2.imwrite(new_path, tiles)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Split whole Image in to smaller tiles, becuase too large image may lead to OOM Error when training')
    parser.add_argument('in_image', type=str, help='Input Image file to split')
    parser.add_argument('output_path', type=str,
                        help='Output path to store splitted tiles')
    parser.add_argument('--size', '-s', type=int, default=256,
                        help='tile size, the size of the image must be evenly divisible by a factor of 32, default 256')
    parser.add_argument('--mask', '-m', default='', type=str,
                        help='specify this image as mask, output tiles name will end with the value(_mask recommended)')
    parser.add_argument('--prefix', '-p', type=str,
                        default='', help='output tile name prefix, useful when there are multi images')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    split(args.in_image, args.output_path, size=args.size,
          mask=args.mask, prefix=args.prefix)


if __name__ == '__main__':
    main()
