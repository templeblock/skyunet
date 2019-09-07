import cv2
import numpy as np
import argparse



def depadding(img, output, width,height):

    img = cv2.imread(img)
    full_width, full_height = img.shape[0:2]
    crop_img = img[0:full_width-width, 0:full_height-height]
    cv2.imwrite(output, crop_img)



def parse_args():
    parser = argparse.ArgumentParser(
        description='depadding image')
    parser.add_argument('image', type=str, help='Image to process')
    parser.add_argument('output', type=str, help='Output')
    parser.add_argument('width', type=int, help='width to cut')
    parser.add_argument('height', type=int, help='height to cut')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    depadding(args.image, args.output, args.width, args.height)


if __name__ == '__main__':
    main()
