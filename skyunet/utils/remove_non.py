import os, sys
import argparse


def remove(in_path, mask_suffix='_mask.jpg', size=1651):

    for f in os.listdir(in_path):
        if not f.endswith(mask_suffix):
            continue

        mask = os.path.join(in_path, f)
        statinfo = os.stat(mask)

        if statinfo.st_size <= size:
            name = f.replace(mask_suffix, '.jpg')
            regular = os.path.join(in_path, name)
            os.remove(regular)
            os.remove(mask)



def parse_args():
    parser = argparse.ArgumentParser(
        description='Combine splitted predict outputs')
    parser.add_argument('in_path', type=str, help='Input tile path')
    parser.add_argument('--mask_suffix', '-m', default='_mask.jpg', type=str, help='Mask suffix in filename')
    parser.add_argument('--size', '-s', default=1651, type=int, help='Remove data when mask size less than this value, in practice, blank mask file size is about 1651')
    return parser.parse_args()


def main():
    args = parse_args()
    remove(args.in_path, args.mask_suffix)


if __name__ == '__main__':
    main()