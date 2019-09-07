import argparse
import numpy as np
import re
from collections import defaultdict
import os
from PIL import Image


def combine(in_path, output):
    "only for files named like 0_0.jpg, 0_1.jpg, 0_2.jpg, 1_0.jpg, 2_0.jpg, 3_0.jpg"
    rows = defaultdict(list)
    regex = re.compile('(\d+)_(\d+).jpg')
    for f in os.listdir(in_path):
        mc = regex.match(f)
        if not mc:
            continue
        row_num, col_num = mc.groups()
        row_num, col_num = int(row_num), int(col_num)
        rows[row_num].append(col_num)

    files = [None] * len(rows)
    for row_num, cols in rows.items():
        fs = [os.path.join(in_path, '{}_{}.jpg'.format(row_num, col_num))
              for col_num in sorted(cols)]
        files[row_num] = fs

    img = np.concatenate([np.concatenate([np.array(Image.open(f))
                                          for f in files_row], axis=1) for files_row in files], axis=0)
    img = Image.fromarray(img)
    img.save(output)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Combine splitted predict outputs')
    parser.add_argument('in_path', type=str, help='Input tile path')
    parser.add_argument('output', type=str, help='output image file')
    return parser.parse_args()


def main():
    args = parse_args()
    combine(args.in_path, args.output)


if __name__ == '__main__':
    main()
