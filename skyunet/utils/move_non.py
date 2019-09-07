import os, sys
import argparse
import shutil


def move(in_path, output_path, size=1651):

    for f in os.listdir(in_path):
        if not f.endswith('.jpg'):
            continue

        jpg = os.path.join(in_path, f)
        statinfo = os.stat(jpg)

        if statinfo.st_size <= size:
            source = os.path.join(in_path, f)
            dest = os.path.join(output_path, f)
            shutil.move(source, dest)

