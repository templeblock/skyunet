
import tensorflow.contrib as tfcontrib
import tensorflow as tf
from PIL import Image
import pandas as pd
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import os
import glob
import zipfile
import functools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12, 12)


def get_train_and_test(dataset_path):
    x_train_filenames = []
    y_train_filenames = []
    for f in os.listdir(dataset_path):
        if not f.endswith('.jpg') or f.endswith('_mask.jpg'):
            continue

        full_path = os.path.join(dataset_path, f)
        x_train_filenames.append(full_path)
        y_train_filenames.append(
            os.path.splitext(full_path)[0] + '_mask.jpg')

    return x_train_filenames, y_train_filenames


def _process_pathnames(fname, label_path):
    # We map this function onto each pathname pair
    img_str = tf.read_file(fname)
    img = tf.image.decode_jpeg(img_str, channels=3)
    label_img_str = tf.read_file(label_path)
    label_img = tf.image.decode_jpeg(label_img_str, channels=3)
    # The label image should only have values of 1 or 0, indicating pixel wise
    # object (car) or not (background). We take the first channel only.
    label_img = label_img[:, :, 0]
    label_img = tf.expand_dims(label_img, axis=-1)
    return img, label_img


def shift_img(output_img, label_img, width_shift_range, height_shift_range):
    """This fn will perform the horizontal or vertical shift"""
    if width_shift_range or height_shift_range:
        if width_shift_range:
            width_shift_range = tf.random_uniform([],
                                                  -width_shift_range *
                                                  255,
                                                  width_shift_range * 255)
        if height_shift_range:
            height_shift_range = tf.random_uniform([],
                                                   -height_shift_range *
                                                   255,
                                                   height_shift_range * 255)
        # Translate both
        output_img = tfcontrib.image.translate(output_img,
                                               [width_shift_range, height_shift_range])
        label_img = tfcontrib.image.translate(label_img,
                                              [width_shift_range, height_shift_range])
    return output_img, label_img


def flip_img(horizontal_flip, tr_img, label_img):
    if horizontal_flip:
        flip_prob = tf.random_uniform([], 0.0, 1.0)
        tr_img, label_img = tf.cond(tf.less(flip_prob, 0.5),
                                    lambda: (tf.image.flip_left_right(
                                        tr_img), tf.image.flip_left_right(label_img)),
                                    lambda: (tr_img, label_img))
    return tr_img, label_img


def augment(img,
            label_img,
            resize=None,  # Resize the image to some size e.g. [256, 256]
            scale=1,  # Scale image e.g. 1 / 255.
            hue_delta=0,  # Adjust the hue of an RGB image by random factor
            horizontal_flip=False,  # Random left right flip,
            width_shift_range=0,  # Randomly translate the image horizontally
            height_shift_range=0):  # Randomly translate the image vertically

    if hue_delta:
        img = tf.image.random_hue(img, hue_delta)

    img, label_img = flip_img(horizontal_flip, img, label_img)
    img, label_img = shift_img(
        img, label_img, width_shift_range, height_shift_range)
    label_img = tf.to_float(label_img) * scale
    img = tf.to_float(img) * scale
    return img, label_img

def scale_to_1(img, label_img, scale=1/255.):
    img = tf.to_float(img) * scale
    label_img = tf.to_float(label_img) * scale
    return img, label_img


def get_baseline_dataset(filenames,
                         labels,
                         preproc_fn=functools.partial(augment),
                         threads=5,
                         batch_size=3,
                         shuffle=True):
    num_x = len(filenames)
    # Create a dataset from the filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # Map our preprocessing function to every element in our dataset, taking
    # advantage of multithreading
    dataset = dataset.map(_process_pathnames, num_parallel_calls=threads)

    dataset = dataset.map(preproc_fn, num_parallel_calls=threads)

    if shuffle:
        dataset = dataset.shuffle(num_x)

    # It's necessary to repeat our data for all epochs
    dataset = dataset.repeat().batch(batch_size)
    return dataset


def get_validate_dataset(filenames, labels, batch_size=5):
    num_x = len(filenames)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_process_pathnames, num_parallel_calls=2)
    dataset = dataset.map(functools.partial(scale_to_1))
    dataset = dataset.batch(batch_size)
    return dataset



def get_predict_files(dataset_path):
    filenames = []
    names = []
    for f in os.listdir(dataset_path):
        if not f.endswith('.jpg'):
            continue
        full_path = os.path.join(dataset_path, f)
        filenames.append(full_path)
        names.append(f)

    return filenames, names


def _process_predict_files(fname):
    # We map this function onto each pathname pair
    img_str = tf.read_file(fname)
    img = tf.image.decode_jpeg(img_str, channels=3)
    return img


def scale_img_to_1(img, scale=1/255.):
    img = tf.to_float(img) * scale
    return img


def get_predict_dataset(filenames, batch_size=5):
    num_x = len(filenames)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(_process_predict_files, num_parallel_calls=2)
    dataset = dataset.map(functools.partial(scale_img_to_1))
    dataset = dataset.batch(batch_size)
    return dataset