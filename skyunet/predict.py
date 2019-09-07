import tensorflow as tf
from .dataset import get_predict_files, get_predict_dataset
from .unet_model import dice_coeff, dice_loss, bce_dice_loss

import matplotlib.pyplot as plt
import argparse
from tensorflow.python.keras import models
from PIL import Image
import os
import numpy as np
import pdb

def predict(dataset, output_path, save_model_path='/tmp/weights.hdf5', batch_size=5, threshold=0.7):

    x_filenames, x_names = get_predict_files(dataset)

    num_val_examples = len(x_filenames)

    val_ds = get_predict_dataset(x_filenames, batch_size=batch_size)

    model = models.load_model(save_model_path, custom_objects={'bce_dice_loss': bce_dice_loss,
                                                               'dice_loss': dice_loss})

    data_aug_iter = val_ds.make_one_shot_iterator()
    next_element = data_aug_iter.get_next()

    # Running next element in our graph will produce a batch of images

    current_name_index = 0
    try:
        while next_element is not None:
            batch_of_imgs = tf.keras.backend.get_session().run(next_element)
            predicted_label = model.predict(batch_of_imgs)

            for i in range(len(batch_of_imgs)):
     
                img_save_path = os.path.join(output_path, x_names[current_name_index])
                output_arr = ((predicted_label[i][:,:,0] > threshold)*255).astype(np.uint8)
                output_arr = np.stack((output_arr,)*3, axis=-1)
                output_img = Image.fromarray(output_arr)
                output_img.save(img_save_path)
    
                current_name_index+=1

    except tf.errors.OutOfRangeError:
        pass



def parse_args():
    parser = argparse.ArgumentParser(
        description='predict tiles in directory')
    parser.add_argument('dataset', type=str, help='Input tile path')
    parser.add_argument('output_path', type=str, help='Output predict tile path')
    parser.add_argument('--model', '-m', type=str, default='/tmp/trained_weights.hdf5', help="Output Model file, for example: /tmp/weights.hdf5")
    parser.add_argument('--batch_size', '-b', type=int, default=5, help="batch size")
    parser.add_argument('--threshold', '-t', type=float, default=0.7, help="threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    predict(args.dataset, args.output_path, save_model_path=args.model, batch_size=args.batch_size, threshold=args.threshold)

if __name__=='__main__':
    main()
