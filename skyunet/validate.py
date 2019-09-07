import tensorflow as tf
from .dataset import get_validate_dataset, get_train_and_test
from .unet_model import dice_coeff, dice_loss, bce_dice_loss

import matplotlib.pyplot as plt
import argparse
from tensorflow.python.keras import models


def validate(dataset, save_model_path='/tmp/weights.hdf5', batch_size=5):

    x_val_filenames, y_val_filenames = get_train_and_test(dataset)

    num_val_examples = len(x_val_filenames)

    val_ds = get_validate_dataset(x_val_filenames, y_val_filenames, batch_size=batch_size)

    model = models.load_model(save_model_path, custom_objects={'bce_dice_loss': bce_dice_loss,
                                                               'dice_loss': dice_loss})

    data_aug_iter = val_ds.make_one_shot_iterator()
    next_element = data_aug_iter.get_next()

    # Running next element in our graph will produce a batch of images

    try:
        while next_element:
            batch_of_imgs, label = tf.keras.backend.get_session().run(next_element)
            predicted_label = model.predict(batch_of_imgs)
            fig = plt.figure(figsize=(8, 8))

            for i in range(len(batch_of_imgs)):

                plt.subplot(batch_size, 3, 3*i+1)
                plt.imshow(batch_of_imgs[i])
                plt.title("Input Image")

                plt.subplot(batch_size, 3, 3*i+2)
                plt.imshow(label[i,:,:,0])
                plt.title("Actual Mask")

                plt.subplot(batch_size, 3, 3*i+3)
                plt.imshow(predicted_label[i][:,:,0])
                plt.title("Predicted Mask")

            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.show()


    except tf.errors.OutOfRangeError:
        pass



def parse_args():
    parser = argparse.ArgumentParser(
        description='validate')
    parser.add_argument('dataset', type=str, help='Input tile path')
    parser.add_argument('--model', '-m', type=str, default='/tmp/trained_weights.hdf5', help="Output Model file, for example: /tmp/weights.hdf5")
    parser.add_argument('--batch_size', '-b', type=int, default=5, help="batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    validate(args.dataset, save_model_path=args.model, batch_size=args.batch_size)

if __name__=='__main__':
    main()
