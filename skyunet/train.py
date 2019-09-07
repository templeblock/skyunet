import tensorflow as tf
from .dataset import get_train_and_test, get_baseline_dataset, augment
from .unet_model import get_model, dice_coeff, dice_loss, bce_dice_loss
from .segnet_model import segnet
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import functools
import numpy as np
from tensorflow.python.keras import layers, models, losses


def train(dataset, save_model_path='/tmp/weights.hdf5', epochs=5, batch_size=3):
    tr_cfg = {
        'scale': 1 / 255.,
        'hue_delta': 0.1,
        'horizontal_flip': True,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1
    }
    tr_preprocessing_fn = functools.partial(augment, **tr_cfg)
    val_cfg = {
        'scale': 1 / 255.,
    }
    val_preprocessing_fn = functools.partial(augment, **val_cfg)

    x_train_filenames, y_train_filenames = get_train_and_test(dataset)
    x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = \
        train_test_split(x_train_filenames, y_train_filenames,
                         test_size=0.2, random_state=42)
    num_train_examples = len(x_train_filenames)
    num_val_examples = len(x_val_filenames)
    train_ds = get_baseline_dataset(x_train_filenames,
                                    y_train_filenames,
                                    preproc_fn=tr_preprocessing_fn,
                                    batch_size=batch_size)

    val_ds = get_baseline_dataset(x_val_filenames,
                                  y_val_filenames,
                                  preproc_fn=val_preprocessing_fn,
                                  batch_size=batch_size)

    model = get_model(256)
    #model = segnet((255,255,3), 2)
    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_loss])
    model.summary()

    cp = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_model_path, monitor='val_dice_loss', save_best_only=True, verbose=1)

    history = model.fit(train_ds,
                        steps_per_epoch=int(
                            np.ceil(num_train_examples / float(batch_size))),
                        epochs=epochs,
                        validation_data=val_ds,
                        validation_steps=int(
                            np.ceil(num_val_examples / float(batch_size))),
                        callbacks=[cp])

    visualize_training_process(history, epochs)
    validate(val_ds, save_model_path)


def visualize_training_process(history, epochs):
    dice = history.history['dice_loss']
    val_dice = history.history['val_dice_loss']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, dice, label='Training Dice Loss')
    plt.plot(epochs_range, val_dice, label='Validation Dice Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Dice Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()


def validate(val_ds, save_model_path):
    model = models.load_model(save_model_path, custom_objects={'bce_dice_loss': bce_dice_loss,
                                                               'dice_loss': dice_loss})

    data_aug_iter = val_ds.make_one_shot_iterator()
    next_element = data_aug_iter.get_next()

    # Running next element in our graph will produce a batch of images
    plt.figure(figsize=(10, 20))
    for i in range(5):
        batch_of_imgs, label = tf.keras.backend.get_session().run(next_element)
        img = batch_of_imgs[0]
        predicted_label = model.predict(batch_of_imgs)[0]

        plt.subplot(5, 3, 3 * i + 1)
        plt.imshow(img)
        plt.title("Input image")

        plt.subplot(5, 3, 3 * i + 2)
        plt.imshow(label[0, :, :, 0])
        plt.title("Actual Mask")
        plt.subplot(5, 3, 3 * i + 3)
        plt.imshow(predicted_label[:, :, 0])
        plt.title("Predicted Mask")
        plt.suptitle("Examples of Input Image, Label, and Prediction")
    plt.show()



def parse_args():
    parser = argparse.ArgumentParser(
        description='train model')
    parser.add_argument('dataset', type=str, help='Input tile path')
    parser.add_argument('--output_model', '-o', type=str, default='/tmp/trained_weights.hdf5', help="Output Model file, for example: /tmp/weights.hdf5")
    parser.add_argument('--batch_size', '-s', type=int, default=3, help="batch size")
    parser.add_argument('--epochs', '-e', type=int, default=5, help='epochs')
    return parser.parse_args()


def main():
    args = parse_args()
    train(args.dataset, save_model_path=args.output_model, epochs=args.epochs, batch_size=args.batch_size)

if __name__=='__main__':
    main()



if __name__ == '__main__':
    main()
