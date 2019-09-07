import tensorflow as tf
import argparse
from .unet_model import dice_loss, bce_dice_loss
from tensorflow.keras import models

def save(save_model_path, export_path):
    # The export path contains the name and the version of the model
    tf.keras.backend.set_learning_phase(0) # Ignore dropout at inference
    model = models.load_model(save_model_path, custom_objects={'bce_dice_loss': bce_dice_loss,
                                                               'dice_loss': dice_loss})


    # Fetch the Keras session and save the model
    # The signature definition is defined by the input and output tensors
    # And stored with the default serving key
    with tf.keras.backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            export_path,
            inputs={'input_image': model.input},
            outputs={t.name:t for t in model.outputs})


def parse_args():
    parser = argparse.ArgumentParser(
        description='save model for serving')

    parser.add_argument('model', type=str, help="Input Model file")
    parser.add_argument('export_path', type=str, help="Output path")
    return parser.parse_args()


def main():
    args = parse_args()
    save(args.model, args.export_path)

if __name__=='__main__':
    main()
