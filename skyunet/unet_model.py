import tensorflow as tf
from tensorflow.python.keras import layers, models, losses



def conv_block(input_tensor, num_filters):
    encoder = layers.Conv2D(num_filters, (3, 3),
                            padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    return encoder

def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)

    return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv2DTranspose(
        num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    return decoder

def get_model256():
    inputs = layers.Input(shape=(256, 256, 3))
    # 256

    encoder0_pool, encoder0 = encoder_block(inputs, 32)
    # 128

    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
    # 64

    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
    # 32

    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
    # 16

    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)
    # 8

    center = conv_block(encoder4_pool, 1024)
    # center

    decoder4 = decoder_block(center, encoder4, 512)
    # 16

    decoder3 = decoder_block(decoder4, encoder3, 256)
    # 32

    decoder2 = decoder_block(decoder3, encoder2, 128)
    # 64

    decoder1 = decoder_block(decoder2, encoder1, 64)
    # 128

    decoder0 = decoder_block(decoder1, encoder0, 32)
    # 256

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


def get_model512():
    inputs = layers.Input(shape=(512, 512, 3))
    # 512

    encoder0_pool, encoder0 = encoder_block(inputs, 32)
    # 256

    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
    # 128

    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
    # 64

    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
    # 32

    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)
    # 16

    encoder5_pool, encoder5 = encoder_block(encoder4_pool, 1024)
    # 8

    center = conv_block(encoder5_pool, 2048)
    # center

    decoder5 = decoder_block(center, encoder5, 1024)
    # 16

    decoder4 = decoder_block(decoder5, encoder4, 512)
    # 32

    decoder3 = decoder_block(decoder4, encoder3, 256)
    # 64

    decoder2 = decoder_block(decoder3, encoder2, 128)
    # 128

    decoder1 = decoder_block(decoder2, encoder1, 64)
    # 256

    decoder0 = decoder_block(decoder1, encoder0, 32)
    # 512

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def get_model128():
    inputs = layers.Input(shape=(128, 128, 3))
    # 128

    encoder0_pool, encoder0 = encoder_block(inputs, 32)
    # 64

    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
    # 32

    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
    # 16

    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
    # 8

    center = conv_block(encoder3_pool, 1024)
    # center

    decoder3 = decoder_block(center, encoder3, 256)
    # 16

    decoder2 = decoder_block(decoder3, encoder2, 128)
    # 32

    decoder1 = decoder_block(decoder2, encoder1, 64)
    # 64

    decoder0 = decoder_block(decoder1, encoder0, 32)
    # 128

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model
# Defining custom metrics and loss functions

def get_model(size):
    return {
        128: get_model128,
        256: get_model256,
        512: get_model512
    }.get(size)()

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / \
        (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(
        y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss
