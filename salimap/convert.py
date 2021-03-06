from keras.models import Model
from keras.layers.core import Dropout, Activation
from keras.layers import Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
from keras.regularizers import l2
import keras.backend as K
import h5py
from salimap.eltwise_product import EltWiseProduct
from salimap.config import *
from keras.utils.conv_utils import convert_kernel


def tensor(img_rows=480, img_cols=640, downsampling_factor_net=8, downsampling_factor_product=10):
    input_ml_net = Input(shape=(img_rows, img_cols, 3))

    #########################################################
    # FEATURE EXTRACTION NETWORK							#
    #########################################################

    # weights = get_weights_vgg16(f, 1)

    conv1_1 = Convolution2D(64, kernel_size=3, strides=3, activation='relu', border_mode='same')(
        input_ml_net)
    # weights = get_weights_vgg16(f, 3)
    conv1_2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv1_1)
    conv1_pool = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(conv1_2)

    # weights = get_weights_vgg16(f, 6)
    conv2_1 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv1_pool)
    # weights = get_weights_vgg16(f, 8)
    conv2_2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv2_1)
    conv2_pool = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(conv2_2)

    # weights = get_weights_vgg16(f, 11)
    conv3_1 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv2_pool)
    # weights = get_weights_vgg16(f, 13)
    conv3_2 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv3_1)
    # weights = get_weights_vgg16(f, 15)
    conv3_3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv3_2)
    conv3_pool = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(conv3_3)

    # weights = get_weights_vgg16(f, 18)
    conv4_1 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv3_pool)
    # weights = get_weights_vgg16(f, 20)
    conv4_2 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv4_1)
    # weights = get_weights_vgg16(f, 22)
    conv4_3 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv4_2)
    conv4_pool = MaxPooling2D((2, 2), strides=(1, 1), border_mode='same')(conv4_3)

    # weights = get_weights_vgg16(f, 25)
    conv5_1 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv4_pool)
    # weights = get_weights_vgg16(f, 27)
    conv5_2 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5_1)
    # weights = get_weights_vgg16(f, 29)
    conv5_3 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5_2)

    #########################################################
    # ENCODING NETWORK										#
    #########################################################
    concatenated = merge([conv3_pool, conv4_pool, conv5_3], mode='concat', concat_axis=1)
    dropout = Dropout(0.5)(concatenated)

    int_conv = Convolution2D(64, 3, 3, init='glorot_normal', activation='relu', border_mode='same')(dropout)

    pre_final_conv = Convolution2D(1, 1, 1, init='glorot_normal', activation='relu')(int_conv)

    #########################################################
    # PRIOR LEARNING										#
    #########################################################
    rows_elt = math.ceil(img_rows / downsampling_factor_net) // downsampling_factor_product
    cols_elt = math.ceil(img_cols / downsampling_factor_net) // downsampling_factor_product
    eltprod = EltWiseProduct(init='zero', W_regularizer=l2(1 / (rows_elt * cols_elt)))(pre_final_conv)
    output_ml_net = Activation('relu')(eltprod)

    model = Model(input=[input_ml_net], output=[output_ml_net])

    # for layer in model.layers:
    #     print(layer.input_shape, layer.output_shape)

    return model


def teano(img_rows=480, img_cols=640, downsampling_factor_net=8, downsampling_factor_product=10):
    f = h5py.File("salimap/model/vgg16_weights.h5")

    input_ml_net = Input(shape=(3, img_rows, img_cols))

    #########################################################
    # FEATURE EXTRACTION NETWORK							#
    #########################################################

    # weights = get_weights_vgg16(f, 1)

    conv1_1 = Convolution2D(64, kernel_size=3, strides=3, activation='relu', border_mode='same')(
        input_ml_net)
    # weights = get_weights_vgg16(f, 3)
    conv1_2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv1_1)
    conv1_pool = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(conv1_2)

    # weights = get_weights_vgg16(f, 6)
    conv2_1 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv1_pool)
    # weights = get_weights_vgg16(f, 8)
    conv2_2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv2_1)
    conv2_pool = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(conv2_2)

    # weights = get_weights_vgg16(f, 11)
    conv3_1 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv2_pool)
    # weights = get_weights_vgg16(f, 13)
    conv3_2 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv3_1)
    # weights = get_weights_vgg16(f, 15)
    conv3_3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv3_2)
    conv3_pool = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(conv3_3)

    # weights = get_weights_vgg16(f, 18)
    conv4_1 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv3_pool)
    # weights = get_weights_vgg16(f, 20)
    conv4_2 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv4_1)
    # weights = get_weights_vgg16(f, 22)
    conv4_3 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv4_2)
    conv4_pool = MaxPooling2D((2, 2), strides=(1, 1), border_mode='same')(conv4_3)

    # weights = get_weights_vgg16(f, 25)
    conv5_1 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv4_pool)
    # weights = get_weights_vgg16(f, 27)
    conv5_2 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5_1)
    # weights = get_weights_vgg16(f, 29)
    conv5_3 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5_2)

    #########################################################
    # ENCODING NETWORK										#
    #########################################################
    concatenated = merge([conv3_pool, conv4_pool, conv5_3], mode='concat', concat_axis=1)
    dropout = Dropout(0.5)(concatenated)

    int_conv = Convolution2D(64, 3, 3, init='glorot_normal', activation='relu', border_mode='same')(dropout)

    pre_final_conv = Convolution2D(1, 1, 1, init='glorot_normal', activation='relu')(int_conv)

    #########################################################
    # PRIOR LEARNING										#
    #########################################################
    rows_elt = math.ceil(img_rows / downsampling_factor_net) // downsampling_factor_product
    cols_elt = math.ceil(img_cols / downsampling_factor_net) // downsampling_factor_product
    eltprod = EltWiseProduct(init='zero', W_regularizer=l2(1 / (rows_elt * cols_elt)))(pre_final_conv)
    output_ml_net = Activation('relu')(eltprod)

    model = Model(input=[input_ml_net], output=[output_ml_net])

    # for layer in model.layers:
    #     print(layer.input_shape, layer.output_shape)

    return model
