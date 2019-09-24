from keras.layers import Add
from keras.layers import Input, Activation
from keras.layers import MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import Conv2D, ZeroPadding1D
from keras.layers import BatchNormalization
from keras.layers import LSTM, LSTMCell, CuDNNLSTM
from keras.models import Model
from keras import backend as K
from keras.utils.data_utils import get_file
import keras

TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block):
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, kernel_size=1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, kernel_size=kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, kernel_size=1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, kernel_size=1, strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, kernel_size=kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, kernel_size=1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, kernel_size=1, strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def conv_block_dilation(input_tensor, kernel_size, filters, stage, block, dilation_rate=(2, 2)):
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, kernel_size=1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, kernel_size=kernel_size, padding='same',
               dilation_rate=dilation_rate, name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, kernel_size=1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, kernel_size=1, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def identity_block_dilation(input_tensor, kernel_size, filters, stage, block, dilation_rate=(2, 2)):
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, kernel_size=1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, kernel_size=kernel_size, dilation_rate=dilation_rate,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, kernel_size=1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def dcn_resnet(input_tensor=None):
    img_input = Input(shape=input_tensor)

    # if input_tensor is None:
    #     img_input = Input(shape=input_shape)
    # else:
    #     if not K.is_keras_tensor(input_tensor):
    #         img_input = Input(tensor=input_tensor)
    #     else:
    #         img_input = input_tensor

    bn_axis = 3
    # conv_1
    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, kernel_size=7, strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # conv_2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # conv_3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', strides=(2, 2))
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # conv_4
    x = conv_block_dilation(x, 3, [256, 256, 1024], stage=4, block='a', dilation_rate=(2, 2))
    x = identity_block_dilation(x, 3, [256, 256, 1024], stage=4, block='b', dilation_rate=(2, 2))
    x = identity_block_dilation(x, 3, [256, 256, 1024], stage=4, block='c', dilation_rate=(2, 2))
    x = identity_block_dilation(x, 3, [256, 256, 1024], stage=4, block='d', dilation_rate=(2, 2))
    x = identity_block_dilation(x, 3, [256, 256, 1024], stage=4, block='e', dilation_rate=(2, 2))
    x = identity_block_dilation(x, 3, [256, 256, 1024], stage=4, block='f', dilation_rate=(2, 2))

    # conv_5
    x = conv_block_dilation(x, 3, [512, 512, 2048], stage=5, block='a', dilation_rate=(4, 4))
    x = identity_block_dilation(x, 3, [512, 512, 2048], stage=5, block='b', dilation_rate=(4, 4))
    x = identity_block_dilation(x, 3, [512, 512, 2048], stage=5, block='c', dilation_rate=(4, 4))

    # Create model
    x = keras.layers.Flatten()(x)
    x = keras.layers.Reshape((1, 1, ))
    x = keras.layers.LSTM(512, activation='tanh')(x)
    x = keras.layers.Dense(2, activation='linear')(x)
    x = keras.layers.Dropout(0.5)(x)
    # x = keras.layers.Dense(10,
    #                        activation='softmax',
    #                        name='x_train_out')(x)

    model = Model(inputs=img_input, outputs=x)
    # Load weights
    # weights_path = get_file('resnet50_weights_th_dim_ordering_th_kernels_notop.h5', TH_WEIGHTS_PATH_NO_TOP,
    #                         cache_subdir='models', md5_hash='f64f049c92468c9affcd44b0976cdafe')
    # model.load_weights(weights_path)

    return model
