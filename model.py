import keras
from keras.layers import TimeDistributed, Flatten, LSTM, Dropout, Dense, Input, Conv2D, ConvLSTM2D, BatchNormalization, \
    Conv3D, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from dataset import DataGenerator
from my_resnet import dcn_resnet
import random
import numpy as np


class Networks(object):

    def __init__(self, input_shape, action_size):
        self.input_shape = input_shape
        self.action_size = action_size

    @staticmethod
    def supervised_lstm(input_shape, action_size, learning_rate=0.01,
                        backbone='resnet', time_distributed=True, multi_gpu=True):
        img_input = Input(shape=input_shape, dtype='float32')
        if backbone == 'resnet':
            x = dcn_resnet(img_input, time_distributed)
        elif backbone == 'mobilenet':
            mobilenet = keras.applications.mobilenet_v2.MobileNetV2(include_top=False,
                                                                    weights=None,
                                                                    pooling='max')
            x = TimeDistributed(mobilenet)(img_input)
        elif backbone == 'convLSTM':
            x = ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=False)(img_input)
            x = BatchNormalization()(x)
            x = ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=False)(x)
            x = BatchNormalization()(x)
            x = ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=False)(x)
            x = BatchNormalization()(x)
            x = ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=False)(x)
            x = BatchNormalization()(x)
        else:
            x = TimeDistributed(Conv2D(32, kernel_size=8, strides=4, activation='relu'))(img_input)
            x = TimeDistributed(Conv2D(64, kernel_size=4, strides=2, activation='relu'))(x)
            x = TimeDistributed(Conv2D(64, kernel_size=3, strides=1, activation='relu'))(x)
        x = TimeDistributed(Flatten())(x)
        x = LSTM(512)(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(action_size, activation='sigmoid', name='x_train_out')(x)
        optimizer = Adam(lr=learning_rate)
        model = Model(inputs=img_input, outputs=x)

        model.compile(optimizer=optimizer, loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
        model.summary()
        return model

    @staticmethod
    def drqn(input_shape, action_size, learning_rate=0.001,
             backbone='mobilenet'):
        img_input = Input(shape=input_shape)
        adam = Adam(lr=learning_rate)

        if backbone == 'resnet':
            x = dcn_resnet(img_input, time_distributed=True)
            x = TimeDistributed(Flatten())(x)
            x = LSTM(512, activation='tanh')(x)
            x = Dropout(0.5)(x)
            x = Dense(action_size, activation='linear')(x)
            model = Model(inputs=img_input, outputs=x)
            model.compile(loss='mse', optimizer=adam)
        elif backbone == 'mobilenet':
            mobilenet = keras.applications.mobilenet_v2.MobileNetV2(include_top=False,
                                                                    weights=None,
                                                                    pooling='max')
            x = TimeDistributed(mobilenet)(img_input)
            x = TimeDistributed(Flatten())(x)
            x = LSTM(512, activation='tanh')(x)
            x = Dropout(0.5)(x)
            x = Dense(action_size, activation='linear')(x)
            model = Model(inputs=img_input, outputs=x)
            model.compile(loss='mse', optimizer=adam)
        elif backbone == 'cnn':
            x = TimeDistributed(Conv2D(32, kernel_size=8, strides=4, activation='relu'))(img_input)
            x = TimeDistributed(Conv2D(64, kernel_size=4, strides=2, activation='relu'))(x)
            x = TimeDistributed(Conv2D(64, kernel_size=3, strides=1, activation='relu'))(x)
            x = TimeDistributed(Flatten())(x)
            x = LSTM(256, activation='tanh')(x)
            x = Dropout(0.5)(x)
            x = Dense(1, activation='sigmoid')(x)
            model = Model(inputs=img_input, outputs=x)
            model.compile(loss='mse', optimizer=adam)
        elif backbone == 'convLSTM':
            print(np.shape(img_input))
            # ?, ?, img_w, img_h, channels
            x = convLSTM(64, 3)(img_input)
            x = BatchNormalization()(x)

            x = convLSTM(64, 3)(x)
            x = BatchNormalization()(x)

            x = convLSTM(64, 3)(x)
            x = BatchNormalization()(x)

            x = convLSTM(64, 3, return_sequences=False)(x)
            x = BatchNormalization()(x)

            # x = Conv3D(filters=1, kernel_size=(3, 3, 3),
            #            activation=LeakyReLU(alpha=0.2),
            #            padding='same', data_format='channels_last')(x)
            x = Flatten()(x)
            x = Dropout(rate=0.5)(x)
            x = Dense(action_size, activation='linear')(x)
            model = Model(inputs=img_input, outputs=x)
            model.compile(loss='mse', optimizer='adam')
        elif backbone == '2.5D':
            base_cnn_model = keras.applications.mobilenet_v2.MobileNetV2(include_top=False,
                                                                         weights=None,
                                                                         pooling='max')
            temporal = TimeDistributed(base_cnn_model)(img_input)
            conv3d_analysis1 = Conv3D(filters=40, kernel_size=3, strides=3, padding='same')(temporal)
            conv3d_analysis2 = Conv3D(filters=40, kernel_size=3, strides=3, padding='same')(conv3d_analysis1)
            output = Flatten()(conv3d_analysis2)
            output = Dense(action_size, activation='linear')(output)
            model = Model(inputs=img_input, output=output)
            model.compile(loss='mose', optimizer='adam')

        else:
            raise ValueError("invalid value")
        model.summary()
        return model

    @staticmethod
    def a2c_lstm(input_shape, action_size, value_size, learning_rate, share_network=True, backbone='resnet'):
        img_input = Input(shape=input_shape)
        adam = Adam(lr=learning_rate)

        if share_network:
            if backbone == 'resnet':
                resnet = keras.applications.resnet50.ResNet50(include_top=False,
                                                              weights=None,
                                                              pooling='max')
                x = TimeDistributed(resnet)(img_input)
            elif backbone == 'mobilenet':
                mobilenet = keras.applications.mobilenet_v2.MobileNetV2(include_top=False,
                                                                        weights=None,
                                                                        pooling='max')
                x = TimeDistributed(mobilenet)(img_input)
            elif backbone == 'cnn':
                x = TimeDistributed(Conv2D(32, kernel_size=8, strides=4, activation='relu'))(img_input)
                x = TimeDistributed(Conv2D(64, kernel_size=4, strides=2, activation='relu'))(x)
                x = TimeDistributed(Conv2D(64, kernel_size=3, strides=1, activation='relu'))(x)
            else:
                raise ValueError("wrong backbone network")
            x = TimeDistributed(Flatten())(x)
            x = LSTM(512, activation='tanh')(x)

            # actor
            actor = Dense(action_size, activation='softmax')(x)

            # critic
            critic = Dense(value_size, activation='linear')(x)

            model = Model(inputs=img_input, outputs=[actor, critic])
            adam = Adam(lr=learning_rate, clipnorm=1.0)
            model.compile(loss=['categorical_crossentropy', 'mse'], optimizer=adam, loss_weights=[1., 1.])
            return model

    @staticmethod
    def value_distribution_network(input_shape, num_atoms, action_size, learning_rate):
        pass

    @staticmethod
    def actor_network(input_shape, action_size, learning_rate):
        pass


def convLSTM(filters, kernal_size, padding='same', return_sequences=True,
             initializer=None, activation=None):
    if initializer is None:
        initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
    if activation is None:
        activation = LeakyReLU(alpha=0.2)
    return ConvLSTM2D(filters=filters,
                      kernel_size=(kernal_size, kernal_size),
                      padding=padding, activation=activation,
                      return_sequences=return_sequences,
                      bias_initializer=initializer,
                      kernel_initializer=initializer,
                      recurrent_initializer=initializer)


class GAIL():
    def __init__(self, observation_space, action_space):
        self.policy_net = None
        self.discriminate_nat = None
        self.observation_space = observation_space
        self.action_space = action_space

    # PI(s, a) - actor
    def build_policy(self, state_input, action_input):
        policy_network = self.construct_network(state_input, action_input)
        
    # Q(s, a) - critic
    def build_value(self, state_input, action_input):
        value_network = self.construct_network(state_input, action_input)

    # D(s,a)
    def build_discriminator(self, state_input, action_input):
        agent = 'agent_'
        expert = 'expert_'

        x = convLSTM(64, 3)(state_input)
        x = BatchNormalization()(x)

        x = convLSTM(64, 3)(x)
        x = BatchNormalization()(x)

        x = convLSTM(64, 3)(x)
        x = BatchNormalization()(x)

        x = convLSTM(64, 3)(x)
        x = BatchNormalization()(x)

        x = Conv3D(filters=1, kernel_size=(3, 3, 3),
                   activation=LeakyReLU(alpha=0.2),
                   padding='same', data_format='channels_last')(x)
        x = Flatten()(x)

        y = Dense(64, activation='relu')(action_input)

        z = keras.layers.Add()([x, y])
        z = Dropout(rate=0.5)(z)
        z = Dense(1, activation='sigmoid')(z)

        model = Model(inputs=[state_input, action_input], outputs=z)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        expert_network = self.construct_network(state_input, action_input)
        agent_network = self.construct_network(state_input, action_input)

    def construct_network(self, state_input, action_input):
        y = Dense(64, activation='relu')(action_input)

        x = convLSTM(64, 3)(state_input)
        x = BatchNormalization()(x)
        x = convLSTM(64, 3)(x)
        x = BatchNormalization()(x)
        x = convLSTM(64, 3)(x)
        x = BatchNormalization()(x)
        x = convLSTM(64, 3)(x)
        x = BatchNormalization()(x)
        x = Conv3D(filters=1, kernel_size=(3, 3, 3),
                   activation=LeakyReLU(alpha=0.2),
                   padding='same', data_format='channels_last')(x)
        x = Flatten()(x)

        z = keras.layers.Add()([x, y])
        z = Dropout(rate=0.5)(z)
        z = Dense(1, activation='sigmoid')(z)

        model = Model(inputs=[state_input, action_input], outputs=z)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model


if __name__ == '__main__':
    img_w = 224
    img_h = 224
    # time, w, h, c
    state_size = (None, img_w, img_h, 3)
    action_size = 2

    train_gen = DataGenerator.generator_for_batch(img_w, img_h, type='train', normalize=False)
    # model = Networks.drqn(state_size, action_size, backbone='convLSTM')
    model = Networks.drqn(state_size, action_size, backbone='convLSTM')
    keras.utils.plot_model(model, to_file='model.png')
