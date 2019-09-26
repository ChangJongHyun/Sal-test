import time

import numpy as np
import os
import tempfile
import random
import matplotlib.pyplot as plt
from my_resnet import dcn_resnet
from dataset import Sal360, data_generator
from custom_env.gym_my_env.envs.viewport import Viewport
from resnet_tf import ResNet50
import tensorflow as tf
import cv2
from keras import Sequential
from keras.layers import TimeDistributed, Convolution2D, Flatten, LSTM, Dense, Conv2D, ConvLSTM2D, ConvLSTM2DCell, \
    Embedding
from keras.optimizers import Adam, rmsprop, SGD
from keras.losses import KLD, binary_crossentropy
from keras.utils.training_utils import multi_gpu_model
import matplotlib.pyplot as plt
from keras.activations import softsign


# 1. supervided learning cnn + rnn
# 2. RL --> object detection --> tracking (saliency) --> reward을
# 3. IRL --> reward function
# action 정의 매프레임? 묶어서?
# multimodal
# 음향 어떻게? 3D JS?
class Supervised:
    def __init__(self, data_format, state_size, action_size, trace_length):
        self.state_size = state_size
        self.action_size = action_size

        # hyperparam
        self.batch_size = 32
        self.learning_rate = 0.0001
        self.trace_length = trace_length

        # model
        self.model = None
        # self.model = ResNet50(data_format, include_top=False)
        self.path = 'model/supervised'

    def train(self, epochs):
        pass

    def save_model(self, sess):
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_path=os.path.join(self.path, 'model.ckpt'))
        print('Model saved in path : {}'.format(self.path + 'model.ckpt'))

    def load_model(self, sess):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, os.path.join(self.path, 'model.ckpt'))
            print("Model restored")


def random_batch(batch_size, data_format):
    shape = (3, 224, 224) if data_format == 'channels_first' else (224, 224, 3)
    shape = (batch_size,) + shape

    num_classes = 1000
    images = tf.random_uniform(shape)
    labels = tf.random_uniform(
        [batch_size], minval=0, maxval=num_classes, dtype=tf.int32)
    one_hot = tf.one_hot(labels, num_classes)

    return images, one_hot


def simple_model(input_shape):
    learning_rate = 0.0001
    action_size = 2
    model = Sequential()

    # TimeDistributed 확인
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu', input_shape=input_shape))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    print(model.output_shape)
    model.add(Flatten())
    print(model.output_shape)
    # Use all traces for training
    # model.add(LSTM(512, return_sequences=True,  activation='tanh'))
    # model.add(TimeDistributed(Dense(output_dim=action_size, activation='linear')))

    # Use last trace for training
    model.add(LSTM(512, activation=softsign))
    model.add(Dense(output_dim=action_size, activation='linear'))

    adam = Adam(lr=learning_rate)
    model.compile(loss=KLD, optimizer=adam, metrics=['accuracy'])

    return model


def resnet(input_shape):
    model = dcn_resnet(input_shape)
    adam = Adam(lr=0.0001)
    model.compile(loss=binary_crossentropy, optimizer=adam)

    return model


def doom_drqn(input_shape):
    learning_rate = 0.0001
    action_size = 2
    model = Sequential()
    model.add(TimeDistributed(Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu')))
    model.add(TimeDistributed(Convolution2D(64, 3, 3, activation='relu')))
    model.add(TimeDistributed(Flatten()))

    # Use all traces for training
    model.add(LSTM(512, return_sequences=True, activation='tanh'))
    model.add(TimeDistributed(Dense(output_dim=action_size, activation='linear')))

    # Use last trace for training
    # model.add(LSTM(512, activation='tanh'))
    # model.add(Dense(output_dim=action_size, activation='linear'))

    adam = Adam(lr=learning_rate)
    model = multi_gpu_model(model, gpus=2)
    model.compile(loss=binary_crossentropy, optimizer=adam)

    print(model.summary())

    return model


def resnet_rcnn(input_shape):
    pass


if __name__ == '__main__':
    # sign_ary = [[0., 0.], [0., 1.], [1., 0.], [1., 1.], [0., -1.], [-1., 0.], [-1., -1.], [-1., 1.], [1., -1.]]
    total_loss = []
    data_format = 'channels_last'

    fixed_input_shape = (224, 224, 3)
    trace_length = None
    state_size = (224, 224, 3)

    max_epochs = 10
    epochs = 0
    x = dcn_resnet(state_size)
    print(x.summary())

    # model = doom_drqn(state_size)
    # print(model.summary())
