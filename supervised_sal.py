import time

import numpy as np
import os
import tempfile
import random
import matplotlib.pyplot as plt
from dataset import Sal360
from custom_env.gym_my_env.envs.viewport import Viewport
from resnet_tf import ResNet50
import tensorflow as tf
import cv2
from keras import Sequential
from keras.layers import TimeDistributed, Convolution2D, Flatten, LSTM, Dense
from keras.optimizers import Adam, rmsprop, SGD
from keras.losses import KLD


# 1. supervided learning cnn + rnn
# 2. RL --> object detection --> tracking (saliency) --> reward을
# 3. IRL --> reward function
# action 정의 매프레임? 묶어서?
# multimodal
# 음향 어떻게? 3D JS?
class Supervised:
    def __init__(self, data_format):
        self.model = ResNet50(data_format, include_top=False)
        self.path = 'model/supervised'

    def learn(self, epochs):
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


def doom_drqn(input_shape):
    learning_rate = 0.0001
    action_size = 2
    model = Sequential()
    model.add(TimeDistributed(Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu'), input_shape=(input_shape)))
    model.add(TimeDistributed(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu')))
    model.add(TimeDistributed(Convolution2D(64, 3, 3, activation='relu')))
    model.add(TimeDistributed(Flatten()))

    # Use all traces for training
    # model.add(LSTM(512, return_sequences=True,  activation='tanh'))
    # model.add(TimeDistributed(Dense(output_dim=action_size, activation='linear')))

    # Use last trace for training
    model.add(LSTM(512, activation='tanh'))
    model.add(Dense(output_dim=action_size, activation='linear'))

    adam = Adam(lr=learning_rate)
    model.compile(loss=KLD, optimizer=adam)

    return model


def resnet_rcnn(input_shape):
    pass


if __name__ == '__main__':
    # sign_ary = [[0., 0.], [0., 1.], [1., 0.], [1., 1.], [0., -1.], [-1., 0.], [-1., -1.], [-1., 1.], [1., -1.]]

    data, action = Sal360().load_sal360_dataset()
    data = data.reshape((19, 57, 100, 7))
    action = action.reshape((19, 57, 100, 2))
    _x_train, x_test = data[:15], data[15:]
    _y_train, y_test = action[:15], action[15:]

    x_train, x_validation = np.array([i[:45] for i in _x_train]), np.array(
        [i[45:] for i in _x_train])  # 15 45 100 7, 15 12 100 7
    y_train, y_validation = np.array([i[:45] for i in _y_train]), np.array([i[45:] for i in _y_train])

    width = 3840
    height = 1920
    data_format = 'channels_last'

    view = Viewport(width, height)

    v_width = int(view.width)
    v_height = int(view.height)

    input_shape = (-1, v_width, v_height, 3)
    fixed_input_shape = (1, 224, 224, 3)

    model = doom_drqn(fixed_input_shape)

    t_start = time.time()
    print("start task!")

    for video, _x, _y in zip(sorted(os.listdir(os.path.join('sample_videos', 'train', '3840x1920'))),
                             x_train, y_train):  # tr: 45, 100, 7
        cap = cv2.VideoCapture(os.path.join('sample_videos/train/3840x1920/', video))
        i_start = time.time()
        print("{} start task!".format(video))
        for x, y in zip(_x, _y):  # _x : 100, 7 _y : 100, 2
            x_iter = iter(x)
            y_iter = iter(y)
            x_data = next(x_iter)
            y_data = next(y_iter)
            frame_idx = x_data[6] - x_data[5] + 1
            prev_frame = None
            index = 0
            while True:
                ret, frame = cap.read()
                inputs = None
                if ret:
                    index += 1
                    print(frame_idx)
                    if index == frame_idx:
                        w, h = x_data[1] * width, x_data[2] * height
                        view.set_center(np.array([w, h]))
                        frame = view.get_view(frame)
                        frame = cv2.resize(frame, (224, 224))
                        x_data = next(x_iter)
                        y_data = next(y_iter)
                        frame_idx = x_data[6] - x_data[5] + 1
                        index = 0
                        print(np.shape(y_data))
                        model.train_on_batch([frame], y_data)
                        prev_frame = frame
                else:
                    print()
                    model.train_on_batch([prev_frame], y_data)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        i_end = time.time()
        print("video task take {}s".format(i_end - i_start))
        # save model
        model_json = model.to_json()
        with open(os.path.join("model/supervised/", video, "_model.json"), "w") as json_file:
            json_file.write(model_json)
        # save weight
        model.save_weights(os.path.join("model/supervised/", video, "_model.h5"))
        print("Saved model to disk")
    t_end = time.time()
    print("total task take {}s".format(t_end - t_start))

