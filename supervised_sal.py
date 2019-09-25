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
    fps_list = {'01_PortoRiverside.mp4': 25, '02_Diner.mp4': 30, '03_PlanEnergyBioLab.mp4': 25,
           '04_Ocean.mp4': 30, '05_Waterpark.mp4': 30, '06_DroneFlight.mp4': 25, '07_GazaFishermen.mp4': 25,
           '08_Sofa.mp4': 24, '09_MattSwift.mp4': 30, '10_Cows.mp4': 24, '11_Abbottsford.mp4': 30,
           '12_TeatroRegioTorino.mp4': 30, '13_Fountain.mp4': 30, '14_Warship.mp4': 25, '15_Cockpit.mp4': 25,
           '16_Turtle.mp4': 30, '17_UnderwaterPark.mp4': 30, '18_Bar.mp4': 25, '19_Touvet.mp4': 30}
    train, validation, test = Sal360().load_sal360_dataset()
    width = 3840
    height = 1920
    data_format = 'channels_last'

    view = Viewport(width, height)

    v_width = int(view.width)
    v_height = int(view.height)

    input_shape = (-1, v_width, v_height, 3)
    fixed_input_shape = (224, 224, 3)
    trace_length = None
    state_size = (trace_length, 224, 224, 3)

    max_epochs = 10
    epochs = 0

    model = doom_drqn(state_size)
    print(model.summary())
    t_start = time.time()
    print("start task!")
    while epochs < max_epochs:
        epochs_per_loss = []
        for video, _x, _y in zip(sorted(os.listdir(os.path.join('sample_videos', 'train', '3840x1920'))),
                                 train[0], train[1]):  # tr: 45, 100, 7
            i_start = time.time()
            video_per_loss = []
            trace_length = fps_list[video]
            print("{} start task!".format(video))
            for x, y in zip(_x, _y):  # _x : 100, 7 _y : 100, 2
                cap = cv2.VideoCapture(os.path.join('sample_videos/train/3840x1920/', video))
                x_iter = iter(x)
                y_iter = iter(y)
                x_data = next(x_iter)
                y_data = next(y_iter)
                frame_idx = x_data[6] - x_data[5] + 1
                prev_frame = None
                index = 0

                sequenceX = []
                sequenceY = []

                batchX = []
                batchY = []

                while True:
                    ret, frame = cap.read()
                    inputs = None
                    if ret:
                        index += 1
                        if index == frame_idx:
                            w, h = x_data[1] * width, x_data[2] * height
                            view.set_center(np.array([w, h]))
                            frame = view.get_view(frame)
                            frame = cv2.resize(frame, (224, 224))

                            x_data = next(x_iter)
                            y_data = next(y_iter)
                            y_data = [y_data[0] * width, y_data[1] * height]
                            sequenceX.append(frame)
                            sequenceY.append(y_data)

                            frame_idx = x_data[6] - x_data[5] + 1
                            index = 0
                            prev_frame = frame
                            if len(sequenceX) == trace_length:
                                batchX.append(sequenceX)
                                batchY.append(sequenceY)
                                print(np.shape(sequenceX))
                                print(np.shape(sequenceY))
                                sequenceX = []
                                sequenceY = []

                    else:
                        if len(sequenceX) > 0 and len(sequenceX) != trace_length:
                            for i in range(trace_length - len(sequenceX)):
                                sequenceX.append(sequenceX[-1])
                                sequenceY.append(sequenceY[-1])
                            batchX.append(sequenceX)
                            batchY.append(sequenceY)
                        sequenceX = []
                        sequenceY = []
                        break

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                print('batch x, y : {}, {}'.format(np.shape(batchX), np.shape(batchY)))
                # loss = model.train_on_batch([batchX], [batchY])
                # video_per_loss.append(loss)
                # print("loss: ", loss)

                cap.release()
                cv2.destroyAllWindows()
            i_end = time.time()

            # plot
            # plt.figure()
            # plt.plot(video_per_loss)
            # plt.title(str(epochs) + "_" + video)
            # plt.xlabel('t')
            # plt.ylabel('loss')
            # plt.show()

            # epochs_per_loss.append(sum(video_per_loss) / len(video_per_loss))

            print("video task take {}s".format(i_end - i_start))
            inner_path = os.path.join("model/supervised", video)
            if not (os.path.isdir(inner_path)):
                os.makedirs(inner_path)
            # save weight
            model.save_weights(os.path.join(inner_path, "_model.h5"))
            print("Saved model to disk")
        t_end = time.time()
        print("total task take {}s".format(t_end - t_start))

        # plt.plot(epochs_per_loss)
        # plt.title(str(epochs) + "_" + "epochs")
        # plt.xlabel('t')
        # plt.ylabel('loss')
        # plt.show()

        outer_path = os.path.join("model/supervised/_" + str(epochs))
        if not (os.path.isdir(outer_path)):
            os.makedirs(outer_path)
        model.save_weights(outer_path, "_model.h5")
        print("{}th epochs saved model to disk".format(epochs))

        epochs += 1
