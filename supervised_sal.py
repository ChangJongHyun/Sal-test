import os

import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential
from keras.activations import softsign
from keras.layers import TimeDistributed, Convolution2D, Flatten, LSTM, Dense
from keras.losses import KLD, binary_crossentropy
from keras.optimizers import Adam
from keras.utils.multi_gpu_utils import multi_gpu_model

from dataset import DataGenerator
from model import Networks
from my_resnet import dcn_resnet


# 1. supervided learning cnn + rnn
# 2. RL --> object detection --> tracking (saliency) --> reward을
# 3. IRL --> reward function
# action 정의 매프레임? 묶어서?
# multimodal
# 음향 어떻게? 3D JS?
if __name__ == '__main__':

    img_w = 224
    img_h = 224
    state_size = (None, img_w, img_h, 3)
    action_size = 2
    steps_per_epoch = 2485
    max_epochs = 10
    epochs = 10
    lr = 10e-2

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            # tf.config.experimental.set_memory_growth(gpus[1], True)
        except RuntimeError as e:
            # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
            print(e)

    train_gen = DataGenerator.generator_for_batch(img_w, img_h, type='train', normalize=False)
    validation_gen = DataGenerator.generator_for_batch(img_w, img_h, type='validation', normalize=False)
    test_gen = DataGenerator.generator_for_batch(img_w, img_h, type='test', normalize=False)

    model = Networks.drqn(state_size, action_size, backbone='2.5D')
    loss_arr = []

    hist = model.fit_generator(train_gen, epochs=10, verbose=1,
                               steps_per_epoch=3000,
                               shuffle=False,
                               validation_data=validation_gen,
                               validation_steps=1000)
    model.save_weights("model/supervised/sal_model.h5")
    #
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()
    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')

    acc_ax.plot(hist.history['acc'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='upper left')

    plt.show()
