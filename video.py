import cv2
import os
from custom_env.gym_my_env.envs.viewport import Viewport
import csv
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from dataset import load_sal360_dataset
from A3C import AC_Network
import matplotlib.pyplot as plt
from resnet import Model
from my_resnet import dcn_resnet


def csv_reader(path):
    dataset = []
    with open(path, 'r') as file:
        data = csv.reader(file)
        tmp = []
        for row in data:
            if not row[0] == '# Idx':
                tmp.append(row)
                if int(row[0]) == 99:
                    dataset.append(tmp)
                    tmp = []
        file.close()
    return dataset


def generate_network(inputs, batch_size, trainLength, h_size):
    conv1 = slim.convolution2d(
        inputs=inputs, num_outputs=32,
        kernel_size=[8, 8], stride=[4, 4], padding='VALID',
        biases_initializer=None, scope='conv1')
    conv2 = slim.convolution2d(
        inputs=conv1, num_outputs=64,
        kernel_size=[4, 4], stride=[2, 2], padding='VALID',
        biases_initializer=None, scope='conv2')
    conv3 = slim.convolution2d(
        inputs=conv2, num_outputs=64,
        kernel_size=[3, 3], stride=[1, 1], padding='VALID',
        biases_initializer=None, scope='conv3')
    conv4 = slim.convolution2d(
        inputs=conv3, num_outputs=h_size,
        kernel_size=[7, 7], stride=[1, 1], padding='VALID',
        biases_initializer=None, scope='conv4')

    # We take the output from the final convolutional layer and send it to a recurrent layer.
    # The input must be reshaped into [batch x trace x units] for rnn processing,
    # and then returned to [batch x units] when sent through the upper levles.
    convFlat = tf.reshape(slim.flatten(conv4), [batch_size, trainLength, h_size])
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
    state_in = rnn_cell.zero_state(batch_size, tf.float32)
    rnn, rnn_state = tf.nn.dynamic_rnn(
        inputs=convFlat, cell=rnn_cell, dtype=tf.float32, initial_state=state_in, scope='rnn')

    return tf.reshape(rnn, shape=[-1, h_size])


video_dir = 'dataset/Stimuli'
train_dir = os.path.join(video_dir, '320x160')
test_dir = os.path.join(video_dir, '3840x1920')
scanpath_h = os.path.join('dataset/H', 'Scanpaths')

image_in = tf.placeholder(dtype=tf.float32, shape=(None, 84, 84, 3))
batch_size = tf.placeholder(dtype=tf.int32, shape=[])
train_length = tf.placeholder(dtype=tf.int32, shape=[])
h_size = 512

network = generate_network(image_in, batch_size, train_length, h_size)

dataset = load_sal360_dataset()

a_size = 2
s_size = 84 * 84 * 3

net = AC_Network(s_size, a_size, scope=None, trainer=True)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

# TODO state 작성

sign_ary = [[0., 0.], [0., 1.], [1., 0.], [1., 1.], [0., -1.], [-1., 0.], [-1., -1.], [-1., 1.], [1., -1.]] #

for video, data in zip(sorted(os.listdir(video_dir)), dataset['train']):
    # data --> [45, 100, 7]
    cap = cv2.VideoCapture(os.path.join(video_dir, video))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    view = Viewport(width, height)

    m = dcn_resnet((width, height, 3))

    print("video name : ", video)
    loss = []
    for scan in data:
        c_idx = 0
        idx = 0
        cap = cv2.VideoCapture(os.path.join(video_dir, video))
        state = (np.zeros([1, 256]), np.zeros([1, 256]))  # initial state

        while True:
            ret, frame = cap.read()

            if ret:
                frame = view.get_view(frame)
                frame = cv2.resize(frame, (84, 84))

                if c_idx < 100 and idx % 5 == 0:
                    w = float(scan[c_idx][2]) * width
                    h = float(scan[c_idx][1]) * height
                    true = np.array([w, h])
                    move = (true - view.center)  # 현재 view 위치에서 다음 5프레임 이후 view의 위치 action

                    c_idx += 1
                    sign = np.sign(move)
                    val = np.zeros(9)
                    val[sign_ary.index(list(sign))] = 1
                    move_abs = np.abs(move)

                    state1 = net.get_state(sess, [frame], state)  # state 업데이트
                    action, l = net.pre_train(sess, [frame], state, val, c_idx)
                    # action = net.get_action(sess, [frame], state)
                    # action, l = net.pre_train(sess, [frame], state, move, c_idx)  # policy를 통해 action과 loss return
                    state = state1

                    view.move(move)
                    print("action, true, loss -> {} {} {}".format(np.argmax(action), val, l))

                    # loss.append(l)
                # print(np.shape(action))  # 1,2 --> np.reshape(action, [2,])
                # 참 값을 설정해줘야되.. 왼쪽? 오른쪽? 위쪽? 아래쪽?
                # cv2.imshow("video", frame)
                idx += 1
            else:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        plt.plot(loss)
    cap.release()
    cv2.destroyAllWindows()


def resnet(input_tensor=None):
    input_shape = (84, 84, 3)  # WHC

    bn_axis = 1
