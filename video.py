import cv2
import os
from custom_env.gym_my_env.envs.viewport import Viewport
import csv
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from dataset import Sal360

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


def generate_network(inputs,batch_size, trainLength, h_size):
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


video_dir = 'sample_videos'
train_dir = os.path.join(video_dir, '320x160')
test_dir = os.path.join(video_dir, '3840x1920')
scanpath_h = os.path.join('datasets/Scanpaths_H', 'Scanpaths')

# gpu_options = tf.GPUOptions(allow_growth=True)
# config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
# sess = tf.Session(config=config)

image_in = tf.placeholder(dtype=tf.float32, shape=(None, 84, 84, 3))
batch_size = tf.placeholder(dtype=tf.int32, shape=[])
train_length = tf.placeholder(dtype=tf.int32, shape=[])
h_size = 512

network = generate_network(image_in, batch_size, train_length, h_size)

# sess.run(tf.global_variables_initializer())

dataset = Sal360().read_scanpath_H()

for video, data in zip(os.listdir(train_dir), dataset['train']):
    # data --> [45, 100, 7]
    cap = cv2.VideoCapture(os.path.join(train_dir, video))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    view = Viewport(width, height)

    tracker = []

    for scan in data:
        c_idx = 0
        idx = 0
        cap = cv2.VideoCapture(os.path.join(train_dir, video))
        while True:
            ret, frame = cap.read()

            if c_idx < 100 and idx % 5 == 0:
                w = float(scan[c_idx][2]) * width
                h = float(scan[c_idx][1]) * height
                view.set_center(np.array([w, h]))
                c_idx += 1

            if ret:
                frame = view.get_view(frame)
                frame = cv2.resize(frame, (84, 84))
                cv2.imshow("video", frame)
                idx += 1
            else:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()