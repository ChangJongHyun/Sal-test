import cv2
import os
import csv
import numpy as np

import tensorflow as tf
from tensorflow.contrib import slim

video_dir = 'sample_videos'
train_dir = os.path.join(video_dir, '320x160')
test_dir = os.path.join(video_dir, '3840x1920')

h_size = 512

# Policy
class Actor:
    def __init__(self):
        self.network = None


# Action - Value function
class Critic:
    def __init__(self):
        self.network = None


def generate_network(input, h_size, rnn_cell):
    conv1 = slim.convolution2d(
        inputs=input, num_outputs=32,
        kernel_size=[8, 8], stride=[4, 4], padding='VALID',
        biases_initializer=None, scope='_conv1')
    conv2 = slim.convolution2d(
        inputs=conv1, num_outputs=64,
        kernel_size=[4, 4], stride=[2, 2], padding='VALID',
        biases_initializer=None, scope='_conv2')
    conv3 = slim.convolution2d(
        inputs=conv2, num_outputs=64,
        kernel_size=[3, 3], stride=[1, 1], padding='VALID',
        biases_initializer=None, scope='_conv3')
    conv4 = slim.convolution2d(
        inputs=conv3, num_outputs=h_size,
        kernel_size=[7, 7], stride=[1, 1], padding='VALID',
        biases_initializer=None, scope='_conv4')

    # We take the output from the final convolutional layer and send it to a recurrent layer.
    # The input must be reshaped into [batch x trace x units] for rnn processing,
    # and then returned to [batch x units] when sent through the upper levles.
    convFlat = tf.reshape(slim.flatten(conv4), [self.batch_size, self.trainLength, h_size])
    # rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
    self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
    rnn, self.rnn_state = tf.nn.dynamic_rnn(
        inputs=convFlat, cell=rnn_cell, dtype=tf.float32, initial_state=self.state_in, scope='_rnn')
    return tf.reshape(rnn, shape=[-1, h_size])


def csv_reader(path):
    dataset = []
    with open(path, 'r') as file:
        data = csv.reader(file)
        tmp = []
        for row in data:
            if not row[0] == '# Idx':
                row = row
                tmp.append(row)
                if int(row[0]) == 99:
                    dataset.append(tmp)
                    tmp = []
        file.close()
        print(np.shape(dataset[0]))
    return dataset


for video in os.listdir(train_dir):
    cap = cv2.VideoCapture(os.path.join(train_dir, video))

    # cellT = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
    # network = generate_network()
    while True:
        ret, frame = cap.read()

        if ret:
            cv2.imshow('video', frame)
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
