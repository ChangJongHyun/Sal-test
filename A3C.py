import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer):
        self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        self.imageIn = tf.reshape(self.inputs, shape=[-1, 84, 84, 3])
        self.conv1 = slim.conv2d(self.imageIn,
                                 activation_fn=tf.nn.elu,
                                 num_outputs=32,
                                 kernel_size=[8, 8],
                                 stride=[4, 4],
                                 padding='VALID')
        self.conv2 = slim.conv2d(self.conv1,
                                 activation_fn=tf.nn.elu,
                                 num_outputs=64,
                                 kernel_size=[4, 4],
                                 stride=[2, 2],
                                 padding='VALID')
        hidden = slim.fully_connected(slim.flatten(self.conv2), 256,
                                      activation_fn=tf.nn.elu)

        # temporal dependency
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(256, reuse=tf.AUTO_REUSE)
        c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
        self.state_in = (c_in, h_in)
        rnn_in = tf.expand_dims(hidden, [0])
        step_size = tf.shape(self.imageIn[:1])  # 84 84 3
        state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in, initial_state=state_in,
                                                     sequence_length=step_size, time_major=False, scope="A3C")
        lstm_c, lstm_h = lstm_state
        self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
        rnn_out = tf.reshape(lstm_outputs, [-1, 256])

        # self.policy = slim.fully_connected(rnn_out, a_size,
        #                                    activation_fn=tf.nn.relu,
        #                                    weights_initializer=normalized_columns_initializer(0.01),
        #                                    biases_initializer=None)

        hidden1 = tf.layers.dense(rnn_out, 16, activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, 16, activation=tf.nn.relu)
        hidden3 = tf.layers.dense(hidden2, 16, activation=tf.nn.relu)

        self.policy = tf.layers.dense(hidden3, a_size, activation=tf.nn.relu)

        self.value = slim.fully_connected(rnn_out, 1,
                                          activation_fn=None,
                                          weights_initializer=normalized_columns_initializer(1.0),
                                          biases_initializer=None)

    def get_action(self, sess, inputs, state):
        action = sess.run(self.policy, feed_dict={
            self.inputs: inputs,
            self.state_in: state
        })
        return action


if __name__ == '__main__':
    s_size = 84 * 84 * 3
    a_size = 2  # softmax: n개의 action 중 1개 선택 --> relu: continuous [x, x]
