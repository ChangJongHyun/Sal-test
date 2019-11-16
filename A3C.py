import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from my_resnet import dcn_resnet
from keras.layers import Input, Dense
from keras.layers import CuDNNLSTM, LSTMCell, LSTM


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


sign_ary = [[0, 0], [0, 1], [1, 0], [1, 1], [0, -1], [-1, 0], [-1, -1], [-1, 1], [1, -1]]


class MyModel:
    def __init__(self):
        action_input = Input(shape=[None, 1])
        state_input = Input(shape=[None, 224, 224, 3])
        value_dcn = dcn_resnet()
        policy_dcn = dcn_resnet()  # 변수 공유? 다른 변수?
        value_lstm = CuDNNLSTM(256)(value_dcn, state_input)  # state-value : expected return
        policy_lstm = CuDNNLSTM(256)(policy_dcn)  # policy : agent's action selection
        self.value_model = Dense(1, activation='relu')(value_lstm)
        self.policy_model = Dense(1, activation='relu')(policy_lstm)

        self.action_max = 2
        self.conv1 = slim.conv2d(self.input_image,
                                 activation_fn=tf.nn.relu,
                                 num_outputs=32,
                                 kernel_size=[8, 8],
                                 stride=[4, 4],
                                 padding='VALID')
        self.conv2 = slim.conv2d(self.conv1,
                                 activation_fn=tf.nn.relu,
                                 num_outputs=64,
                                 kernel_size=[4, 4],
                                 stride=[2, 2],
                                 padding='VALID')
        self.conv3 = slim.convolution2d(
            inputs=self.conv2, num_outputs=64,
            kernel_size=[3, 3], stride=[1, 1], padding='VALID',
            activation_fn=tf.nn.relu)

        self.conv4 = slim.convolution2d(
            inputs=self.conv3, num_outputs=256,
            kernel_size=[7, 7], stride=[1, 1], padding='VALID',
            activation_fn=tf.nn.relu)
        hidden = slim.fully_connected(slim.flatten(self.conv4), 256,
                                      activation_fn=tf.nn.relu)

        # temporal dependency
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(256, reuse=tf.AUTO_REUSE)
        c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)

        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])

        # self.state_in = (c_in, h_in)

        self.rnn_in = tf.expand_dims(hidden, [0])
        # step_size = tf.shape(self.imageIn[:1])  # 84 84 3
        self.state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)  # c --> hidden, h --> output

        lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(lstm_cell, self.rnn_in, initial_state=self.state_in,
                                                          time_major=False, scope="A3C")
        lstm_c, lstm_h = self.lstm_state
        self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
        rnn_out = tf.reshape(lstm_outputs, [-1, 256])

        # self.policy = slim.fully_connected(rnn_out, a_size,
        #                                    activation_fn=tf.nn.relu,
        #                                    weights_initializer=normalized_columns_initializer(0.01),
        #                                    biases_initializer=None)
        #
        # hidden1 = tf.layers.dense(rnn_out, 16, activation=tf.nn.relu)
        # hidden2 = tf.layers.dense(hidden1, 16, activation=tf.nn.relu)
        # hidden3 = tf.layers.dense(hidden2, 16, activation=tf.nn.relu)

        self.policy = tf.layers.dense(rnn_out, 9, activation=tf.nn.relu)
        self.policy = tf.nn.softmax(self.policy)
        self.value = slim.fully_connected(rnn_out, 1,
                                          activation_fn=None,
                                          weights_initializer=normalized_columns_initializer(1.0),
                                          biases_initializer=None)

        self.true_val = tf.placeholder(tf.int32, shape=[9])
        # self.error = tf.reduce_mean(tf.square(self.true_val - self.policy))
        # self.train_op = tf.train.AdamOptimizer(0.001)
        self.error = tf.nn.softmax_cross_entropy_with_logits(labels=self.true_val, logits=self.policy)
        # self.error = tf.reduce_mean(tf.square(tf.subtract(self.true_val, self.policy)))
        self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.error)

        self.saver = tf.train.Saver()

    # 프레임 1개와 state를 받고 output (true 값이 움직여야할 위치, policy가 deep neural net으로 이동한 위치)
    def pre_train(self, sess, inputs, state_in, true, step):
        action, _, loss = sess.run([self.policy, self.train_op, self.error], feed_dict={self.input_image: inputs,
                                                                                        self.state_in: state_in,
                                                                                        self.true_val: true})
        return action, loss

    def get_action(self, sess, inputs, state):
        action = sess.run(self.policy, feed_dict={
            self.input_image: inputs,
            self.state_in: state
        })
        return action * (self.action_max - self.action_min) + self.action_min

    def get_state(self, sess, inputs, state):
        return sess.run(self.state_out, feed_dict={
            self.input_image: inputs,
            self.state_in: state})


if __name__ == '__main__':
    s_size = 84 * 84 * 3
    a_size = 2  # softmax: n개의 action 중 1개 선택 --> relu: continuous [x, x]
