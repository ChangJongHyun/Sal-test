# coding=utf8

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

#### HYPER PARAMETERS ####
gamma = 0.99  # reward discount factor

h_critic = 16
h_actor = 16

lr_critic = 3e-3  # learning rate for the critic
lr_actor = 1e-3  # learning rate for the actor

tau = 1e-2  # soft target update rate

save_file = "./results/nn/"
load_file = "./results/nn/" + '-5000'


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


class DDPG:
    def __init__(self, state_dim, action_dim, action_max, action_min):

        tf.reset_default_graph()
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_max = float(action_max)
        self.action_min = float(action_min)

        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim])
        self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[None])
        self.next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim])
        self.done_ph = tf.placeholder(dtype=tf.float32, shape=[None])
        with tf.variable_scope('network'):
            self.input = self.generate_network()
        with tf.variable_scope('actor'):
            self.action = self.generate_actor_network(self.state_ph, True)
        with tf.variable_scope('target_actor'):
            self.target_action = self.generate_actor_network(self.next_state_ph, False)
        with tf.variable_scope('critic'):
            self.qvalue = self.generate_critic_network(self.state_ph, self.action, True)
        with tf.variable_scope('target_critic'):
            self.target_qvalue = self.generate_critic_network(self.next_state_ph, self.target_action, False)

        self.a_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')
        self.ta_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_actor')
        self.c_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')
        self.tc_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_critic')

        q_target = tf.expand_dims(self.reward_ph, 1) + gamma * self.target_qvalue * (
                1 - tf.expand_dims(self.done_ph, 1))
        td_errors = q_target - self.qvalue
        critic_loss = tf.reduce_mean(tf.square(td_errors))
        self.train_critic = tf.train.AdamOptimizer(lr_critic).minimize(critic_loss, var_list=self.c_params)

        actor_loss = - tf.reduce_mean(self.qvalue)
        self.train_actor = tf.train.AdamOptimizer(lr_actor).minimize(actor_loss, var_list=self.a_params)

        self.soft_target_update = [[tf.assign(ta, (1 - tau) * ta + tau * a), tf.assign(tc, (1 - tau) * tc + tau * c)]
                                   for a, ta, c, tc in
                                   zip(self.a_params, self.ta_params, self.c_params, self.tc_params)]

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        # if not FLAGS.train:
        #     self.saver.restore(self.sess, load_file)

    def choose_action(self, state):
        return self.sess.run(self.action, feed_dict={self.state_ph: state[None]})[0]

    def train_network(self, state, action, reward, next_state, done, step):

        self.sess.run(self.train_critic, feed_dict={self.state_ph: state,
                                                    self.action: action,
                                                    self.reward_ph: reward,
                                                    self.next_state_ph: next_state,
                                                    self.done_ph: done})
        self.sess.run(self.train_actor, feed_dict={self.state_ph: state})
        self.sess.run(self.soft_target_update)

        if step % 1000 == 0:
            self.saver.save(self.sess, save_file, step)

    def generate_network(self, h_size, rnn_cell):
        conv1 = slim.convolution2d(
            inputs=self.state_ph, num_outputs=32,
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

    def generate_critic_network(self, rnn_out):
        # hidden1 = tf.layers.dense(tf.concat([state, action], axis=1), h_critic, activation=tf.nn.relu,
        #                           trainable=trainable)
        # hidden2 = tf.layers.dense(hidden1, h_critic, activation=tf.nn.relu, trainable=trainable)
        # hidden3 = tf.layers.dense(hidden2, h_critic, activation=tf.nn.relu, trainable=trainable)
        #
        # qvalue = tf.layers.dense(hidden3, 1, trainable=trainable)
        #
        # return qvalue
        return slim.fully_connected(rnn_out, 1,
                                    activation_fn=None,
                                    weights_initializer=normalized_columns_initializer(1.0),
                                    biases_initializer=None)

    def generate_actor_network(self, rnn_out, action_size):
        # hidden1 = tf.layers.dense(state, h_actor, activation=tf.nn.relu, trainable=trainable)
        # hidden2 = tf.layers.dense(hidden1, h_actor, activation=tf.nn.relu, trainable=trainable)
        # hidden3 = tf.layers.dense(hidden2, h_actor, activation=tf.nn.relu, trainable=trainable)
        #
        # non_scaled_action = tf.layers.dense(hidden3, self.action_dim, activation=tf.nn.sigmoid, trainable=trainable)
        # # action = non_scaled_action * (self.action_max - self.action_min) + self.action_min
        # action = non_scaled_action
        # return action
        return slim.fully_connected(rnn_out, action_size,
                                    activation_fn=tf.nn.softmax,
                                    weights_initializer=normalized_columns_initializer(0.01),
                                    biases_initializer=None)
