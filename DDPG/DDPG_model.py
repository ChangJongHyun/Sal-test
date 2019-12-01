# coding=utf8
import os

import numpy as np
import tensorflow as tf

#### HYPER PARAMETERS ####
gamma = 0.99  # reward discount factor

h_critic = 16
h_actor = 16

lr_critic = 3e-3  # learning rate for the critic
lr_actor = 1e-3  # learning rate for the actor

tau = 1e-2  # soft target update rate


class DDPG:
    def __init__(self, sess, state_dim, action_dim, action_max, action_min, isRNN=False, load=None):
        self.sess = sess

        self.isRNN = isRNN
        time_step = 6
        self.state_dim = [time_step] + state_dim
        self.action_dim = action_dim
        self.action_max = float(action_max)
        self.action_min = float(action_min)

        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[1] + self.state_dim)
        self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[1, 1])
        self.next_state_ph = tf.placeholder(dtype=tf.float32, shape=[1] + self.state_dim)

        # self.done_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        with tf.variable_scope('actor'):
            self.action = self.generate_actor_network(self.state_ph, True, isTarget=False)
        with tf.variable_scope('target_actor'):
            self.target_action = self.generate_actor_network(self.next_state_ph, False, isTarget=True)
        with tf.variable_scope('critic'):
            self.qvalue = self.generate_critic_network(self.state_ph, self.action, True, isTarget=False)
        with tf.variable_scope('target_critic'):
            self.target_qvalue = self.generate_critic_network(self.next_state_ph, self.target_action, False,
                                                              isTarget=True)

        self.a_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')
        self.ta_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_actor')
        self.c_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')
        self.tc_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_critic')

        q_target = tf.expand_dims(self.reward_ph,
                                  1) + gamma * self.target_qvalue  # * (1 - tf.expand_dims(self.done_ph, 1))
        td_errors = q_target - self.qvalue
        self.critic_loss = tf.reduce_mean(tf.square(td_errors))
        self.train_critic = tf.train.AdamOptimizer(lr_critic).minimize(self.critic_loss, var_list=self.c_params)

        self.actor_loss = - tf.reduce_mean(self.qvalue)
        if self.isRNN:
            self.train_actor = tf.train.AdamOptimizer(lr_actor).minimize(self.actor_loss)
        else:
            self.train_actor = tf.train.AdamOptimizer(lr_actor).minimize(self.actor_loss, var_list=self.a_params)

        self.soft_target_update = [[tf.assign(ta, (1 - tau) * ta + tau * a), tf.assign(tc, (1 - tau) * tc + tau * c)]
                                   for a, ta, c, tc in
                                   zip(self.a_params, self.ta_params, self.c_params, self.tc_params)]
        self.sess.run(tf.global_variables_initializer())
        # self.actor_loss_summ = tf.summary.scalar('actor_loss', actor_loss)
        # self.critic_loss_summ = tf.summary.scalar('critic_loss', critic_loss)
        self.reset_state()
        self.saver = tf.train.Saver()
        tf.get_default_graph()
        self.writer = tf.summary.FileWriter('./log_ddpg/', self.sess.graph)

        if load is not None:
            self.load(load)

    def choose_action(self, state):
        return self.sess.run(self.action, feed_dict={self.state_ph: state})

    def update(self, step, state, action, reward, next_state, done):
        # merged = tf.summary.merge([self.actor_loss_summ, self.critic_loss_summ])

        self.sess.run([self.train_critic, self.critic_loss], feed_dict={self.state_ph: state,
                                                                        self.action: action,
                                                                        self.reward_ph: reward,
                                                                        self.next_state_ph: next_state})
        self.sess.run(self.train_actor, feed_dict={self.state_ph: state})
        actor_loss, critic_loss = self.sess.run([self.actor_loss, self.critic_loss], feed_dict={self.state_ph: state,
                                                                                                self.action: action,
                                                                                                self.reward_ph: reward,
                                                                                                self.next_state_ph: next_state})
        self.sess.run(self.soft_target_update)
        # summ = self.sess.run(merged, feed_dict={self.state_ph: state,
        #                                         self.action: action,
        #                                         self.reward_ph: reward,
        #                                         self.next_state_ph: next_state})
        # if step % 100 == 0:
        #     self.writer.add_summary(summ, step)
        return actor_loss, critic_loss

    def save(self, save_path, epochs):
        self.saver.save(self.sess, os.path.join(save_path, 'model_ddpg_' + str(epochs) + '.ckpt'))

    def load(self, load_path):
        self.saver.restore(self.sess, load_path)

    # policy
    def generate_critic_network(self, state, action, trainable, isTarget=True):
        if self.isRNN:
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, 3))(state)
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, 3))(x)
            if isTarget:
                x = tf.keras.layers.ConvLSTM2D(32, 3, name='state_layer1', return_sequences=False,
                                               stateful=True,
                                               trainable=trainable)(x)
            else:
                self.critic_s_layer_1 = tf.keras.layers.ConvLSTM2D(32, 3, name='state_layer1', return_sequences=False,
                                                                   stateful=True,
                                                                   trainable=trainable)
                x = self.critic_s_layer_1(x)
            s_batch_norm_1 = tf.keras.layers.BatchNormalization(name='state_batch_norm1', trainable=trainable)(x)

            s_flatten = tf.keras.layers.Flatten(name='state_flatten', trainable=trainable)(s_batch_norm_1)

            y = tf.keras.layers.Dense(128)(action)
            concat = tf.keras.layers.concatenate([s_flatten, y])
            q_value = tf.keras.layers.Dense(1, name='c_out', activation=None, trainable=trainable)(concat)
            return q_value

    # action-value
    def generate_actor_network(self, state, trainable, isTarget=True):
        if self.isRNN:
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, 3))(state)
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, 3))(x)

            if isTarget:
                x = tf.keras.layers.ConvLSTM2D(32, 3, name='state_layer1', return_sequences=False,
                                               stateful=True, trainable=trainable)(x)
            else:
                self.actor_s_layer_1 = tf.keras.layers.ConvLSTM2D(32, 3, name='state_layer1', return_sequences=False,
                                                                  stateful=True, trainable=trainable)
                x = self.actor_s_layer_1(x)
            s_batch_norm_1 = tf.keras.layers.BatchNormalization(name='state_batch_norm1', trainable=trainable)(x)

            s_flatten = tf.keras.layers.Flatten(name='state_flatten', trainable=trainable)(s_batch_norm_1)

            action = tf.keras.layers.Dense(2, name='a_out', activation='tanh', trainable=trainable)(s_flatten)
            return action

    def reset_state(self):
        self.critic_s_layer_1.reset_states()
        self.actor_s_layer_1.reset_states()
