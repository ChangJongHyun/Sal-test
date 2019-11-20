# coding=utf8

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
    def __init__(self, sess, state_dim, action_dim, action_max, action_min, isRNN=True):
        self.sess = sess

        self.isRNN = isRNN

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_max = float(action_max)
        self.action_min = float(action_min)

        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None] + self.state_dim)
        self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None] + self.state_dim)
        # self.done_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])

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

        q_target = tf.expand_dims(self.reward_ph, 1) + gamma * self.target_qvalue # * (1 - tf.expand_dims(self.done_ph, 1))
        td_errors = q_target - self.qvalue
        critic_loss = tf.reduce_mean(tf.square(td_errors))
        self.train_critic = tf.train.AdamOptimizer(lr_critic).minimize(critic_loss)

        actor_loss = - tf.reduce_mean(self.qvalue)
        self.train_actor = tf.train.AdamOptimizer(lr_actor).minimize(actor_loss)

        self.soft_target_update = [[tf.assign(ta, (1 - tau) * ta + tau * a), tf.assign(tc, (1 - tau) * tc + tau * c)]
                                   for a, ta, c, tc in
                                   zip(self.a_params, self.ta_params, self.c_params, self.tc_params)]

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        self.actor_loss_summ = tf.summary.scalar('actor_loss', actor_loss)
        self.critic_loss_summ = tf.summary.scalar('critic_loss', critic_loss)

    def choose_action(self, state):
        return self.sess.run(self.action, feed_dict={self.state_ph: state})

    def update(self, writer, step, state, action, reward, next_state, done):
        merged = tf.summary.merge([self.actor_loss_summ, self.critic_loss_summ])

        self.sess.run(self.train_critic, feed_dict={self.state_ph: state,
                                                    self.action: action,
                                                    self.reward_ph: reward,
                                                    self.next_state_ph: next_state})
        self.sess.run(self.train_actor, feed_dict={self.state_ph: state})
        self.sess.run(self.soft_target_update)
        summ = self.sess.run(merged, feed_dict={self.state_ph: state,
                                                self.action: action,
                                                self.reward_ph: reward,
                                                self.next_state_ph: next_state})
        writer.add_summary(summ, step)

    # policy
    def generate_critic_network(self, state, action, trainable):
        if self.isRNN:
            s_layer_1 = tf.keras.layers.ConvLSTM2D(32, 3, name='state_layer1', return_sequences=True)(self.state_ph)
            s_batch_norm_1 = tf.keras.layers.BatchNormalization(name='state_batch_norm1')(s_layer_1)

            s_layer_2 = tf.keras.layers.ConvLSTM2D(32, 3, name='state_layer2', return_sequences=True)(s_batch_norm_1)
            s_batch_norm_2 = tf.keras.layers.BatchNormalization(name='state_batch_norm2')(s_layer_2)

            s_layer_3 = tf.keras.layers.ConvLSTM2D(32, 3, name='state_layer3', return_sequences=False)(s_batch_norm_2)
            s_batch_norm_3 = tf.keras.layers.BatchNormalization(name='state_batch_norm3')(s_layer_3)

            s_flatten = tf.keras.layers.Flatten(name='state_flatten')(s_batch_norm_3)

            c_out = tf.keras.layers.Dense(1, name='c_out', activation=None)(s_flatten)

        else:
            s_layer_1 = tf.keras.layers.Conv3D(32, 3, name='state_layer1')(self.state_ph)
            s_layer_2 = tf.keras.layers.Conv3D(32, 3, name='state_layer2')(s_layer_1)
            s_layer_3 = tf.keras.layers.Conv3D(32, 3, name='state_layer2')(s_layer_2)
            s_flatten = tf.keras.layers.Flatten(name='state_flatten')(s_layer_3)
            c_out = tf.keras.layers.Dense(1, name='c_out', activation=None)(s_flatten)
        return c_out

    # action-value
    def generate_actor_network(self, state, trainable):
        if self.isRNN:
            s_layer_1 = tf.keras.layers.ConvLSTM2D(32, 3, name='state_layer1', return_sequences=True)(self.state_ph)
            s_batch_norm_1 = tf.keras.layers.BatchNormalization(name='state_batch_norm1')(s_layer_1)

            s_layer_2 = tf.keras.layers.ConvLSTM2D(32, 3, name='state_layer2', return_sequences=True)(s_batch_norm_1)
            s_batch_norm_2 = tf.keras.layers.BatchNormalization(name='state_batch_norm2')(s_layer_2)

            s_layer_3 = tf.keras.layers.ConvLSTM2D(32, 3, name='state_layer3', return_sequences=False)(s_batch_norm_2)
            s_batch_norm_3 = tf.keras.layers.BatchNormalization(name='state_batch_norm3')(s_layer_3)

            s_flatten = tf.keras.layers.Flatten(name='state_flatten')(s_batch_norm_3)

            a_out = tf.keras.layers.Dense(2, name='a_out', activation='tanh')(s_flatten)

        else:
            s_layer_1 = tf.keras.layers.Conv3D(32, 3, name='state_layer1')(self.state_ph)
            s_layer_2 = tf.keras.layers.Conv3D(32, 3, name='state_layer2')(s_layer_1)
            s_layer_3 = tf.keras.layers.Conv3D(32, 3, name='state_layer2')(s_layer_2)
            s_flatten = tf.keras.layers.Flatten(name='state_flatten')(s_layer_3)
            a_out = tf.keras.layers.Dense(2, name='a_out', activation='tanh')(s_flatten)

        action = a_out * (self.action_max - self.action_min) + self.action_min
        return action
