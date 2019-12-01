import os

import gym
import tensorflow as tf

from custom_env.gym_my_env.envs.my_env import MyEnv


class Reward:
    def __init__(self, sess, env, isRNN=True):
        self.sess = sess
        self.env = env
        self.obs_dim = [6] + list(self.env.observation_space.shape)
        self.acs_dim = list(self.env.action_space.shape)
        self.isRNN = isRNN
        self._build_ph()

        with tf.variable_scope("reward_network"):
            self.network = self._build_network(self.obs_ph, self.acs_ph)
        with tf.variable_scope("loss"):
            self.loss = tf.losses.sigmoid_cross_entropy(self.reward_ph, self.network)
        with tf.variable_scope("train_op"):
            self.train = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)

        self.init_graph()

        self.loss_summary = tf.summary.scalar('loss', self.loss)

        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter('./log_reward/', self.sess.graph)

    def init_graph(self):
        self.sess.run(tf.global_variables_initializer())

    def _build_ph(self):
        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=[None] + self.obs_dim)
        self.acs_ph = tf.placeholder(dtype=tf.float32, shape=[None] + self.acs_dim)
        self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    def _build_network(self, obs, acs):
        x = tf.keras.layers.ConvLSTM2D(64, 3, 3, return_sequences=True)(obs)
        x = tf.keras.layers.ConvLSTM2D(32, 3, 3, return_sequences=True)(x)
        x = tf.keras.layers.ConvLSTM2D(16, 3, 3, return_sequences=False)(x)
        x = tf.keras.layers.Flatten()(x)
        y = tf.keras.layers.Dense(64)(acs)

        transition = tf.concat([x, y], axis=1)
        reward = tf.keras.layers.Dense(1, activation='sigmoid')(transition)
        return reward

    def update(self, step, obs, acs, reward):

        feed_dict = {self.obs_ph: obs, self.acs_ph: acs,
                     self.reward_ph: reward}
        if step % 100 == 0:
            _, summary = self.sess.run([self.train, self.loss_summary], feed_dict)
            self.writer.add_summary(summary, step)
        else:
            self.sess.run(self.train, feed_dict)

    def get_reward(self, obs, acs):
        return self.sess.run(self.network, feed_dict={self.obs_ph: obs,
                                                      self.acs_ph: acs})

    def save(self, save_path, step):
        self.saver.save(self.sess, os.path.join(save_path, "model_reward_convlstm_" + str(step) + "ckpt"))

    def load(self, load_path):
        self.saver.restore(self.sess, load_path)
