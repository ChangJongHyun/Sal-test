import os

import tensorflow as tf
import numpy as np

import GAIL.jupyter.tf_utils as U


def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)


""" Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51"""
def logit_bernoulli_entropy(logits):
    ent = (1. - tf.nn.sigmoid(logits)) * logits - logsigmoid(logits)
    return ent


class Discriminator:
    def __init__(self, sess, ob_shape, ac_shape, name, entcoeff=0.001, lr=1e-3, isRNN=True):
        self.sess = sess
        self.ob_shape = ob_shape
        self.ac_shape = ac_shape
        self.num_actions = ac_shape[0]
        self.lr = lr
        self.name = name
        self.isRNN = isRNN

        self.build_ph()
        # Build graph
        generator_logits = self.build_graph(self.generator_obs_ph, self.generator_acs_ph, reuse=False)
        expert_logits = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=False)

        # Build accuracy
        generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))
        expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))
        # Build regression loss
        # let x = logits, z = targes
        # z * -log(sigmoid(x)) + (1-z) * -log(1-sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits,
                                                                 labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits,
                                                              labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(expert_loss)

        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits=logits))
        entropy_loss = -entcoeff * entropy

        # Loss + Accuracy terms
        self.total_loss = generator_loss + expert_loss + entropy_loss
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss, var_list=self.get_trainable_variables())

        # Build Reward for policy
        self.reward_op = -tf.log(1 - tf.nn.sigmoid(generator_logits) + 1e-8)

        generator_acc_summary = tf.summary.histogram('generator_acc', generator_acc)
        expert_acc_summary = tf.summary.histogram('expert_acc', expert_acc)
        total_loss_summary = tf.summary.scalar('total_loss', self.total_loss)

        self.merged = tf.summary.merge([generator_acc_summary, expert_acc_summary, total_loss_summary])

    def build_ph(self):
        self.generator_obs_ph = tf.placeholder(tf.float32, [None] + self.ob_shape, name="observations_ph")
        self.generator_acs_ph = tf.placeholder(tf.float32, [None] + self.ac_shape, name="actions_ph")
        self.expert_obs_ph = tf.placeholder(tf.float32, [None] + self.ob_shape, name="expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(tf.float32, [None] + self.ac_shape, name="expert_actions_ph")

    def build_graph(self, obs_ph, acs_ph, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            # with tf.variable_scope('obfilter'):
            #     self.obs_rms = RunningMeanStd(shape=self.obs_rms)
            # leaky_relu = tf.keras.layers.LeakyReLU(0.2)
            # initializer = tf.keras.initializers.RandomNormal(0, 0.02)
            if self.isRNN:
                masking = tf.keras.layers.Masking(mask_value=0., input_shape=(8, 112, 112, 3))(obs_ph)
                x = tf.keras.layers.ConvLSTM2D(64, 3, 3, return_sequences=True)(obs_ph)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.ConvLSTM2D(64, 3, 3, return_sequences=True)(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.ConvLSTM2D(64, 3, 3, return_sequences=False)(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Flatten()(x)

                y = tf.keras.layers.Dense(64, activation='tanh')(acs_ph)

                transition = tf.concat([x, y], axis=1)
                logits = tf.keras.layers.Dense(1)(transition)
            else:
                backbone = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False,
                                                                          weights=None,
                                                                          pooling='max')
                x = tf.keras.layers.TimeDistributed(backbone)(obs_ph)
                x = tf.keras.layers.Flatten()(x)

                y = tf.keras.layers.Dense(64, activation='tanh')(acs_ph)

                transition = tf.concat([x, y], axis=1)
                logits = tf.keras.layers.Dense(1)(transition)

            return logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

    def learn(self, writer, step, generator_obs, generator_acs, expert_obs, expert_acs):
        feed_dict = {self.generator_obs_ph: generator_obs,
                     self.generator_acs_ph: generator_acs,
                     self.expert_obs_ph: expert_obs,
                     self.expert_acs_ph: expert_acs}
        _, summary = self.sess.run([self.train_op, self.merged], feed_dict)
        writer.add_summary(summary, step)

    def get_reward(self, obs, acs):
        feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: acs}
        reward = self.sess.run(self.reward_op, feed_dict)
        return reward
