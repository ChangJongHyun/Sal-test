import os

import tensorflow as tf


class Discriminator:
    def __init__(self, sess, ob_shape, ac_shape, hidden_size, lr, name, isRNN=True, isTPU=False):
        self.sess = sess
        self.ob_shape = ob_shape
        self.ac_shape = ac_shape
        self.hidden_size = hidden_size
        self.lr = lr
        self.name = name
        self.isRNN = isRNN
        self.isTPU = isTPU
        with tf.variable_scope('discriminator'):
            # self.ob_ac = tf.placeholder(dtype=tf.float32, shape=[None, ob_shape[0] + ac_shape[0]])
            self.expert_ob = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.ob_shape))
            self.expert_ac = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.ac_shape))

            self.agent_ob = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.ob_shape))
            self.agent_ac = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.ac_shape))
            with tf.variable_scope(name) as network_scope:
                expert_prob = self._build_network(self.expert_ob, self.expert_ac)
                network_scope.reuse_variables()  # share parameter
                agent_prob = self._build_network(self.agent_ob, self.agent_ac)

            with tf.variable_scope('loss'):
                loss_expert = tf.reduce_mean(tf.log(tf.clip_by_value(expert_prob, 0.01, 1)))
                loss_agent = tf.reduce_mean(tf.log(tf.clip_by_value(1 - agent_prob, 0.01, 1)))
                loss = loss_expert + loss_agent
                loss = -loss
                self.loss_summary = tf.summary.scalar('discriminator_loss', loss)

            with tf.name_scope('train_op'):
                # grads = tf.gradients(loss, self.params())
                # self.grads = list(zip(grads, self.params()))
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

            # log(P(expert|s,a)) larger is better for agent
            self.rewards = tf.log(tf.clip_by_value(agent_prob, 1e-10, 1))
            self.reward_summary = tf.summary.histogram('reward', self.rewards)
            self.agent_prob_sum = tf.summary.histogram('agent_prob', agent_prob)
            self.expert_prob_sum = tf.summary.histogram('expert_prob', expert_prob)

    def _build_network(self, ob, ac):
        with tf.variable_scope('discriminator'):
            if self.isRNN:
                s_layer_1 = tf.keras.layers.ConvLSTM2D(64, 3, name='state_layer1', return_sequences=True)(ob)
                s_batch_norm_1 = tf.keras.layers.BatchNormalization(name='state_batch_norm1')(s_layer_1)

                s_layer_2 = tf.keras.layers.ConvLSTM2D(64, 3, name='state_layer2', return_sequences=True)(
                    s_batch_norm_1)
                s_batch_norm_2 = tf.keras.layers.BatchNormalization(name='state_batch_norm2')(s_layer_2)

                s_layer_3 = tf.keras.layers.ConvLSTM2D(32, 3, name='state_layer3', return_sequences=False)(
                    s_batch_norm_2)
                s_batch_norm_3 = tf.keras.layers.BatchNormalization(name='state_batch_norm3')(s_layer_3)

                s_flatten = tf.keras.layers.Flatten(name='state_flatten')(s_batch_norm_3)
                x, y = tf.split(ac, [1, 1], 1)
                a_layer_1 = tf.keras.layers.Dense(16, activation='relu', name='action_layer_x')(x)
                a_layer_2 = tf.keras.layers.Dense(16, activation='relu', name='action_layer_y')(y)

                concat = tf.keras.layers.concatenate([s_flatten, a_layer_1, a_layer_2], name='s_a_concat')

                d_out = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, name='prob')(concat)
                return d_out
            else:
                s_layer_1 = tf.keras.layers.Conv3D(40, 3, name='state_layer1')(ob)
                s_batch_norm_1 = tf.keras.layers.BatchNormalization(name='state_batch_norm1')(s_layer_1)

                s_layer_2 = tf.keras.layers.Conv3D(40, 3, name='state_layer2')(s_batch_norm_1)
                s_batch_norm_2 = tf.keras.layers.BatchNormalization(name='state_batch_norm2')(s_layer_2)

                s_layer_3 = tf.keras.layers.Conv3D(40, 3, name='state_layer3')(s_batch_norm_2)
                s_batch_norm_3 = tf.keras.layers.BatchNormalization(name='state_batch_norm3')(s_layer_3)

                s_flatten = tf.keras.layers.Flatten(name='state_flatten')(s_batch_norm_3)

                x, y = tf.split(ac, [1, 1], 1)
                a_layer_1 = tf.keras.layers.Dense(16, activation='relu', name='action_layer_x')(x)
                a_layer_2 = tf.keras.layers.Dense(16, activation='relu', name='action_layer_y')(y)

                concat = tf.keras.layers.concatenate([s_flatten, a_layer_1, a_layer_2], name='s_a_concat')

                d_out = tf.keras.layers.Dense(1, name='prob')(concat)
                return d_out

    def params(self):
        return tf.global_variables(os.path.join('discriminator', self.name)).copy()

    def get_reward(self, agent_ob, agent_ac):
        feed_dict = {self.agent_ob: agent_ob,
                     self.agent_ac: agent_ac}

        return self.sess.run(self.rewards, feed_dict=feed_dict)

    def update(self, writer, step, expert_ob, expert_ac, agent_ob, agent_ac):
        feed_dict = {self.expert_ob: expert_ob,
                     self.expert_ac: expert_ac,
                     self.agent_ob: agent_ob,
                     self.agent_ac: agent_ac}

        merged = tf.summary.merge([self.loss_summary, self.reward_summary, self.agent_prob_sum, self.expert_prob_sum])
        op_list = [self.train_op, merged]
        _, summ = self.sess.run(op_list, feed_dict=feed_dict)
        writer.add_summary(summ, step)
