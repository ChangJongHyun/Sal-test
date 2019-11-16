import tensorflow as tf


class Discriminator:
    def __init__(self, observation_space, action_space):
        with tf.variable_scope('discriminator'):
            self.scope = tf.get_variable_scope().name
            self.expert_s = tf.placeholder(dtype=tf.float32, shape=[None] + list(observation_space))
            self.expert_a = tf.placeholder(dtype=tf.float32, shape=[None] + list(action_space))
            # expert_a_one_hot = tf.one_hot(self.expert_a, depth=action_space)
            # add noise for stabilise training
            # expert_a_one_hot += tf.random_normal(tf.shape(expert_a_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2
            # expert_s_a = tf.concat([self.expert_s, expert_a_one_hot], axis=1)

            self.agent_s = tf.placeholder(dtype=tf.float32, shape=[None] + list(observation_space))
            self.agent_a = tf.placeholder(dtype=tf.float32, shape=[None] + list(action_space))
            # agent_a_one_hot = tf.one_hot(self.agent_a, depth=action_space)
            # add noise for stabilise training
            # agent_a_one_hot += tf.random_normal(tf.shape(agent_a_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32) / 1.2
            # agent_s_a = tf.concat([self.agent_s, agent_a_one_hot], axis=1)

            with tf.variable_scope('network') as network_scope:
                expert_prob = self.construct_network(self.expert_s, self.expert_a)
                network_scope.reuse_variables()  # share parameter
                agent_prob = self.construct_network(self.agent_s, self.agent_a)

            with tf.variable_scope('loss'):
                loss_expert = tf.reduce_mean(tf.log(tf.clip_by_value(expert_prob, 0.01, 1)))
                loss_agent = tf.reduce_mean(tf.log(tf.clip_by_value(1 - agent_prob, 0.01, 1)))
                loss = loss_expert + loss_agent
                loss = -loss
                tf.summary.scalar('discriminator', loss)

            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(loss)

            self.rewards = tf.log(tf.clip_by_value(agent_prob, 1e-10, 1))  # log(P(expert|s,a)) larger is better for agent

    def construct_network(self, state_input, action_input):
        s_layer_1 = tf.keras.layers.ConvLSTM2D(64, 3, name='state_layer1', return_sequences=True)(state_input)
        s_batch_norm_1 = tf.keras.layers.BatchNormalization(name='state_batch_norm1')(s_layer_1)

        s_layer_2 = tf.keras.layers.ConvLSTM2D(64, 3, name='state_layer2', return_sequences=True)(s_batch_norm_1)
        s_batch_norm_2 = tf.keras.layers.BatchNormalization(name='state_batch_norm2')(s_layer_2)

        s_layer_3 = tf.keras.layers.ConvLSTM2D(64, 3, name='state_layer3', return_sequences=False)(s_batch_norm_2)
        s_batch_norm_3 = tf.keras.layers.BatchNormalization(name='state_batch_norm3')(s_layer_3)

        s_flatten = tf.keras.layers.Flatten(name='state_flatten')(s_batch_norm_3)

        a_layer_1 = tf.keras.layers.Dense(20, activation='relu', name='action_layer1')(action_input)
        a_layer_2 = tf.keras.layers.Dense(20, activation='relu', name='action_layer2')(a_layer_1)
        a_layer_3 = tf.keras.layers.Dense(20, activation='relu', name='action_layer2')(a_layer_2)

        concat = tf.keras.layers.concatenate([s_flatten, a_layer_3], name='s_a_concat')
        
        prob = tf.keras.layers.Dense(1, name='prob')(concat)
        return prob

    def train(self, expert_s, expert_a, agent_s, agent_a):
        return tf.get_default_session().run(self.train_op, feed_dict={self.expert_s: expert_s,
                                                                      self.expert_a: expert_a,
                                                                      self.agent_s: agent_s,
                                                                      self.agent_a: agent_a})

    def get_rewards(self, agent_s, agent_a):
        return tf.get_default_session().run(self.rewards, feed_dict={self.agent_s: agent_s,
                                                                     self.agent_a: agent_a})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


if __name__ == '__main__':
    obs = (None, 224, 224, 3)
    action = (2,)
    disc = Discriminator(obs, action)