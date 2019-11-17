import tensorflow as tf
import tensorflow.contrib.distributions as dist


class ActorCritic:
    def __init__(self, sess, obs, acs, hidden_size, name, trainable, init_std=1.0):
        self.sess = sess
        self.obs = obs
        self.acs = acs
        self.hidden_size = hidden_size
        self.name = name
        self.trainable = trainable
        self.init_std = init_std

        self.num_ac = self.acs.get_shape().as_list()[-1]

        with tf.variable_scope(name):
            self._build_network()

    def _build_network(self):
        with tf.variable_scope('critic'):
            s_layer_1 = tf.keras.layers.ConvLSTM2D(64, 3, name='state_layer1', return_sequences=True)(self.obs)
            s_batch_norm_1 = tf.keras.layers.BatchNormalization(name='state_batch_norm1')(s_layer_1)

            s_layer_2 = tf.keras.layers.ConvLSTM2D(64, 3, name='state_layer2', return_sequences=True)(s_batch_norm_1)
            s_batch_norm_2 = tf.keras.layers.BatchNormalization(name='state_batch_norm2')(s_layer_2)

            s_layer_3 = tf.keras.layers.ConvLSTM2D(64, 3, name='state_layer3', return_sequences=False)(s_batch_norm_2)
            s_batch_norm_3 = tf.keras.layers.BatchNormalization(name='state_batch_norm3')(s_layer_3)

            s_flatten = tf.keras.layers.Flatten(name='state_flatten')(s_batch_norm_3)

            c_out = tf.keras.layers.Dense(1, name='c_out')(s_flatten)

        with tf.variable_scope('actor'):
            s_layer_1 = tf.keras.layers.ConvLSTM2D(64, 3, name='state_layer1', return_sequences=True)(self.obs)
            s_batch_norm_1 = tf.keras.layers.BatchNormalization(name='state_batch_norm1')(s_layer_1)

            s_layer_2 = tf.keras.layers.ConvLSTM2D(64, 3, name='state_layer2', return_sequences=True)(s_batch_norm_1)
            s_batch_norm_2 = tf.keras.layers.BatchNormalization(name='state_batch_norm2')(s_layer_2)

            s_layer_3 = tf.keras.layers.ConvLSTM2D(64, 3, name='state_layer3', return_sequences=False)(s_batch_norm_2)
            s_batch_norm_3 = tf.keras.layers.BatchNormalization(name='state_batch_norm3')(s_layer_3)

            s_flatten = tf.keras.layers.Flatten(name='state_flatten')(s_batch_norm_3)

            a_out = tf.keras.layers.Dense(1, name='c_out')(s_flatten)

            log_std = tf.get_variable('log_std', [1, self.num_ac], dtype=tf.float32,
                                      initializer=tf.constant_initializer(self.init_std),
                                      trainable=self.trainable)

        std = tf.exp(log_std)
        a_dist = dist.Normal(a_out, std)
        self.log_prob = a_dist.log_prob(self.acs)
        self.entropy = tf.reduce_mean(a_dist.entropy())

        self.value = tf.identity(c_out)
        self.action = a_dist.sample()

    def params(self):
        return tf.global_variables(self.name).copy()
