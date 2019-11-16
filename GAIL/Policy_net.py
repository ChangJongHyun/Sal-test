import tensorflow as tf
import tensorflow.contrib.distributions as dist


class Policy_net:
    def __init__(self, name: str,  observation_space, action_space):
        """
        :param name: string
        :param env: gym env
        """

        ob_space = observation_space
        act_space = action_space

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None] + list(ob_space), name='obs')

            with tf.variable_scope('policy_net'):
                layer_1 = tf.keras.layers.ConvLSTM2D(64, 3, return_sequences=True)(self.obs)
                batch_norm_1 = tf.keras.layers.BatchNormalization()(layer_1)

                layer_2 = tf.keras.layers.ConvLSTM2D(64, 3, return_sequences=True)(batch_norm_1)
                batch_norm_2 = tf.keras.layers.BatchNormalization()(layer_2)

                layer_3 = tf.keras.layers.ConvLSTM2D(64, 3, return_sequences=False)(batch_norm_2)
                batch_norm_3 = tf.keras.layers.BatchNormalization()(layer_3)
                flatten = tf.keras.layers.Flatten()(batch_norm_3)

                self.act_probs = tf.keras.layers.Dense(act_space, activation='linear')(flatten)

            with tf.variable_scope('value_net'):
                layer_1 = tf.keras.layers.ConvLSTM2D(64, 3, return_sequences=True)(self.obs)
                batch_norm_1 = tf.keras.layers.BatchNormalization()(layer_1)

                layer_2 = tf.keras.layers.ConvLSTM2D(64, 3, return_sequences=True)(batch_norm_1)
                batch_norm_2 = tf.keras.layers.BatchNormalization()(layer_2)

                layer_3 = tf.keras.layers.ConvLSTM2D(64, 3, return_sequences=False)(batch_norm_2)
                batch_norm_3 = tf.keras.layers.BatchNormalization()(layer_3)
                flatten = tf.keras.layers.Flatten()(batch_norm_3)

                self.v_preds = tf.keras.layers.Dense(units=1, activation='sigmoid')(flatten)

            self.act_stochastic = tf.random.categorical(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = self.act_probs

            self.scope = tf.get_variable_scope().name

    def act(self, obs, stochastic=True):
        if stochastic:
            return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs})
        else:
            return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


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
            c_h1 = layers.fully_connected(self.obs, self.hidden_size, trainable=self.trainable)
            c_out = layers.fully_connected(c_h1, 1, activation_fn=None, trainable=self.trainable)

        with tf.variable_scope('actor'):
            a_h1 = layers.fully_connected(self.obs, self.hidden_size, trainable=self.trainable)
            a_out = layers.fully_connected(a_h1, self.num_ac, activation_fn=None, trainable=self.trainable)

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