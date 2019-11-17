import tensorflow as tf


class Discriminator:
    def __init__(self, sess, ob_shape, ac_shape, hidden_size, lr, name):
        self.sess = sess
        self.ob_shape = ob_shape
        self.ac_shape = ac_shape
        self.hidden_size = hidden_size
        self.lr = lr
        self.name = name

        self.ob_ac = tf.placeholder(dtype=tf.float32, shape=[None, ob_shape[0] + ac_shape[0]])

        self.ob = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.ob_shape))
        self.ac = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.ac_shape))

        with tf.variable_scope(name):
            self._build_network()

    def _build_network(self):
        with tf.variable_scope('discriminator'):
            # d_h1 = layers.fully_connected(self.ob_ac, self.hidden_size, activation_fn=tf.tanh)
            # d_h2 = layers.fully_connected(d_h1, self.hidden_size, activation_fn=tf.tanh)
            # d_out = layers.fully_connected(d_h2, 1, activation_fn=None)

            s_layer_1 = tf.keras.layers.ConvLSTM2D(64, 3, name='state_layer1', return_sequences=True)(self.ob)
            s_batch_norm_1 = tf.keras.layers.BatchNormalization(name='state_batch_norm1')(s_layer_1)

            s_layer_2 = tf.keras.layers.ConvLSTM2D(64, 3, name='state_layer2', return_sequences=True)(s_batch_norm_1)
            s_batch_norm_2 = tf.keras.layers.BatchNormalization(name='state_batch_norm2')(s_layer_2)

            s_layer_3 = tf.keras.layers.ConvLSTM2D(64, 3, name='state_layer3', return_sequences=False)(s_batch_norm_2)
            s_batch_norm_3 = tf.keras.layers.BatchNormalization(name='state_batch_norm3')(s_layer_3)

            s_flatten = tf.keras.layers.Flatten(name='state_flatten')(s_batch_norm_3)

            a_layer_1 = tf.keras.layers.Dense(20, activation='relu', name='action_layer1')(self.ac)
            a_layer_2 = tf.keras.layers.Dense(20, activation='relu', name='action_layer2')(a_layer_1)
            a_layer_3 = tf.keras.layers.Dense(20, activation='relu', name='action_layer2')(a_layer_2)

            concat = tf.keras.layers.concatenate([s_flatten, a_layer_3], name='s_a_concat')

            d_out = tf.keras.layers.Dense(1, name='prob')(concat)

        self.reward = - tf.squeeze(tf.log(tf.sigmoid(d_out)))

        expert_out, policy_out = tf.split(d_out, num_or_size_splits=2, axis=0)

        self.loss = (tf.losses.sigmoid_cross_entropy(tf.ones_like(policy_out), policy_out)
                     + tf.losses.sigmoid_cross_entropy(tf.zeros_like(expert_out), expert_out))

        with tf.name_scope('train_op'):
            grads = tf.gradients(self.loss, self.params())
            self.grads = list(zip(grads, self.params()))
            self.train_op = tf.train.AdamOptimizer(self.lr).apply_gradients(self.grads)

    def params(self):
        return tf.global_variables(self.name).copy()

    def get_reward(self, expert_ob, expert_ac):
        feed_dict = {self.ob: expert_ob,
                     self.ac: expert_ac}

        return self.sess.run(self.reward, feed_dict=feed_dict)

    def update(self, all_ob, all_ac):
        feed_dict = {self.ob: all_ob,
                     self.ac: all_ac}

        self.sess.run(self.train_op, feed_dict=feed_dict)
