import tensorflow as tf
from GAIL.jupyter.Policy import ActorCritic


class PPO:
    def __init__(self, sess, ob_shape, ac_shape, lr, hidden_size, eps=0.2, v_coeff=0.5, ent_coeff=0.01, isRNN=True):
        self.sess = sess
        self.ob_shape = ob_shape
        self.ac_shape = ac_shape
        self.lr = lr
        self.hidden_size = hidden_size
        self.eps = eps
        self.v_coeff = v_coeff
        self.ent_coeff = ent_coeff
        self.isRNN = isRNN
        self._create_ppo_graph()

    def _create_ppo_graph(self):
        self.obs = tf.placeholder(dtype=tf.float32, shape=[None] + self.ob_shape, name='observation')
        self.acs = tf.placeholder(dtype=tf.float32, shape=[None] + self.ac_shape, name='action')
        self.returns = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.advs = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.pi = ActorCritic(self.sess, self.obs, self.acs, self.hidden_size, 'new_pi', trainable=True, isRNN=self.isRNN)
        self.old_pi = ActorCritic(self.sess, self.obs, self.acs, self.hidden_size, 'old_pi', trainable=False, isRNN=self.isRNN)

        self.pi_param = self.pi.params()
        self.old_pi_param = self.old_pi.params()

        with tf.name_scope('update_old_policy'):
            self.oldpi_update = [oldp.assign(p) for p, oldp in zip(self.pi_param, self.old_pi_param)]

        with tf.name_scope('loss'):
            ratio = tf.exp(self.pi.log_prob - self.old_pi.log_prob)
            surr = ratio * self.advs
            self.actor_loss = tf.reduce_mean(
                tf.minimum(surr, tf.clip_by_value(ratio, 1 - self.eps, 1 + self.eps) * self.advs))
            self.critic_loss = tf.reduce_mean(tf.square(self.returns - self.pi.value))

            self.loss = (- self.actor_loss - self.ent_coeff * tf.reduce_mean(self.pi.entropy)
                         + self.v_coeff * self.critic_loss)

            self.actor_summary = tf.summary.scalar('actor_loss', self.actor_loss)
            self.critic_summary = tf.summary.scalar('critic_loss', self.critic_loss)
            self.PPO_loss_summary = tf.summary.scalar('PPO_loss', self.loss)

        with tf.name_scope('train_op'):
            grads = tf.gradients(self.loss, self.pi_param)
            self.grads = list(zip(grads, self.pi_param))
            self.train_op = tf.train.AdamOptimizer(self.lr).apply_gradients(self.grads)

    def get_action(self, obs):
        return self.sess.run(self.pi.action, feed_dict={self.obs: obs})

    def get_value(self, obs):
        return self.sess.run(self.pi.value, feed_dict={self.obs: obs})

    def assign_old_pi(self):
        self.sess.run(self.oldpi_update)

    def update(self, writer, step, obs, acs, returns, advs):
        feed_dict = {self.obs: obs,
                     self.acs: acs,
                     self.returns: returns,
                     self.advs: advs
                     }
        op_list = [self.train_op, self.PPO_loss_summary]
        _, summ = self.sess.run(op_list, feed_dict=feed_dict)
        writer.add_summary(summ, step)
