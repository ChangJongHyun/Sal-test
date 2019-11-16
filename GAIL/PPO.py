import tensorflow as tf
import copy


class PPOTrain:
    def __init__(self, Policy, Old_Policy, gamma=0.95, clip_value=0.2, c_1=1, c_2=0.01):
        """
        :param Policy:
        :param Old_Policy:
        :param gamma:
        :param clip_value:
        :param c_1: parameter for value difference
        :param c_2: parameter for entropy bonus
        """

        self.Policy = Policy
        self.Old_Policy = Old_Policy
        self.gamma = gamma

        pi_trainable = self.Policy.get_trainable_variables()
        old_pi_trainable = self.Old_Policy.get_trainable_variables()

        # assign_operations for policy parameter values to old policy parameters
        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for v_old, v in zip(old_pi_trainable, pi_trainable):
                self.assign_ops.append(tf.assign(v_old, v))

        # inputs for train_op
        with tf.variable_scope('train_inp'):
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
            self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')

        act_probs = self.Policy.act_probs
        act_probs_old = self.Old_Policy.act_probs

        # probabilities of actions which agent took with policy
        act_probs = act_probs * tf.one_hot(indices=self.actions, depth=act_probs.shape[1])
        act_probs = tf.reduce_sum(act_probs, axis=1)

        # probabilities of actions which agent took with old policy
        act_probs_old = act_probs_old * tf.one_hot(indices=self.actions, depth=act_probs_old.shape[1])
        act_probs_old = tf.reduce_sum(act_probs_old, axis=1)

        with tf.variable_scope('loss'):
            # construct computation graph for loss_clip
            # ratios = tf.divide(act_probs, act_probs_old)
            ratios = tf.exp(tf.log(tf.clip_by_value(act_probs, 1e-10, 1.0))
                            - tf.log(tf.clip_by_value(act_probs_old, 1e-10, 1.0)))
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value)
            loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
            loss_clip = tf.reduce_mean(loss_clip)
            tf.summary.scalar('loss_clip', loss_clip)

            # construct computation graph for loss of entropy bonus
            entropy = -tf.reduce_sum(self.Policy.act_probs *
                                     tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)), axis=1)
            entropy = tf.reduce_mean(entropy, axis=0)  # mean of entropy of pi(obs)
            tf.summary.scalar('entropy', entropy)

            # construct computation graph for loss of value function
            v_preds = self.Policy.v_preds
            loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)
            loss_vf = tf.reduce_mean(loss_vf)
            tf.summary.scalar('value_difference', loss_vf)

            # construct computation graph for loss
            loss = loss_clip - c_1 * loss_vf + c_2 * entropy

            # minimize -loss == maximize loss
            loss = -loss
            tf.summary.scalar('total', loss)

        self.merged = tf.summary.merge_all()
        optimizer = tf.train.AdamOptimizer(learning_rate=5e-5, epsilon=1e-5)
        self.gradients = optimizer.compute_gradients(loss, var_list=pi_trainable)
        self.train_op = optimizer.minimize(loss, var_list=pi_trainable)

    def train(self, obs, actions, gaes, rewards, v_preds_next):
        tf.get_default_session().run(self.train_op, feed_dict={self.Policy.obs: obs,
                                                               self.Old_Policy.obs: obs,
                                                               self.actions: actions,
                                                               self.rewards: rewards,
                                                               self.v_preds_next: v_preds_next,
                                                               self.gaes: gaes})

    def get_summary(self, obs, actions, gaes, rewards, v_preds_next):
        return tf.get_default_session().run(self.merged, feed_dict={self.Policy.obs: obs,
                                                                    self.Old_Policy.obs: obs,
                                                                    self.actions: actions,
                                                                    self.rewards: rewards,
                                                                    self.v_preds_next: v_preds_next,
                                                                    self.gaes: gaes})

    def assign_policy_parameters(self):
        # assign policy parameter values to old policy parameters
        return tf.get_default_session().run(self.assign_ops)

    def get_gaes(self, rewards, v_preds, v_preds_next):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
        return gaes

    def get_grad(self, obs, actions, gaes, rewards, v_preds_next):
        return tf.get_default_session().run(self.gradients, feed_dict={self.Policy.obs: obs,
                                                                       self.Old_Policy.obs: obs,
                                                                       self.actions: actions,
                                                                       self.rewards: rewards,
                                                                       self.v_preds_next: v_preds_next,
                                                                       self.gaes: gaes})


class PPO:
    def __init__(self, sess, ob_shape, ac_shape, lr, hidden_size, eps=0.2, v_coeff=0.5, ent_coeff=0.01):
        self.sess = sess
        self.ob_shape = ob_shape
        self.ac_shape = ac_shape
        self.lr = lr
        self.hidden_size = hidden_size
        self.eps = eps
        self.v_coeff = v_coeff
        self.ent_coeff = ent_coeff

        self._create_ppo_graph()

    def _create_ppo_graph(self):
        self.obs = tf.placeholder(dtype=tf.float32, shape=[None] + self.ob_shape, name='observation')
        self.acs = tf.placeholder(dtype=tf.float32, shape=[None] + self.ac_shape, name='action')
        self.returns = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.advs = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.pi = ActorCritic(self.sess, self.obs, self.acs, self.hidden_size, 'new_pi', trainable=True)
        self.old_pi = ActorCritic(self.sess, self.obs, self.acs, self.hidden_size, 'old_pi', trainable=False)

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

    def update(self, obs, acs, returns, advs):
        feed_dict = {self.obs: obs,
                     self.acs: acs,
                     self.returns: returns,
                     self.advs: advs
                     }

        self.sess.run(self.train_op, feed_dict=feed_dict)