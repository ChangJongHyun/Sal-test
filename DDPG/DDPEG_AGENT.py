import os

import numpy as np
import tensorflow as tf
from DDPG.DDPG_model import DDPG
from DDPG.utils import observation_dim, action_dim, ReplayBuffer
import gym

from custom_env.gym_my_env.envs.my_env import MyEnv

#### HYPER PARAMETERS ####
train_step = 50_000
test_step = 1000

minibatch_size = 100
pre_train_step = 3
max_step_per_episode = 600

mu = 0
theta = 0.15
sigma = 0.2


class Agent:

    def __init__(self, env, sess):
        self.env = env
        print("DDPG Agent")

        self.sess = sess
        self.action_dim = action_dim(env.action_space)  ### KH: for continuous action task
        self.obs_dim = observation_dim(env.observation_space)
        self.action_max = 1  ### KH: DDPG action bound
        self.action_min = -1  ### KH: DDPG action bound
        self.model = self.set_model()
        self.replay_buffer = ReplayBuffer(minibatch_size=minibatch_size)

    def set_model(self, load_path=None):
        # model can be q-table or q-network
        model = DDPG(self.sess, self.obs_dim, self.action_dim, self.action_max, self.action_min, isRNN=True,
                     load=load_path)
        return model

    def learn(self):
        print("Start Learn")

        global_step = 0
        episode_num = 0
        while global_step < train_step:

            episode_num += 1

            if episode_num >= 500:
                break

            step_in_ep = 0
            total_reward = 0
            done = False
            self.noise = np.zeros([2])

            obs, acs, target_video = self.env.reset()  # Reset environment
            # max_step_per_episode = 200,
            while not done and step_in_ep < max_step_per_episode and global_step < train_step:  ### KH: reset every 200 steps

                global_step += 1
                step_in_ep += 1
                action = self.get_action(obs, global_step)
                action = np.reshape(action, [2])
                obs_next, reward, done, _ = self.env.step(action)

                if done:
                    break
                else:
                    actor_loss, critic_loss = self.train_agent(obs, acs, [reward], obs_next, done, global_step)
                    with open(os.path.join('log_ddpg','log.txt'), "w") as f:
                        f.write("[ actor_loss: {}, critic_loss: {}]\n".format(actor_loss, critic_loss))

                    # GUI
                    # self.env.render()

                    obs = obs_next
                    total_reward += reward

            # self.model.reset_state()
            if episode_num is not 0 and episode_num % 10 == 0:
                self.model.save('./model/DDPG_ACTION', episode_num)

            print("[ target video: {}, train_ep: {}, total reward: {} ]".format(target_video, episode_num,
                                                                                total_reward))  ### KH: train result

    def test(self, global_step=0):
        print("Test step: {}".format(global_step))

        global_step = 0
        episode_num = 0
        total_reward = 0

        while global_step < test_step:

            episode_num += 1
            step_in_ep = 0

            obs, acs, target_video = self.env.reset()  # Reset environment
            total_reward = 0  ### KH: Added missing
            done = False

            while (
                    not done and step_in_ep < max_step_per_episode and global_step < test_step):  ### KH: reset every 200 steps

                global_step += 1
                step_in_ep += 1
                action = self.get_action(obs, global_step, False)
                obs_next, reward, done, acs_ = self.env.step()
                print(action, acs_)

                # GUI
                # self.env.render()

                obs = obs_next
                if reward is not None:
                    total_reward += reward

            print("[ target_video: {}, test_ep: {}, total reward: {} ]".format(target_video, episode_num,
                                                                               total_reward))  ### KH: test result

    def get_action(self, obs, global_step, train=True):
        # 최적의 액션 선택 + Exploration (Epsilon greedy)
        action = self.model.choose_action([obs])

        if train:
            scale = 1 - global_step / train_step
            self.noise = self.ou_noise(self.noise)
            action = action + self.noise * scale
            action = np.maximum(action, self.action_min)
            action = np.minimum(action, self.action_max)
        return action

    def train_agent(self, obs, action, reward, obs_next, done, step):

        if obs_next is None or obs is None:
            return

        obs = np.array(obs) / 255.
        obs_next = np.array(obs_next) / 255.

        # self.replay_buffer.add_to_memory((obs, action, reward, obs_next, done))
        #
        # if len(self.replay_buffer.replay_memory) < minibatch_size * pre_train_step:
        #     return None
        #
        # minibatch = self.replay_buffer.sample_from_memory()
        # s, a, r, ns, d = map(np.array, zip(*minibatch))
        # self.model.update(step, s, a, r, ns, d)

        actor_loss, critic_loss = self.model.update(step, [obs], [action], [reward], [obs_next], done)

        return actor_loss, critic_loss

    def ou_noise(self, x):
        return x + theta * (mu - x) + sigma * np.random.randn(2)


if __name__ == '__main__':
    # Load environmentZ
    print('Environment: my-env-v0')
    env = gym.make('my-env-v0')

    # Load agent
    print('Agent: ddpg')

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    #
    # if gpus:
    #     try:
    #         tf.config.experimental.set_virtual_device_configuration(gpus[0],
    #                                                                 [tf.config.experimental.VirtualDeviceConfiguration(
    #                                                                     memory_limit=1024 * 9)])
    #     except RuntimeError as e:
    #         pass

    sess = tf.Session()

    agent = Agent(env, sess)

    # start learning and testing
    agent.learn()
    # agent.test()
