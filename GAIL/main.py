import os
import time

import numpy as np
import tensorflow as tf
import gym

from GAIL.Discriminator import Discriminator
from GAIL.PPO import PPO
import matplotlib.pyplot as plt

from custom_env.gym_my_env.envs import my_env

epochs = 10
num_envs = 16

ppo_hidden_size = 256
discriminator_hidden_size = 128
lr = 0.005  # 원래 2e-4
num_steps = 20
mini_batch_size = 5
ppo_epochs = 4
threshold_reward = -200

max_frames = 100000

test_rewards = []

envs = gym.make("my-env-v0")

early_stop = False
render = False


class Agent():
    def __init__(self, env, sess):
        self.env = env
        print("GAIL Agent")

        self.sess = sess
        self.action_dim = list(envs.action_space.shape)
        self.obs_dim = [None] + list(envs.observation_space.shape)
        self.action_max = 1
        self.action_min = -1
        self.ppo, self.discriminator = self.set_model()

    def set_model(self):
        ppo = PPO()
        discriminator = Discriminator()
        return ppo, discriminator

    def learn(self):
        # while frame_idx < max_frames and not elary_stop:
        #     i_update += 1
        #     values = []
        #     obs = []
        #     acs = []
        #     rewards = []
        #     masks = []
        #     entropy = 0
        #     ob, target_video = self.env.reset()
        #     for _ in range(num_steps):
        #         ac = self.ppo.get_action(ob)
        #         next_ob, _, done, _ = envs.step()
        #         reward = self.discriminator.get_reward(ob, ac)
        pass

    def test(self):
        pass

if __name__ == '__main__':
    ob_shape = [None] + list(envs.observation_space.shape)  # n_samples + (width, height, channels)
    ac_shape = list(envs.action_space.shape)
    print(ac_shape)
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                    [tf.config.experimental.VirtualDeviceConfiguration(
                                                                        memory_limit=1024 * 8)])
        except RuntimeError as e:
            pass
